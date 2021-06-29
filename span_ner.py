3  # !/usr/bin/env python
# coding=utf-8
"""
author: yunanzhang
model: span-level ner model
"""
import os
import torch
import numpy as np
import random
import time, sys, copy
from pathlib import Path
import datautils as utils
import argparse
from modules import BiLSTM_Span, Bert_Span, BiLSTM_CRF, Bert_Seq, Bert_Cls, Bert_MultiMrc
from data_reader import NerDataReader


def setup_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


setup_seed(1234)
sys.path.append('.')


class Trainer():
    def __init__(self, args, conf, model_type='span', exist_ckpt=None, evaluate=False):
        self.args = args
        self.conf = conf
        self.evaluate = evaluate
        self.model_type = model_type

        time_series = time.strftime('%y_%m_%d_%H_%M', time.localtime(time.time()))
        model_info = f'{model_type}'
        if model_type == 'span': model_info += f'_{conf["span_layer_type"]}'
        args.info = '_' + args.info if args.info else ''
        self.curr_ckpt_dir = args.ckpt_dir / f'{time_series}_{model_info}{args.info}'

        # TODO print and save args
        # print("=========args=========")
        # for k, v in vars(args).items():
        #     print(f'{k}: {str(v)[:100]}...')
        # print("======================")

        Model = {
            'span': Bert_Span,
            'seq': Bert_Seq,
            'cls': Bert_Cls,
            'multimrc': Bert_MultiMrc,
        }.get(model_type, None)

        self.model = Model(
            conf,
            tok2id={'char2id': args.char2id,
                    'tag2id': args.tag2id,
                    'ent2id': args.ent2id,
                    'id2char': args.id2char,
                    'id2tag': args.id2tag,
                    'id2ent': args.id2ent,
                    },
        )

        # for k, v in self.model.state_dict().items():
        #     print(k, v.shape, v.numel())

        use_gpu = torch.cuda.is_available()
        if use_gpu: self.model.cuda()
        print('total_params:', sum(p.numel() for p in self.model.parameters()))
        self.opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)

        if exist_ckpt is not None:
            self.model.load_state_dict(torch.load(exist_ckpt))
            print(f'load exist model ckpt success. {exist_ckpt} ')

        self.best_test_epo = 0
        self.best_test_step = 0
        self.best_test_f1 = -1
        self.best_dev_epo = 0
        self.best_dev_step = 0
        self.best_dev_f1 = -1
        self.test_f1_in_best_dev = -1

        self.metrics = []

        self.train_data_loader, self.test_data_loader, self.dev_data_loader = None, None, None
        self.curr_step = 0
        if getattr(args, 'train_dataset', None):
            self.train_data_loader = torch.utils.data.DataLoader(args.train_dataset, batch_size=args.batch_size, shuffle=True,
                                                                 collate_fn=args.datareader.get_batcher_fn(gpu=use_gpu, mode=model_type))
            print(f'train num: {len(args.train_dataset)}')
            self.train_num_steps = (len(args.train_dataset) - 1) // args.batch_size + 1
            self.total_num_steps = args.num_epochs * self.train_num_steps

        if getattr(args, 'dev_dataset', None):
            self.dev_data_loader = torch.utils.data.DataLoader(args.dev_dataset, batch_size=args.batch_size, shuffle=False,
                                                               collate_fn=args.datareader.get_batcher_fn(gpu=use_gpu, mode=model_type))
            print(f'dev num: {len(args.dev_dataset)}')
        if getattr(args, 'test_dataset', None):
            self.test_data_loader = torch.utils.data.DataLoader(args.test_dataset, batch_size=args.batch_size, shuffle=False,
                                                                collate_fn=args.datareader.get_batcher_fn(gpu=use_gpu, mode=model_type))
            print(f'test num: {len(args.test_dataset)}')

        if self.evaluate and getattr(args, 'infer_dataset', None):
            self.test_data_loader = torch.utils.data.DataLoader(args.infer_dataset, batch_size=args.batch_size, shuffle=False,
                                                                collate_fn=args.datareader.get_batcher_fn(gpu=use_gpu, mode=model_type))
            print(f'infer num: {len(args.infer_dataset)}')

        self.run_epo = {
            'span': self.run_epo_span,
            'seq': self.run_epo_seq,
            'cls': self.run_epo_cls,
            'multimrc': self.run_epo_multimrc,
        }.get(model_type, None)
        self.use_gpu = use_gpu
        self.run()

    def run(self):
        if self.evaluate:
            test_f1, test_ef1 = self.run_epo(1, mode='test')
            print(test_f1, test_ef1)
            return

        for epo in range(1, self.args.num_epochs + 1):
            train_f1, train_ef1 = self.run_epo(epo, mode='train')
            if self.dev_data_loader is not None and self.test_data_loader is not None:  # 没有验证集只有测试集
                self.eval_dev(epo)
            elif self.dev_data_loader is None and self.test_data_loader is not None:  # 有验证集有测试集
                self.eval_test(epo)
            else:
                raise NotImplementedError

    def eval_dev(self, epo):
        dev_f1, dev_ef1 = self.run_epo(epo, mode='dev')
        test_f1, test_ef1 = self.run_epo(epo, mode='test')
        if dev_ef1 > self.best_dev_f1:
            Path(self.curr_ckpt_dir / f'best_dev_model_{self.best_dev_epo}_{self.best_dev_step}.ckpt').unlink(missing_ok=True)  # 删除已有
            self.best_dev_epo, self.best_dev_step, self.best_dev_f1, self.test_f1_in_best_dev = epo, self.curr_step, dev_ef1, test_ef1
            torch.save(self.model.state_dict(), self.curr_ckpt_dir / f'best_dev_model_{self.best_dev_epo}_{self.best_dev_step}.ckpt')
        print(f'best_dev_f1: {self.best_dev_f1} best_test_epo: {self.best_dev_epo}_{self.best_dev_step} test_f1_in_best_dev: {self.test_f1_in_best_dev}')

    def eval_test(self, epo):
        test_f1, test_ef1 = self.run_epo(epo, mode='test')
        # if epo % 5 == 0:
        #     self.best_test_epo, self.best_test_step, self.best_test_f1 = epo, self.curr_step, test_ef1
        #     torch.save(self.model.state_dict(), self.curr_ckpt_dir / f'best_test_model_{self.best_test_epo}_{self.best_test_step}.ckpt')
        if test_ef1 > self.best_test_f1:
            Path(self.curr_ckpt_dir / f'best_test_model_{self.best_test_epo}_{self.best_test_step}.ckpt').unlink(missing_ok=True)  # 删除已有
            self.best_test_epo, self.best_test_step, self.best_test_f1 = epo, self.curr_step, test_ef1
            torch.save(self.model.state_dict(), self.curr_ckpt_dir / f'best_test_model_{self.best_test_epo}_{self.best_test_step}.ckpt')
        print(f'best_test_f1: {self.best_test_f1} best_test_epo: {self.best_test_epo}_{self.best_test_step}')

    def after_per_step(self, epo, mode):
        if mode == 'train':
            self.curr_step += 1
            if self.args.eval_every_step != 0 and self.curr_step >= self.args.eval_after_step and self.curr_step % self.args.eval_every_step == 0:
                print('')
                if self.dev_data_loader is not None and self.test_data_loader is not None:  # 没有验证集只有测试集
                    self.eval_dev(epo)
                    print(f'step: {self.curr_step}, best_dev_f1: {self.best_dev_f1} best_dev_epo: {self.best_dev_epo}_{self.best_dev_step} test_f1_in_best_dev: {self.test_f1_in_best_dev}')
                elif self.dev_data_loader is None and self.test_data_loader is not None:  # 有验证集有测试集
                    self.eval_test(epo)
                    print(f'step: {self.curr_step}, best_test_f1: {self.best_test_f1} best_test_epo: {self.best_test_epo}_{self.best_test_step}')
                else:
                    raise NotImplementedError
                self.model.train()

    def run_epo_cls(self, epo, mode='train'):
        epo_start_time = time.time()
        print(f'\n{mode}...{"=" * 60}')
        data_loader = {
            'train': self.train_data_loader,
            'test': self.test_data_loader,
            'dev': self.dev_data_loader,
        }.get(mode, None)

        if data_loader is None:
            return None, None

        self.model.train() if mode == 'train' else self.model.eval()

        total_steps = len(data_loader)

        total_f1 = []
        total_acc = []
        total_loss = []

        for num_steps, inputs in enumerate(data_loader, start=1):
            step_start_time = time.time()
            batch_bert_chars = inputs['input_ids']
            batch_seq_len = inputs['seq_len']
            batch_cls_tgt = inputs['cls_tgt']
            batch_bert_token_type_ids = inputs['bert_token_type_ids']
            batch_bert_attention_mask = inputs['bert_attention_mask']
            batch_ner_exm = inputs['batch_ner_exm']

            if mode == 'train':
                self.opt.zero_grad()

            model_start_time = time.time()
            if mode == 'train':
                cls_loss, cls_pred_prob, f1, acc = self.model(
                    batch_bert_chars,
                    batch_bert_token_type_ids,
                    batch_bert_attention_mask,
                    batch_seq_len, batch_cls_tgt)

            else:
                with torch.no_grad():  # 不计算梯度 更进一步节约内存 与train() eval()无关
                    cls_loss, cls_pred_prob, f1, acc = self.model(
                        batch_bert_chars,
                        batch_bert_token_type_ids,
                        batch_bert_attention_mask,
                        batch_seq_len, batch_cls_tgt)

            if mode == 'train':
                cls_loss.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.)
                self.opt.step()
            # print('model_time', time.time() - model_start_time)
            total_loss.append(float(cls_loss))
            total_f1.append(float(f1))
            total_acc.append(float(acc))
            cls_pred_prob = cls_pred_prob.cpu().detach().numpy()
            cls_pred = (cls_pred_prob >= 0.5).astype('int')

            for exm, _cls_pred, _cls_pred_prob in zip(batch_ner_exm, cls_pred.tolist(), cls_pred_prob.tolist()):
                exm.cls_pred = [_cls_pred, _cls_pred_prob]

            print(f'\repo:{epo} step:{num_steps}/{total_steps} '
                  f'cur_loss:{cls_loss:.3f} epo_loss:{np.mean(total_loss):.3f} '
                  f'cur_f1:{f1:.3f} epo_f1:{np.mean(total_f1):.3f} '
                  f'cur_acc:{f1:.3f} epo_acc:{np.mean(total_acc):.3f} '
                  f'sec/step:{time.time() - step_start_time:.2f}',
                  end=f'{os.linesep if num_steps == total_steps else ""}',
                  )

            self.after_per_step(epo, mode)

        # metrics
        exm_lst = data_loader.dataset.instances
        utils.NerExample.save_to_jsonl(exm_lst, self.curr_ckpt_dir / f'{mode}_{epo}_{self.curr_step}_exm_lst.jsonl', for_human_read=True, external_attrs=['source_id', 'cls_pred', 'cls_tgt'])

        glod_cls_tgt = np.array([exm.cls_tgt for exm in exm_lst]).astype('int')
        pred_cls_tgt = np.array([exm.cls_pred[0] for exm in exm_lst])

        acc = np.sum(glod_cls_tgt == pred_cls_tgt) / len(glod_cls_tgt)
        tp = np.sum(glod_cls_tgt * pred_cls_tgt)
        num_glod = np.sum(glod_cls_tgt)
        num_pred = np.sum(pred_cls_tgt)
        p = tp / (num_pred + 1e-10)
        r = tp / (num_glod + 1e-10)
        f = 2 * tp / (num_glod + num_pred + 1e-10)

        print(f'{mode} p-r-f-acc: {p:.4f}-{r:.4f}-{f:.4f}-{acc:.4f}')
        metric_info = {'mode': mode, 'epo': epo, 'step': self.curr_step, 'info': {'loss': np.mean(total_loss)}}
        metric_info['info'].update({'p': p, 'r': r, 'f': f, 'acc': acc})
        self.metrics.append(metric_info)
        self.metrics.sort(key=lambda e: e['epo'])
        self.metrics.sort(key=lambda e: e['mode'])
        utils.save_jsonl(self.metrics, (self.curr_ckpt_dir / 'metrics/metrics.json'), verbose=False)
        return 0., f

    def run_epo_span(self, epo, mode='train'):
        epo_start_time = time.time()
        print(f'\n{mode}...{"=" * 60}')
        data_loader = {
            'train': self.train_data_loader,
            'test': self.test_data_loader,
            'dev': self.dev_data_loader,
        }.get(mode, None)

        if data_loader is None:
            return None, None

        self.model.train() if mode == 'train' else self.model.eval()

        total_steps = len(data_loader)

        total_f1 = []
        total_loss = []

        for num_steps, inputs in enumerate(data_loader, start=1):

            step_start_time = time.time()
            batch_bert_chars = inputs['input_ids']
            batch_seq_len = inputs['seq_len']
            batch_span_ner_tgt_lst = inputs['span_ner_tgt_lst']
            # batch_neg_span_mask = inputs['neg_span_mask']
            batch_bert_token_type_ids = inputs['bert_token_type_ids']
            batch_bert_attention_mask = inputs['bert_attention_mask']
            batch_ner_exm = inputs['batch_ner_exm']
            # mask_matrix = inputs['mask_matrix']
            # print('batch_span_ner_tgt_lst', batch_span_ner_tgt_lst.shape)
            # print('batch_neg_span_mask', batch_neg_span_mask.shape)
            # sample_neg_start_time = time.time()
            # if self.args.negative_sample:  # 降采样负样本-负采样
            # pass
            # batch_span_ner_tgt_lst = get_ner_span_random_sample_lst_of_batch(batch_span_ner_tgt_lst, batch_seq_len, ratio=0.35)
            # batch_span_ner_tgt_lst = torch.LongTensor(batch_span_ner_tgt_lst).cuda()

            # batch_span_ner_tgt_lst_mask = get_span_lst_neg_sample_mask(batch_span_ner_tgt_lst, batch_seq_len, ratio=0.35)
            # batch_span_ner_tgt_lst_mask = torch.FloatTensor(batch_span_ner_tgt_lst_mask).cuda()
            # print(batch_span_ner_tgt_lst.tolist())
            # print('sample_neg_time', time.time() - sample_neg_start_time)

            if mode == 'train':
                self.opt.zero_grad()

            model_start_time = time.time()
            if mode == 'train':
                span_loss, span_ner_mat_tensor, batch_span_ner_pred_lst, conj_dot_product_score, f1 = self.model(
                    batch_bert_chars,
                    batch_bert_token_type_ids,
                    batch_bert_attention_mask,
                    batch_seq_len, batch_span_ner_tgt_lst)

            else:
                with torch.no_grad():  # 不计算梯度 更进一步节约内存 与train() eval()无关
                    span_loss, span_ner_mat_tensor, batch_span_ner_pred_lst, conj_dot_product_score, f1 = self.model(
                        batch_bert_chars,
                        batch_bert_token_type_ids,
                        batch_bert_attention_mask,
                        batch_seq_len, batch_span_ner_tgt_lst)

            if mode == 'train':
                span_loss.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.)
                self.opt.step()
            # print('model_time', time.time() - model_start_time)
            total_loss.append(float(span_loss))
            total_f1.append(float(f1))

            if mode in ['test', 'dev'] or epo > 0:
                # exm_proc_start_time = time.time()
                # batch中切分每个的span_level_lst
                batch_span_lst_len = [(l + 1) * l // 2 for l in batch_seq_len.tolist()]
                batch_span_ner_pred_lst = utils.split_list(batch_span_ner_pred_lst.cpu().detach().numpy(), batch_span_lst_len)
                for exm, length, span_ner_pred_lst in zip(batch_ner_exm, batch_seq_len, batch_span_ner_pred_lst):
                    exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst(span_ner_pred_lst, length, self.args.id2ent)  # softmax
                    # exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst_sigmoid(span_ner_pred_lst, length, self.args.id2ent)  # sigmoid
                    # print(exm.pred_ent_dct)
                # print('exm_proc', time.time() - exm_proc_start_time)
            else:
                for exm in batch_ner_exm:
                    exm.pred_ent_dct = {}

            print(f'\repo:{epo} step:{num_steps}/{total_steps} '
                  f'cur_loss:{span_loss:.3f} epo_loss:{np.mean(total_loss):.3f} '
                  f'cur_f1:{f1:.3f} epo_f1:{np.mean(total_f1):.3f} '
                  f'sec/step:{time.time() - step_start_time:.2f}',
                  end=f'{os.linesep if num_steps == total_steps else ""}',
                  )

            self.after_per_step(epo, mode)

            # 输出连接分数
            # print()
            # out = []
            # conj_dot_product_score = conj_dot_product_score.tolist()
            # for bdx, exm in enumerate(batch_ner_exm):modu
            #     conj_res = exm.get_conj_info(conj_dot_product_score[bdx])
            #     out.append(conj_res)
            #     out.append(exm.get_ent_lst(for_human=True))
            #     out.append(exm.get_pred_ent_lst(for_human=True))
            #     out.append(exm.text)
            #     out.append(conj_dot_product_score[bdx])
            # print(*out, sep='\n')
            # input('请输入任意键继续')
            # # exit(0)

        # metrics
        # ==数据特殊处理
        def yb_ent_convert(ent_type):
            if ent_type.startswith('指标-'):
                ent_type = '指标'
            return ent_type

        exm_lst = data_loader.dataset.instances
        tmp_exm_lst = copy.deepcopy(exm_lst)
        for exm in tmp_exm_lst:
            exm.filter_ent_by_startswith('指标', mode='keep')
            # exm.ent_type_convert(yb_ent_convert)  # yb
        # ==数据特殊处理
        # tmp_exm_lst = exm_lst

        print('flat_metric:')
        p, r, f, *_ = utils.NerExample.eval(tmp_exm_lst, verbose=False, use_flat_pred_ent_dct=True)
        print(f'{mode} p-r-f: {p:.4f}-{r:.4f}-{f:.4f}')
        metric_info = {'mode': mode, 'epo': epo, 'step': self.curr_step, 'info': {'loss': np.mean(total_loss)}}
        metric_info['info'].update({'p': p, 'r': r, 'f': f, })
        if epo > 0:
            print('ori_metric:')
            ep, er, ef, *_ = utils.NerExample.eval(tmp_exm_lst, verbose=False, use_flat_pred_ent_dct=False)
            metric_info['info'].update({'ep': ep, 'er': er, 'ef': ef, })
            utils.NerExample.save_to_jsonl(exm_lst, self.curr_ckpt_dir / f'{mode}_{epo}_{self.curr_step}_exm_lst.jsonl', for_human_read=True, external_attrs=['cls_tgt', 'cls_tgt', 'source_id'])
        self.metrics.append(metric_info)
        self.metrics.sort(key=lambda e: e['epo'])
        self.metrics.sort(key=lambda e: e['mode'])
        utils.save_jsonl(self.metrics, (self.curr_ckpt_dir / 'metrics/metrics.json'), verbose=False)
        return 0., f

    def run_epo_seq(self, epo, mode='train'):
        if not hasattr(self, 'crf_diff_opt'):
            crf_parameters = self.model.crf_layer.parameters()
            crf_parameters_idset = set(map(id, crf_parameters))
            other_parameters = filter(lambda p: id(p) not in crf_parameters_idset, self.model.parameters())
            self.crf_diff_opt = torch.optim.AdamW([
                {'params': other_parameters},
                {'params': crf_parameters, 'lr': 1e-4}
            ], lr=self.args.lr)
        print(f'\n{mode}...{"=" * 60}')
        data_loader = {
            'train': self.train_data_loader,
            'test': self.test_data_loader,
            'dev': self.dev_data_loader,
        }.get(mode, None)

        if data_loader is None:
            return None, None

        self.model.train() if mode == 'train' else self.model.eval()

        total_steps = len(data_loader)
        total_loss = []

        for num_steps, inputs in enumerate(data_loader, start=1):
            step_start_time = time.time()
            batch_bert_chars = inputs['input_ids']
            batch_seq_len = inputs['seq_len']
            batch_tags = inputs['tag_ids']
            batch_bert_token_type_ids = inputs['bert_token_type_ids']
            batch_bert_attention_mask = inputs['bert_attention_mask']
            batch_ner_exm = inputs['batch_ner_exm']

            if mode == 'train':
                self.opt.zero_grad()

            # model_start_time = time.time()
            if mode == 'train':
                crf_loss, emission, decode_ids = self.model(batch_bert_chars,
                                                            batch_bert_token_type_ids,
                                                            batch_bert_attention_mask,
                                                            batch_tags, batch_seq_len)

                # decode_ids: List
            else:
                with torch.no_grad():  # 不计算梯度 更进一步节约内存 与train() eval()无关
                    crf_loss, emission, decode_ids = self.model(batch_bert_chars,
                                                                batch_bert_token_type_ids,
                                                                batch_bert_attention_mask,
                                                                batch_tags, batch_seq_len)
            if mode == 'train':
                crf_loss.backward()
                self.opt.step()
            # print('model_time', time.time() - model_start_time)

            total_loss.append(float(crf_loss))

            for exm, decode_ids_ in zip(batch_ner_exm, decode_ids):
                tag_lst = [self.args.id2tag[tag_id] for tag_id in decode_ids_]
                # assert len(tag_lst) == len(exm.char_lst)
                pred_ent_dct, _ = utils.NerExample.extract_entity_by_tags(tag_lst)
                for k, v_lst in pred_ent_dct.items():
                    for e in v_lst:
                        e.append(1.)  # 假设概率为1
                exm.pred_ent_dct = pred_ent_dct

            print(f'\repo:{epo} step:{num_steps}/{total_steps} '
                  f'cur_loss:{crf_loss:.3f} epo_loss:{np.mean(total_loss):.3f} '
                  f'sec/step:{time.time() - step_start_time:.2f}',
                  end=f'{os.linesep if num_steps == total_steps else ""}',
                  )

            self.after_per_step(epo, mode)

        # metrics
        # ==数据特殊处理
        def yb_ent_convert(ent_type):
            if ent_type.startswith('指标-'):
                ent_type = '指标'
            return ent_type

        exm_lst = data_loader.dataset.instances
        tmp_exm_lst = copy.deepcopy(exm_lst)
        for exm in tmp_exm_lst:
            exm.filter_ent_by_startswith('指标', mode='keep')
            # exm.ent_type_convert(yb_ent_convert)  # yb
        # ==数据特殊处理
        p, r, f, *_ = utils.NerExample.eval(tmp_exm_lst, verbose=True, use_flat_pred_ent_dct=True)
        print(f'{mode} p-r-f: {p:.6f}-{r:.6f}-{f:.6f}')
        metric_info = {'mode': mode, 'epo': epo, 'step': self.curr_step, 'info': {'loss': np.mean(total_loss)}}
        metric_info['info'].update({'p': p, 'r': r, 'f': f})
        utils.NerExample.save_to_jsonl(exm_lst, self.curr_ckpt_dir / f'{mode}_{epo}_{self.curr_step}_exm_lst.jsonl', for_human_read=True, external_attrs=['cls_tgt', 'cls_tgt', 'source_id'])
        self.metrics.append(metric_info)
        self.metrics.sort(key=lambda e: e['epo'])
        self.metrics.sort(key=lambda e: e['mode'])
        utils.save_jsonl(self.metrics, (self.curr_ckpt_dir / 'metrics/metrics.json'), verbose=False)
        return 0, f

    def run_epo_multimrc(self, epo, mode='train'):
        epo_start_time = time.time()
        print(f'\n{mode}...{"=" * 60}')
        data_loader = {
            'train': self.train_data_loader,
            'test': self.test_data_loader,
            'dev': self.dev_data_loader,
        }.get(mode, None)

        if data_loader is None:
            return None, None

        self.model.train() if mode == 'train' else self.model.eval()

        total_steps = len(data_loader)

        # total_f1 = []
        total_loss = []

        for num_steps, inputs in enumerate(data_loader, start=1):

            step_start_time = time.time()
            batch_bert_chars = inputs['input_ids']
            batch_token_type_id = inputs['token_type_id']
            batch_bert_attention_mask = inputs['attn_mask']
            batch_query_mask = inputs['query_mask']
            batch_answers = inputs['answers']
            batch_ner_exm = inputs['batch_ner_exm']

            if mode == 'train':
                self.opt.zero_grad()

            model_start_time = time.time()
            if mode == 'train':
                mrc_loss, batch_prob, batch_context_start_index = self.model(
                    batch_bert_chars,
                    batch_token_type_id,
                    batch_bert_attention_mask,
                    batch_query_mask, batch_answers)

            else:
                with torch.no_grad():  # 不计算梯度 更进一步节约内存 与train() eval()无关
                    mrc_loss, batch_prob, batch_context_start_index = self.model(
                        batch_bert_chars,
                        batch_token_type_id,
                        batch_bert_attention_mask,
                        batch_query_mask, batch_answers)

            if mode == 'train':
                mrc_loss.backward()
                # torch.nn.utils.clip_grad_norm(self.model.parameters(), 1.)
                self.opt.step()
            # print('model_time', time.time() - model_start_time)
            total_loss.append(float(mrc_loss))

            # 整理预测结果
            # all_atps = ['上游', '下游', '业务', '产品', '公司', '区域', '品牌', '客户', '市场', '年龄',
            #             '性别', '时间', '机构', '来源', '渠道', '行业', '项目', ]
            all_atps = self.args.datareader.all_atps
            num_atps = len(all_atps)
            batch_prob = batch_prob.cpu().detach().numpy()
            batch_context_start_index = batch_context_start_index.cpu().detach().numpy().tolist()
            for i, (exm, context_start_index) in enumerate(zip(batch_ner_exm, batch_context_start_index)):
                atp_res = []
                prob = batch_prob[i * num_atps: (i + 1) * num_atps]
                for prob_, atp_name in zip(prob, all_atps):
                    # prob_ [len,2]
                    if sum(np.max(prob_, axis=0) >= 0.5) == 2:  # start和end都分别存在prob大于0.5的
                        pos = np.argmax(prob_, axis=0)  # [2]
                        start, end = pos.tolist()
                        if start <= end:
                            atp_res.append([atp_name, start - context_start_index, end - context_start_index + 1])
                pred_ent_dct = {f'属性-{e[0]}': [[e[1], e[2], 1.]] for e in atp_res}  # prob = 1.
                exm.pred_ent_dct = pred_ent_dct
                # print(exm.pred_ent_dct)
                # print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct','source_id']))

            print(f'\repo:{epo} step:{num_steps}/{total_steps} '
                  f'cur_loss:{mrc_loss:.3f} epo_loss:{np.mean(total_loss):.3f} '
                  f'sec/step:{time.time() - step_start_time:.2f}',
                  end=f'{os.linesep if num_steps == total_steps else ""}',
                  )

            self.after_per_step(epo, mode)

        # metrics
        exm_lst = data_loader.dataset.instances
        utils.NerExample.save_to_jsonl(exm_lst, self.curr_ckpt_dir / f'{mode}_{epo}_{self.curr_step}_exm_lst.jsonl', for_human_read=True, external_attrs=['itp2atp_dct', 'source_id'])
        tmp_exm_lst = copy.deepcopy(exm_lst)
        for exm in tmp_exm_lst:
            exm.ent_dct = {ent_type: pos_lst for ent_type, pos_lst in exm.ent_dct.items() if not ent_type.startswith('指标')}  # 只保留属性实体来验证
        p, r, f, *_ = utils.NerExample.eval(tmp_exm_lst, verbose=True, use_flat_pred_ent_dct=True)
        print(f'{mode} p-r-f: {p:.4f}-{r:.4f}-{f:.4f}')
        metric_info = {'mode': mode, 'epo': epo, 'step': self.curr_step, 'info': {'loss': np.mean(total_loss)}}
        metric_info['info'].update({'p': p, 'r': r, 'f': f})
        self.metrics.append(metric_info)
        self.metrics.sort(key=lambda e: e['epo'])
        self.metrics.sort(key=lambda e: e['mode'])
        utils.save_jsonl(self.metrics, (self.curr_ckpt_dir / 'metrics/metrics.json'), verbose=False)
        return 0., f


def get_ner_span_random_sample_lst_of_batch(ner_span_lst_of_batch, batch_seq_len, ratio=0.35):
    # 负采样
    ner_span_lst_of_batch = ner_span_lst_of_batch.tolist()
    batch_seq_len = batch_seq_len.tolist()
    batch_num_span = list(map(lambda x: int(x * (x + 1) / 2), batch_seq_len))
    num_span_indices = np.cumsum(batch_num_span).tolist()
    split_lst = [ner_span_lst_of_batch[i:j] for i, j in zip([0] + num_span_indices, num_span_indices)]
    # assert len(split_lst) == len(batch_seq_len)  # batch_size应能对齐
    # assert split_lst[-1][-1] == ner_span_lst_of_batch[-1]  # 末尾应能对齐
    for span_lst, length in zip(split_lst, batch_seq_len):
        neg_indices = [idx for idx, val in enumerate(span_lst) if val == 1]  # 1:O
        # num_neg = len(neg_indices)
        num_sample_neg = int(length * ratio + 0.5)  # 四舍五入
        # print('!'*20)
        # print(span_lst)
        # print(length)
        # print(num_sample_neg)
        neg_sample_indices = random.sample(neg_indices, num_sample_neg)  # 四舍五入  去掉哪些负样本
        neg_sample_indices = set(neg_sample_indices)
        for idx in neg_indices:
            if idx not in neg_sample_indices:
                span_lst[idx] = 0  # 0:<pad>
        # print(span_lst)
    ner_span_random_sample_mask = sum(split_lst, [])
    return ner_span_random_sample_mask


def get_span_lst_neg_sample_mask(batch_span_lst, batch_seq_len, ratio=0.35):
    batch_span_lst  # [len, num_class(one_hot)]
    batch_span_lst = (torch.sum(batch_span_lst, dim=-1) != 0).float()  # [len]

    batch_span_lst_len = [(l + 1) * l // 2 for l in batch_seq_len.tolist()]
    batch_span_lst = utils.split_list(batch_span_lst, batch_span_lst_len)

    # batch_mask_on_neg = 1. - batch_mask

    # ratio 保留句子长度*ratio 数量的负样本
    # batch_span_lst = batch_span_lst.tolist()
    #
    # batch_span_lst_len = [(l + 1) * l // 2 for l in batch_seq_len.tolist()]
    # batch_span_lst = utils.split_list(batch_span_lst, batch_span_lst_len)

    batch_mask = []
    for span_ner_lst, length in zip(batch_span_lst, batch_seq_len):
        num_sample_neg = int(length * ratio + 0.5)  # 四舍五入 采样的负样本个数

        neg_indices = [idx for idx, val in enumerate(span_ner_lst) if val == 0.]  # onehot = all 0
        sampled_neg_indices = set(random.sample(neg_indices, num_sample_neg))  # 采样哪些负样本

        mask = [1] * len(span_ner_lst)
        for idx in neg_indices:
            if idx not in sampled_neg_indices:
                mask[idx] = 0  #
        batch_mask.append(mask)
    batch_mask = sum(batch_mask, [])
    return batch_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser("span")
    parser.add_argument('--info', help='information to distinguish model.', default='')
    parser.add_argument('--gpu', help='information to distinguish model.', default='6')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用CPU设为'-1'
    print(f'gpu_id:{args.gpu}')

    conf = {
        # 'lr': 1e-3,
        'dropout_rate': 0.2,
        # 'dropout_rate': 0.5,
        # 'batch_size': 2,
        # 'test_batch_size': 4,
        # 'max_len': 256,
        # 'embed_size': 256,
        # 'rnn_num_layers': 1,
        # 'rnn_hidden_size': 256,
        # 'use_bert': True,
        # 'bert_lr': 2e-5,
        # 'bert_model_dir': '/home/zyn/huggingface_model_resource/chinese-roberta-wwm-ext-large',
        'bert_model_dir': '/home/zyn/huggingface_model_resource/chinese-bert-wwm-ext',
        # 'bert_model_dir': '/home/zyn/huggingface_model_resource/bert-base-uncased',
        # 'bert_model_dir': '/home/zyn/huggingface_model_resource/bert-large-cased',
        # 'bert_model_dir': '/home/zyn/huggingface_model_resource/bert-base-chinese',
        # 'span_layer_type':'biaffine',d
        # 'span_layer_type': 'tencent',
        # 'span_layer_type': 'self_attn',
        'span_layer_type': 'self_attn_mask_mean',
        'span_model_negsample_rate': 0  # 0.3  # 0不进行负采样 非0 按比例进行负采样
    }
    args.ckpt_dir = Path('span_ner_modules/ckpt')
    args.eval_every_step = 0
    args.eval_after_step = 0

    # cjhx
    # args.train_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/cjhx/train.jsonl'), mode='span')
    # args.test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/cjhx/test_out.jsonl'), mode='span')

    # yanbao
    datareader = NerDataReader(conf['bert_model_dir'], 512,
                               # ent_file='data/yanbao/sent/ent_lst_addO.txt',
                               # ent_file='data/yanbao/sent/ent_lst_addO_remove_atp.txt',
                               # ent_file='data/yanbao/sent/full_ent_lst.txt',
                               ent_file='data/yanbao/sent/full_ent_lst_remove_atp.txt',
                               # ent_file='data/yanbao/sent/ent_lst_onehot.txt',
                               )

    # instances = utils.NerExample.load_from_jsonl('data/yanbao/sent/sent_exm_final_combine.jsonl')
    # instances = utils.NerExample.load_from_jsonl('data/yanbao/sent/sent_exm_final.jsonl', external_attrs=['source_id'])
    # instances = utils.NerExample.load_from_jsonl('data/yanbao/sent/sent_exm_final.jsonl', external_attrs=['source_id'])
    # instances = utils.NerExample.load_from_jsonl('data/yanbao/sent_m/sent_exm_final.jsonl', external_attrs=['source_id'])
    # instances = utils.NerExample.load_from_jsonl('data/yanbao/sent_mzb/sent_exm_final.jsonl', external_attrs=['source_id'])
    # instances = utils.NerExample.load_from_jsonl('data/yanbao/sent_szb/sent_exm_final.jsonl', external_attrs=['source_id'])
    # random.shuffle(instances)
    # total_num = len(instances)
    # train_num = int(0.9 * total_num)
    # utils.NerExample.save_to_jsonl(instances[:train_num], 'data/yanbao/sent_szb/sent_exm_final_train.jsonl', for_human_read=True, external_attrs=['source_id'], overwrite=False)
    # utils.NerExample.save_to_jsonl(instances[train_num:], 'data/yanbao/sent_szb/sent_exm_final_test.jsonl', for_human_read=True, external_attrs=['source_id'], overwrite=False)
    # args.train_dataset = datareader.build_dataset(instances[:train_num], mode='cls')
    # args.test_dataset = datareader.build_dataset(instances[train_num:], mode='cls')

    # 预测的数据集下一步pipeline
    # instances = utils.NerExample.load_from_jsonl('data/yanbao/sent/test_1_0_exm_lst.jsonl', external_attrs=['source_id', 'cls_tgt'])
    # instances = [exm for exm in instances if exm.cls_tgt[0] == 1]
    # utils.NerExample.save_to_jsonl(instances, 'data/yanbao/sent/test_1_0_exm_lst_cls_pos.jsonl', for_human_read=True, external_attrs=['source_id', 'cls_tgt'])
    # exit(0)

    # instances = utils.NerExample.load_from_jsonl('data/yanbao/sent/test_1_0_exm_lst_cls_pos.jsonl', external_attrs=['source_id', 'cls_tgt'])
    # # instances = utils.NerExample.load_from_jsonl('data/yanbao/sent/old/sent_exm_final_combine.jsonl', external_attrs=['source_id', 'cls_tgt'])
    # total_num = len(instances)
    # train_num = int(0.9 * total_num)

    # train_data=instances[:train_num]
    # test_data = instances[train_num:]
    # utils.NerExample.save_to_jsonl(train_data, 'data/yanbao/sent/after_cls/yanbao_yimei_train.jsonl', for_human_read=True,  external_attrs=['source_id', 'cls_tgt'])
    # utils.NerExample.save_to_jsonl(test_data, 'data/yanbao/sent/after_cls/yanbao_yimei_test.jsonl', for_human_read=True,  external_attrs=['source_id', 'cls_tgt'])
    # utils.NerExample.stats(train_data)
    # utils.NerExample.stats(test_data)

    # name2id = {e['elem_name']:e['elem_id'] for e in utils.load_json('data/yanbao/sent/after_cls/elem_dict.json')}
    # def yb_ent_convert(ent_type):
    #     return name2id[ent_type]
    #
    # for exm in train_data:
    #     exm.ent_type_convert(yb_ent_convert)
    # for exm in test_data:
    #     exm.ent_type_convert(yb_ent_convert)
    # utils.NerExample.save_to_col_format_file(train_data, 'data/yanbao/sent/after_cls/yanbao_yimei_train.txt')
    # utils.NerExample.save_to_col_format_file(test_data, 'data/yanbao/sent/after_cls/yanbao_yimei_test.txt')
    # exit(0)

    # args.train_dataset = datareader.build_dataset(instances[:train_num], mode='seq')
    # args.test_dataset = datareader.build_dataset(instances[train_num:], mode='seq')

    # =====after szb cls
    instances = utils.NerExample.load_from_jsonl('data/yanbao/sent_szb/21_06_10_11_37_cls/test_1_0_exm_lst.jsonl', external_attrs=['cls_pred', 'cls_tgt', 'source_id'])
    instances = [exm for exm in instances if exm.cls_pred[0] == 1]
    train_num = int(0.9 * len(instances))
    train_data = instances[:train_num]
    test_data = instances[train_num:]
    utils.NerExample.save_to_jsonl(train_data, 'data/yanbao/sent_szb/sent_exm_seq_after_cls_train.jsonl', for_human_read=True, external_attrs=['cls_pred', 'cls_tgt', 'source_id'], overwrite=False)
    utils.NerExample.save_to_jsonl(test_data, 'data/yanbao/sent_szb/sent_exm_seq_after_cls_test.jsonl', for_human_read=True, external_attrs=['cls_pred', 'cls_tgt', 'source_id'], overwrite=False)
    args.train_dataset = datareader.build_dataset(train_data, mode='span')
    args.test_dataset = datareader.build_dataset(test_data, mode='span')

    # ====multimrc====
    # instances = utils.NerExample.load_from_jsonl('data/yanbao/sent/doc_exm_w_re_sent.jsonl', external_attrs=['itp2atp_dct', 'source_id'])
    # random.shuffle(instances)
    # total_num = len(instances)
    # train_num = int(0.9 * total_num)
    # utils.NerExample.save_to_jsonl(instances[:train_num], 'data/yanbao/sent/doc_exm_w_re_sent_train.jsonl', for_human_read=True, external_attrs=['itp2atp_dct', 'source_id'], overwrite=False)
    # utils.NerExample.save_to_jsonl(instances[train_num:], 'data/yanbao/sent/doc_exm_w_re_sent_test.jsonl', for_human_read=True, external_attrs=['itp2atp_dct', 'source_id'], overwrite=False)
    # args.train_dataset = datareader.build_dataset(instances[:train_num], mode='multimrc')
    # args.test_dataset = datareader.build_dataset(instances[train_num:], mode='multimrc')
    # ====multimrc====

    args.dev_dataset = None
    args.batch_size = 8
    # args.eval_every_step = 400

    # utils.NerExample.stats(instances[train_num:])
    # utils.NerExample.stats(instances, ent_anal_out_file='data/yanbao/sent/stats_ent.txt')
    # exit(0)
    #
    # # =====CLUNER
    # datareader = NerDataReader(conf['bert_model_dir'], 256,
    #                            ent_file='data/cluener_public/ent_lst_addO.txt',
    #                            # ent_file='data/cluener_public/ent_lst.txt,  # sigmoid
    #                            )
    # args.train_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/cluener_public/train_stand.jsonl'), mode='span')
    # args.test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/cluener_public/dev_stand.jsonl'), mode='span')
    # # args.test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/cluener_public/test_stand.jsonl'), mode='span')
    # args.dev_dataset = None
    # args.batch_size = 16

    # # =====people daily
    # datareader = NerDataReader(conf['bert_model_dir'], 256, ent_file='data/people_daily_public/ent_lst.txt')
    # args.train_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/people_daily_public/train_stand.jsonl'), mode='span')
    # args.dev_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/people_daily_public/dev_stand.jsonl'), mode='span')
    # args.test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/people_daily_public/test_stand.jsonl'), mode='span')
    # args.batch_size = 16

    # # =====ontonote5
    # datareader = NerDataReader(conf['bert_model_dir'], 512, ent_file='data/ontonote_public/Eng/ent_lst_addO.txt')  # softmax
    # args.train_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/ontonote_public/Eng/train_tokenized_c.jsonl'), mode='span')
    # args.test_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/ontonote_public/Eng/test_tokenized_c.jsonl'), mode='span')
    # args.dev_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/ontonote_public/Eng/dev_tokenized_c.jsonl'), mode='span')
    # args.batch_size = 16
    # args.eval_every_step = 500
    # args.eval_after_step = 4000

    args.char2id = datareader.char2id
    args.ent2id = datareader.ent2id
    args.tag2id = datareader.tag2id
    args.id2char = args.char2id.get_reverse()
    args.id2tag = args.tag2id.get_reverse()
    args.id2ent = args.ent2id.get_reverse()
    args.lr = 2e-5  # 原来1e-5
    # args.lr = 1e-3  # 原来1e-5
    args.num_epochs = 100
    # args.negative_sample = True
    args.negative_sample = False

    # ====span====
    trainer = Trainer(args, conf, model_type='span')
    # ====span====

    # ====seq====
    # trainer = Trainer(args, conf, model_type='seq')
    # ====seq====

    # ====multimrc====
    # trainer = Trainer(args, conf, model_type='multimrc')
    # trainer = Trainer(args, conf, model_type='multimrc', exist_ckpt='/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_08_22_07_multimrc_test/best_test_model_20_1980.ckpt', evaluate=False)
    # ====multimrc====

    # ====cls====
    # trainer = Trainer(args, conf, model_type='cls')
    # args.infer_dataset = datareader.build_dataset(utils.NerExample.load_from_jsonl('data/yanbao/sent_szb/sent_exm_final.jsonl', external_attrs=['source_id']), mode='cls')
    # trainer = Trainer(args, conf, model_type='cls', exist_ckpt='/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_07_21_47_cls_szb_cls_bert/best_test_model_8_3200.ckpt', evaluate=True)
    # ====cls====

    # cls yanbao best
    # args.test_dataset = datareader.build_dataset(instances, mode='cls')
    # trainer = Trainer(args, conf, model_type='cls', evaluate=True,
    #                   exist_ckpt='/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_01_20_39_cls_yanbao_cls_allsample_possweight10/best_dev_model_4_3260.ckpt')

    # CLUNER best 80.53
    # trainer = Trainer(args, conf, model_type='span', evaluate=True,
    #                   exist_ckpt='/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_05_21_11_45_span_self_attn_mask_mean_cluener_softmax_nonesample_min_minus_mask_worope/best_test_model_6.ckpt')
    # trainer = Trainer(args, conf, model_type='span', exist_ckpt='/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_05_19_16_57_span_self_attn_mask_mean_test_min/best_test_model_2.ckpt')

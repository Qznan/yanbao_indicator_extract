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
import time, sys, copy, json
from pathlib import Path
from collections import defaultdict
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


class Predictor:
    def __init__(self, args, conf, ckpt_path, datareader, mode='span'):
        self.char2id = datareader.char2id
        self.ent2id = datareader.ent2id
        self.tag2id = datareader.tag2id
        self.id2char = args.char2id.get_reverse()
        self.id2tag = args.tag2id.get_reverse()
        self.id2ent = args.ent2id.get_reverse()
        self.datareader = datareader
        self.args = args
        self.conf = conf
        self.mode = mode

        self.model = {
            'span': Bert_Span,
            'seq': Bert_Seq,
            'cls': Bert_Cls,
            'multimrc': Bert_MultiMrc,
        }.get(mode)(
            conf,
            {'char2id': args.char2id,
             'tag2id': args.tag2id,
             'ent2id': args.ent2id,
             'id2char': args.id2char,
             'id2tag': args.id2tag,
             'id2ent': args.id2ent,
             }
        )

        self.model.cuda()
        self.model.eval()
        self.model.load_state_dict(torch.load(ckpt_path))
        print(f'load model param success! {ckpt_path}')

    def predict(self, *args, **kwargs):
        spec_predict = {
            'span': self.predict_span,
            'seq': self.predict_seq,
            'cls': self.predict_cls,
            'multimrc': self.predict_multimrc,
        }.get(self.mode, None)
        return spec_predict(*args, **kwargs)

    def predict_cls(self, batch_sents, max_len=512, bsz=6, verbose=False):
        if not hasattr(self, 'cls_batcher'): self.cls_batcher = self.datareader.get_batcher_fn(gpu=True, mode='cls', train=False)

        exm_lst = [utils.NerExample(char_lst=list(sent), ent_dct={}) for sent in batch_sents]
        for exm in exm_lst:
            exm.truncate(max_size=max_len - 2, direction='tail')
        post_process_exm_lst = [self.datareader.cls_post_process(exm, train=False) for exm in exm_lst]

        for bdx in range(0, len(exm_lst), bsz):
            batch_e = post_process_exm_lst[bdx: bdx + bsz]  # 模仿dataloader batch功能
            inputs = self.cls_batcher(batch_e)

            batch_bert_chars = inputs['input_ids']
            batch_seq_len = inputs['seq_len']
            batch_bert_token_type_ids = inputs['bert_token_type_ids']
            batch_bert_attention_mask = inputs['bert_attention_mask']
            batch_ner_exm = inputs['batch_ner_exm']

            cls_pred_prob = self.model.predict(
                batch_bert_chars,
                batch_bert_token_type_ids,
                batch_bert_attention_mask,
                batch_seq_len)

            cls_pred_prob = cls_pred_prob.cpu().detach().numpy()
            cls_pred = (cls_pred_prob >= 0.5).astype('int')

            for exm, _cls_pred, _cls_pred_prob in zip(batch_ner_exm, cls_pred.tolist(), cls_pred_prob.tolist()):
                exm.cls_pred = [_cls_pred, _cls_pred_prob]

        if verbose:
            for exm in exm_lst:
                print(exm.to_json_str(for_human_read=True, external_attrs=['cls_pred', 'itp2atp_dct', 'source_id']))
            # utils.NerExample.save_to_jsonl(exm_lst, ckpt_path.parent / 'pred_exm.jsonl')
        return exm_lst
    
    def predict_multimrc(self, batch_sents, batch_itp, max_len=512, bsz=6, verbose=False):
        if not hasattr(self, 'multimrc_batcher'): self.multimrc_batcher = self.datareader.get_batcher_fn(gpu=True, mode='multimrc', train=False)
        exm_lst = []
        for sent, (itp_name, start, end) in zip(batch_sents, batch_itp):
            ent_dct = {f'指标-{itp_name}': [[start, end, -1]]}
            exm = utils.NerExample(char_lst=list(sent), ent_dct=ent_dct)
            exm.itp2atp_dct = {'-1': []}
            exm.truncate(max_size=max_len - 2, direction='tail')
            exm_lst.append(exm)
        post_process_exm_lst = [self.datareader.multimrc_post_process(exm, train=False) for exm in exm_lst]

        for bdx in range(0, len(exm_lst), bsz):
            batch_e = post_process_exm_lst[bdx: bdx + bsz]  # 模仿dataloader batch功能
            inputs = self.multimrc_batcher(batch_e)

            batch_bert_chars = inputs['input_ids']
            batch_token_type_id = inputs['token_type_id']
            batch_bert_attention_mask = inputs['attn_mask']
            batch_query_mask = inputs['query_mask']
            batch_answers = inputs['answers']
            batch_ner_exm = inputs['batch_ner_exm']

            mrc_loss, batch_prob, batch_context_start_index = self.model.predict(
                batch_bert_chars,
                batch_token_type_id,
                batch_bert_attention_mask,
                batch_query_mask, batch_answers)

            all_atps = self.datareader.all_atps
            num_all_atps = len(all_atps)

            batch_prob = batch_prob.cpu().detach().numpy()
            batch_context_start_index = batch_context_start_index.cpu().detach().numpy().tolist()
            for i, (exm, context_start_index) in enumerate(zip(batch_ner_exm, batch_context_start_index)):
                atp_res = []
                prob = batch_prob[i * num_all_atps: (i + 1) * num_all_atps]
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

        if verbose:
            for exm in exm_lst:
                # print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct', 'source_id']))
                print(multimrc_exm_to_json(exm))
            # utils.NerExample.save_to_jsonl(exm_lst, ckpt_path.parent / 'pred_exm.jsonl', for_human_read=True, external_attrs=['itp2atp_dct', 'source_id'])
        return exm_lst

    def predict_span(self, batch_sents, max_len=512, bsz=6, verbose=False):
        if not hasattr(self, 'span_batcher'): self.span_batcher = self.datareader.get_batcher_fn(gpu=True, mode='span', train=False)

        exm_lst = [utils.NerExample(char_lst=list(sent), ent_dct={}) for sent in batch_sents]
        for exm in exm_lst:
            exm.truncate(max_size=max_len - 2, direction='tail')
        post_process_exm_lst = [self.datareader.span_post_process_softmax(exm, train=False) for exm in exm_lst]

        for bdx in range(0, len(exm_lst), bsz):
            batch_e = post_process_exm_lst[bdx: bdx + bsz]  # 模仿dataloader batch功能
            inputs = self.span_batcher(batch_e)

            batch_bert_chars = inputs['input_ids']
            batch_seq_len = inputs['seq_len']
            batch_bert_token_type_ids = inputs['bert_token_type_ids']
            batch_bert_attention_mask = inputs['bert_attention_mask']
            batch_ner_exm = inputs['batch_ner_exm']

            span_ner_mat_tensor, batch_span_ner_pred_lst, conj_dot_product_score = self.model.predict(
                batch_bert_chars,
                batch_bert_token_type_ids,
                batch_bert_attention_mask,
                batch_seq_len)

            batch_span_lst_len = [(l + 1) * l // 2 for l in batch_seq_len.tolist()]
            batch_span_ner_pred_lst = utils.split_list(batch_span_ner_pred_lst.cpu().detach().numpy(), batch_span_lst_len)
            for exm, length, span_ner_pred_lst in zip(batch_ner_exm, batch_seq_len, batch_span_ner_pred_lst):
                exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst(span_ner_pred_lst, length, self.args.id2ent)  # softmax
                # exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst_sigmoid(span_ner_pred_lst, length, self.args.id2ent)  # sigmoid

                exm.pred_ent_dct = exm.get_flat_pred_ent_dct()  # 平展 TODO

        if verbose:
            for exm in exm_lst:
                print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct', 'source_id']))
            # utils.NerExample.save_to_jsonl(exm_lst, ckpt_path.parent / 'pred_exm.jsonl')
        return exm_lst

    def predict_seq(self, batch_sents, max_len=512, bsz=6, verbose=False):
        if not hasattr(self, 'seq_batcher'): self.seq_batcher = self.datareader.get_batcher_fn(gpu=True, mode='seq', train=False)

        exm_lst = [utils.NerExample(char_lst=list(sent), ent_dct={}) for sent in batch_sents]
        for exm in exm_lst:
            exm.truncate(max_size=max_len - 2, direction='tail')
        post_process_exm_lst = [self.datareader.seq_post_process(exm, train=False) for exm in exm_lst]

        for bdx in range(0, len(exm_lst), bsz):
            batch_e = post_process_exm_lst[bdx: bdx + bsz]  # 模仿dataloader batch功能
            inputs = self.seq_batcher(batch_e)

            batch_bert_chars = inputs['input_ids']
            batch_seq_len = inputs['seq_len']
            batch_bert_token_type_ids = inputs['bert_token_type_ids']
            batch_bert_attention_mask = inputs['bert_attention_mask']
            batch_ner_exm = inputs['batch_ner_exm']

            emission, decode_ids = self.model.predict(batch_bert_chars,
                                                      batch_bert_token_type_ids,
                                                      batch_bert_attention_mask,
                                                      batch_seq_len)

            for exm, decode_ids_ in zip(batch_ner_exm, decode_ids):
                tag_lst = [args.id2tag[tag_id] for tag_id in decode_ids_]
                # assert len(tag_lst) == len(exm.char_lst)
                pred_ent_dct, _ = utils.NerExample.extract_entity_by_tags(tag_lst)
                for k, v_lst in pred_ent_dct.items():
                    for e in v_lst:
                        e.append(1.)  # 假设概率为1
                exm.pred_ent_dct = pred_ent_dct

        if verbose:
            for exm in exm_lst:
                print(exm)
            # utils.NerExample.save_to_jsonl(exm_lst, ckpt_path.parent / 'pred_exm.jsonl')
        return exm_lst

    def save_cls(self, domain,batch_sents, max_len=512, bsz=6, verbose=False):
        if not hasattr(self, 'cls_batcher'): self.cls_batcher = self.datareader.get_batcher_fn(gpu=True, mode='cls', train=False)

        exm_lst = [utils.NerExample(char_lst=list(sent), ent_dct={}) for sent in batch_sents]
        for exm in exm_lst:
            exm.truncate(max_size=max_len - 2, direction='tail')
        post_process_exm_lst = [self.datareader.cls_post_process(exm, train=False) for exm in exm_lst]

        for bdx in range(0, len(exm_lst), bsz):
            batch_e = post_process_exm_lst[bdx: bdx + bsz]  # 模仿dataloader batch功能
            inputs = self.cls_batcher(batch_e)

            batch_bert_chars = inputs['input_ids']
            batch_seq_len = inputs['seq_len']
            batch_bert_token_type_ids = inputs['bert_token_type_ids']
            batch_bert_attention_mask = inputs['bert_attention_mask']
            batch_ner_exm = inputs['batch_ner_exm']
            tag = torch.tensor(1)
            traced_script_module  =  torch.jit.trace(self.model,(batch_bert_chars,
                batch_bert_token_type_ids,
                batch_bert_attention_mask,
                batch_seq_len,tag))
            traced_script_module.save(f"./model1/{domain}_predictor_cls.pth")
            return
    def save_multimrc(self, domain,batch_sents, batch_itp, max_len=512, bsz=6, verbose=False):
        if not hasattr(self, 'multimrc_batcher'): self.multimrc_batcher = self.datareader.get_batcher_fn(gpu=True, mode='multimrc', train=False)
        exm_lst = []
        for sent, (itp_name, start, end) in zip(batch_sents, batch_itp):
            ent_dct = {f'指标-{itp_name}': [[start, end, -1]]}
            exm = utils.NerExample(char_lst=list(sent), ent_dct=ent_dct)
            exm.itp2atp_dct = {'-1': []}
            exm.truncate(max_size=max_len - 2, direction='tail')
            exm_lst.append(exm)
        post_process_exm_lst = [self.datareader.multimrc_post_process(exm, train=False) for exm in exm_lst]

        for bdx in range(0, len(exm_lst), bsz):
            batch_e = post_process_exm_lst[bdx: bdx + bsz]  # 模仿dataloader batch功能
            inputs = self.multimrc_batcher(batch_e)

            batch_bert_chars = inputs['input_ids']
            batch_token_type_id = inputs['token_type_id']
            batch_bert_attention_mask = inputs['attn_mask']
            batch_query_mask = inputs['query_mask']
            batch_answers = inputs['answers']
            # batch_ner_exm = inputs['batch_ner_exm']
            # print(type(batch_bert_chars))
            # print(type(batch_token_type_id))
            # print(type(batch_bert_attention_mask))
            # print(type(batch_query_mask))
            # print(type(batch_answers))
            
            tag = torch.tensor(1)
            # print(type(tag))
            # traced_script_module = torch.jit.trace(self.model,(
            #     batch_bert_chars,
            #     batch_token_type_id,
            #     batch_bert_attention_mask,
            #     batch_query_mask, tag, batch_answers))
            traced_script_module = torch.jit.trace(self.model,(
                batch_bert_chars,
                batch_token_type_id,
                batch_bert_attention_mask,
                batch_query_mask, tag))
            traced_script_module.save(f"./model1/{domain}_predictor_multimrc.pth")
            return
    def save_span(self, domain,batch_sents, max_len=512, bsz=6, verbose=False):
        if not hasattr(self, 'span_batcher'): self.span_batcher = self.datareader.get_batcher_fn(gpu=True, mode='span', train=False)

        exm_lst = [utils.NerExample(char_lst=list(sent), ent_dct={}) for sent in batch_sents]
        for exm in exm_lst:
            exm.truncate(max_size=max_len - 2, direction='tail')
        post_process_exm_lst = [self.datareader.span_post_process_softmax(exm, train=False) for exm in exm_lst]

        for bdx in range(0, len(exm_lst), bsz):
            batch_e = post_process_exm_lst[bdx: bdx + bsz]  # 模仿dataloader batch功能
            inputs = self.span_batcher(batch_e)

            batch_bert_chars = inputs['input_ids']
            batch_seq_len = inputs['seq_len']
            batch_bert_token_type_ids = inputs['bert_token_type_ids']
            batch_bert_attention_mask = inputs['bert_attention_mask']
            batch_ner_exm = inputs['batch_ner_exm']
            tag = torch.tensor(1)
            traced_script_module = torch.jit.trace(self.model,(
                batch_bert_chars,
                batch_bert_token_type_ids,
                batch_bert_attention_mask,
                batch_seq_len,tag))
            traced_script_module.save(f"./model1/{domain}_predictor_span.pth")
            return

class Predictor_1:
    def __init__(self, args, conf, datareader, model):
        self.char2id = datareader.char2id
        self.ent2id = datareader.ent2id
        self.tag2id = datareader.tag2id
        self.id2char = args.char2id.get_reverse()
        self.id2tag = args.tag2id.get_reverse()
        self.id2ent = args.ent2id.get_reverse()
        self.datareader = datareader
        self.args = args
        self.conf = conf


        self.model = model

        self.model.cuda()
        self.model.eval()
        # self.model.load_state_dict(torch.load(ckpt_path))
        # print(f'load model param success! {ckpt_path}')

    def predict(self, *args, **kwargs):
        spec_predict = {
            'span': self.predict_span,
            'seq': self.predict_seq,
            'cls': self.predict_cls,
            'multimrc': self.predict_multimrc,
        }.get(self.mode, None)
        return spec_predict(*args, **kwargs)

    def predict_cls(self, batch_sents, max_len=512, bsz=6, verbose=False):
        if not hasattr(self, 'cls_batcher'): self.cls_batcher = self.datareader.get_batcher_fn(gpu=True, mode='cls', train=False)

        exm_lst = [utils.NerExample(char_lst=list(sent), ent_dct={}) for sent in batch_sents]
        for exm in exm_lst:
            exm.truncate(max_size=max_len - 2, direction='tail')
        post_process_exm_lst = [self.datareader.cls_post_process(exm, train=False) for exm in exm_lst]

        for bdx in range(0, len(exm_lst), bsz):
            batch_e = post_process_exm_lst[bdx: bdx + bsz]  # 模仿dataloader batch功能
            inputs = self.cls_batcher(batch_e)

            batch_bert_chars = inputs['input_ids']
            batch_seq_len = inputs['seq_len']
            batch_bert_token_type_ids = inputs['bert_token_type_ids']
            batch_bert_attention_mask = inputs['bert_attention_mask']
            batch_ner_exm = inputs['batch_ner_exm']
            tag = torch.tensor(1)
            cls_pred_prob = self.model(
                batch_bert_chars,
                batch_bert_token_type_ids,
                batch_bert_attention_mask,
                batch_seq_len,tag)

            cls_pred_prob = cls_pred_prob.cpu().detach().numpy()
            cls_pred = (cls_pred_prob >= 0.5).astype('int')

            for exm, _cls_pred, _cls_pred_prob in zip(batch_ner_exm, cls_pred.tolist(), cls_pred_prob.tolist()):
                exm.cls_pred = [_cls_pred, _cls_pred_prob]

        if verbose:
            for exm in exm_lst:
                print(exm.to_json_str(for_human_read=True, external_attrs=['cls_pred', 'itp2atp_dct', 'source_id']))
            # utils.NerExample.save_to_jsonl(exm_lst, ckpt_path.parent / 'pred_exm.jsonl')
        return exm_lst
    
    def predict_multimrc(self, batch_sents, batch_itp, max_len=512, bsz=6, verbose=False):
        if not hasattr(self, 'multimrc_batcher'): self.multimrc_batcher = self.datareader.get_batcher_fn(gpu=True, mode='multimrc', train=False)
        exm_lst = []
        for sent, (itp_name, start, end) in zip(batch_sents, batch_itp):
            ent_dct = {f'指标-{itp_name}': [[start, end, -1]]}
            exm = utils.NerExample(char_lst=list(sent), ent_dct=ent_dct)
            exm.itp2atp_dct = {'-1': []}
            exm.truncate(max_size=max_len - 2, direction='tail')
            exm_lst.append(exm)
        post_process_exm_lst = [self.datareader.multimrc_post_process(exm, train=False) for exm in exm_lst]

        for bdx in range(0, len(exm_lst), bsz):
            batch_e = post_process_exm_lst[bdx: bdx + bsz]  # 模仿dataloader batch功能
            inputs = self.multimrc_batcher(batch_e)

            batch_bert_chars = inputs['input_ids']
            batch_token_type_id = inputs['token_type_id']
            batch_bert_attention_mask = inputs['attn_mask']
            batch_query_mask = inputs['query_mask']
            batch_answers = inputs['answers']
            batch_ner_exm = inputs['batch_ner_exm']
            tag = torch.tensor(1)

            mrc_loss, batch_prob, batch_context_start_index = self.model(
                batch_bert_chars,
                batch_token_type_id,
                batch_bert_attention_mask,
                batch_query_mask, tag)

            all_atps = self.datareader.all_atps
            num_all_atps = len(all_atps)

            batch_prob = batch_prob.cpu().detach().numpy()
            batch_context_start_index = batch_context_start_index.cpu().detach().numpy().tolist()
            for i, (exm, context_start_index) in enumerate(zip(batch_ner_exm, batch_context_start_index)):
                atp_res = []
                prob = batch_prob[i * num_all_atps: (i + 1) * num_all_atps]
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

        if verbose:
            for exm in exm_lst:
                # print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct', 'source_id']))
                print(multimrc_exm_to_json(exm))
            # utils.NerExample.save_to_jsonl(exm_lst, ckpt_path.parent / 'pred_exm.jsonl', for_human_read=True, external_attrs=['itp2atp_dct', 'source_id'])
        return exm_lst

    def predict_span(self, batch_sents, max_len=512, bsz=6, verbose=False):
        if not hasattr(self, 'span_batcher'): self.span_batcher = self.datareader.get_batcher_fn(gpu=True, mode='span', train=False)

        exm_lst = [utils.NerExample(char_lst=list(sent), ent_dct={}) for sent in batch_sents]
        for exm in exm_lst:
            exm.truncate(max_size=max_len - 2, direction='tail')
        post_process_exm_lst = [self.datareader.span_post_process_softmax(exm, train=False) for exm in exm_lst]

        for bdx in range(0, len(exm_lst), bsz):
            batch_e = post_process_exm_lst[bdx: bdx + bsz]  # 模仿dataloader batch功能
            inputs = self.span_batcher(batch_e)

            batch_bert_chars = inputs['input_ids']
            batch_seq_len = inputs['seq_len']
            batch_bert_token_type_ids = inputs['bert_token_type_ids']
            batch_bert_attention_mask = inputs['bert_attention_mask']
            batch_ner_exm = inputs['batch_ner_exm']

            tag = torch.tensor(1)
            span_ner_mat_tensor, batch_span_ner_pred_lst, conj_dot_product_score = self.model(
                batch_bert_chars,
                batch_bert_token_type_ids,
                batch_bert_attention_mask,
                batch_seq_len,tag)

            batch_span_lst_len = [(l + 1) * l // 2 for l in batch_seq_len.tolist()]
            batch_span_ner_pred_lst = utils.split_list(batch_span_ner_pred_lst.cpu().detach().numpy(), batch_span_lst_len)
            for exm, length, span_ner_pred_lst in zip(batch_ner_exm, batch_seq_len, batch_span_ner_pred_lst):
                exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst(span_ner_pred_lst, length, self.args.id2ent)  # softmax
                # exm.pred_ent_dct = utils.NerExample.from_span_level_ner_tgt_lst_sigmoid(span_ner_pred_lst, length, self.args.id2ent)  # sigmoid

                exm.pred_ent_dct = exm.get_flat_pred_ent_dct()  # 平展 TODO

        if verbose:
            for exm in exm_lst:
                print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct', 'source_id']))
            # utils.NerExample.save_to_jsonl(exm_lst, ckpt_path.parent / 'pred_exm.jsonl')
        return exm_lst

    

class PipeLine_Infer:
    def __init__(self, args, max_len=512):
        print(f'PipeLine Infer for domain: {args.domain} init...')
        self.domain = args.domain
        self.max_len = max_len
        self.conf = {
            'dropout_rate': 0.2,
            'bert_model_dir': '/home/zyn/huggingface_model_resource/chinese-bert-wwm-ext',
            'span_layer_type': 'self_attn_mask_mean',
            'span_model_negsample_rate': 0.3
        }
        args.ckpt_dir = Path('span_ner_modules/ckpt')
        args.ckpt_dir = Path(f'data/yanbao/{self.domain}/ckpt')
        args.batch_size = 16
        datareader = NerDataReader(self.conf['bert_model_dir'],
                                   self.max_len,
                                   # ent_file='data/yanbao/sent/full_ent_lst_remove_atp.txt',
                                   ent_file=f'data/yanbao/{self.domain}/full_ent_lst_remove_atp.txt',
                                   atp_file=f'data/yanbao/{self.domain}/atp_lst.txt',
                                   )
        args.char2id = datareader.char2id
        args.ent2id = datareader.ent2id
        args.tag2id = datareader.tag2id
        args.id2char = args.char2id.get_reverse()
        args.id2tag = args.tag2id.get_reverse()
        args.id2ent = args.ent2id.get_reverse()
        self.datareader = datareader
        self.args = args
        self.init_model()

    def init_model(self):
        # ====cls====

        # self.cls_ckpt_path = Path('/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_07_21_47_cls_szb_cls_bert/best_test_model_8_3200.ckpt')
        # self.cls_ckpt_path = Path(f'/home/zyn/MultiQueryMRC/span_ner_modules/{self.domain}/ckpt/21_06_17_22_44_cls_szb_ssent_bert/best_test_model_19_23940.ckpt')
        # print("----------------------")
        # print(self.args.ckpt_dir.glob('*cls*'))
        # for i in self.args.ckpt_dir.glob('*cls*'):
        #     print(i)
        # print(sorted(list(self.args.ckpt_dir.glob('*cls*')), key=lambda p: p.stat().st_ctime, reverse=False)[0].glob('*.ckpt'))
        # print("----------------------")
        self.cls_ckpt_path = list(sorted(list(self.args.ckpt_dir.glob('*cls*')), key=lambda p: p.stat().st_ctime, reverse=False)[0].glob('*.ckpt'))[0]
        self.predictor_cls = Predictor(self.args, self.conf, ckpt_path=self.cls_ckpt_path, datareader=self.datareader, mode='cls')
        # torch.save(self.predictor_cls,f"./model_save/{self.domain}_predictor_cls.pth")
        # ====span=====
        # self.span_ckpt_path = Path('/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_10_19_37_span_self_attn_mask_mean_after_mul_szb_cls_bert/best_test_model_34_7446.ckpt')
        # self.span_ckpt_path = Path('/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_16_22_43_span_self_attn_mask_mean_after_mul_szb_cls_bert_orinegsample/best_test_model_63_13797.ckpt')
        # self.span_ckpt_path = Path(f'/home/zyn/MultiQueryMRC/span_ner_modules/{self.domain}/ckpt/21_06_18_12_33_span_self_attn_mask_mean_mzb_ssent_bert/best_test_model_98_20972.ckpt')
        self.span_ckpt_path = list(sorted(list(self.args.ckpt_dir.glob('*span*')), key=lambda p: p.stat().st_ctime, reverse=False)[0].glob('*.ckpt'))[0]
        self.predictor_span = Predictor(self.args, self.conf, ckpt_path=self.span_ckpt_path, datareader=self.datareader, mode='span')
        # torch.save(self.predictor_span,f"./model_save/{self.domain}_predictor_span.pth")
        # ====multimrc====
        # self.multimrc_ckpt_path = Path('/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_09_15_09_multimrc_init/best_test_model_50_9850.ckpt')
        # self.multimrc_ckpt_path = Path(f'/home/zyn/MultiQueryMRC/span_ner_modules/{self.domain}/ckpt/21_06_17_23_12_multimrc_bert/best_test_model_5_2085.ckpt')
        self.multimrc_ckpt_path = list(sorted(list(self.args.ckpt_dir.glob('*multimrc*')), key=lambda p: p.stat().st_ctime, reverse=False)[0].glob('*.ckpt'))[0]
        self.predictor_multimrc = Predictor(self.args, self.conf, ckpt_path=self.multimrc_ckpt_path, datareader=self.datareader, mode='multimrc')
        # torch.save(self.predictor_multimrc,f"./model_save/{self.domain}_predictor_multimrc.pth")

    def predict(self, doc_str):
        print(doc_str)
        sent_lst = doc_str.split('。')
        sent_lst = [sent + '。' for sent in sent_lst if sent]
        print(f'doc总句子数:{len(sent_lst)}')

        exm_lst = self.predictor_cls.predict(batch_sents=sent_lst)
        exm_lst = [exm for exm in exm_lst if exm.cls_pred[0] == 1]  # 有指标的样本
        print(f'有指标的句子数:{len(exm_lst)}')

        batch_sents = [exm.text for exm in exm_lst]
        exm_lst = self.predictor_span.predict(batch_sents=batch_sents)  # 抽取指标
        for exm in exm_lst:
            print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct', 'source_id']))
            # input()

        mrc_input_lst = []
        for exm in exm_lst:
            sent = exm.text
            for itp_name, pos_lst in exm.pred_ent_dct.items():
                for itp_start, itp_end, prob in pos_lst:
                    mrc_input_lst.append([sent, itp_name.replace('指标-', ''), itp_start, itp_end])

        mrc_batch_sent, mrc_batch_itp = [], []
        for sent, itp_name, itp_start, itp_end in mrc_input_lst:
            mrc_batch_sent.append(sent)
            mrc_batch_itp.append([itp_name, itp_start, itp_end])
        exm_lst = self.predictor_multimrc.predict(batch_sents=mrc_batch_sent, batch_itp=mrc_batch_itp)

        # for exm in exm_lst:
        #     print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct', 'source_id']))
        json_str = PipeLine_Infer.print_res(exm_lst)
        print(json_str)
        return json_str
    def save(self, doc_str):
        print(doc_str)
        sent_lst = doc_str.split('。')
        sent_lst = [sent + '。' for sent in sent_lst if sent]
        print(f'doc总句子数:{len(sent_lst)}')
        self.predictor_cls.save_cls(self.domain,batch_sents=sent_lst)
        exm_lst = self.predictor_cls.predict(batch_sents=sent_lst)
        exm_lst = [exm for exm in exm_lst if exm.cls_pred[0] == 1]  # 有指标的样本
        print(f'有指标的句子数:{len(exm_lst)}')

        batch_sents = [exm.text for exm in exm_lst]
        self.predictor_span.save_span(self.domain,batch_sents=batch_sents)
        exm_lst = self.predictor_span.predict(batch_sents=batch_sents)  # 抽取指标
        for exm in exm_lst:
            print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct', 'source_id']))
            # input()

        mrc_input_lst = []
        for exm in exm_lst:
            sent = exm.text
            for itp_name, pos_lst in exm.pred_ent_dct.items():
                for itp_start, itp_end, prob in pos_lst:
                    mrc_input_lst.append([sent, itp_name.replace('指标-', ''), itp_start, itp_end])

        mrc_batch_sent, mrc_batch_itp = [], []
        for sent, itp_name, itp_start, itp_end in mrc_input_lst:
            mrc_batch_sent.append(sent)
            mrc_batch_itp.append([itp_name, itp_start, itp_end])
        self.predictor_multimrc.save_multimrc(self.domain,batch_sents=mrc_batch_sent, batch_itp=mrc_batch_itp)
        exm_lst = self.predictor_multimrc.predict(batch_sents=mrc_batch_sent, batch_itp=mrc_batch_itp)

        # for exm in exm_lst:
        #     print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct', 'source_id']))
        json_str = PipeLine_Infer.print_res(exm_lst)
        print(json_str)
        return json_str

    @staticmethod
    def print_res(exm_lst):
        # multimrc exm_lst
        text_dct = defaultdict(dict)
        for exm in exm_lst:
            text = exm.text
            itp_name = list(exm.ent_dct.keys())[0].replace('指标-', '')
            i_s, i_e, i_prob = list(exm.ent_dct.values())[0][0]
            itp = f'{itp_name} = {text[i_s:i_e]} = {i_s}|{i_e}'
            text_dct[text][itp] = {}
            atp_lst = []
            for atp_name, v in exm.pred_ent_dct.items():
                for a_s, a_e, a_prob in v:
                    atp_lst.append(f'{atp_name.replace("属性-", "")} = {text[a_s: a_e]} = {a_s}|{a_e}')
            atp_lst.sort()
            text_dct[text][itp] = atp_lst
        text_dct = dict(text_dct)
        json_str = json.dumps(text_dct, ensure_ascii=False, indent=2)
        return json_str

    def test(self):
        # exm_lst = utils.NerExample.load_from_jsonl('data/yanbao/sent/doc_exm.jsonl')
        # # print('doc_str:', doc_str)
        # print('=======预测doc')
        # self.predict(exm_lst[26].text)
        # print('=======预测doc')
        # self.predict(exm_lst[49].text)

        exm_lst = utils.NerExample.load_from_jsonl(f'data/yanbao/{self.domain}/test/doc_exm.jsonl')
        for exm in exm_lst:
            print('=======预测doc')
            self.predict(exm.text)


class PipeLine_Infer_1:
    def __init__(self, args, max_len=512):
        print(f'PipeLine Infer for domain: {args.domain} init...')
        self.domain = args.domain
        self.max_len = max_len
        self.conf = {
            'dropout_rate': 0.2,
            'bert_model_dir': '/home/zyn/huggingface_model_resource/chinese-bert-wwm-ext',
            'span_layer_type': 'self_attn_mask_mean',
            'span_model_negsample_rate': 0.3
        }
        args.ckpt_dir = Path('span_ner_modules/ckpt')
        args.ckpt_dir = Path(f'data/yanbao/{self.domain}/ckpt')
        args.batch_size = 16
        datareader = NerDataReader(self.conf['bert_model_dir'],
                                   self.max_len,
                                   # ent_file='data/yanbao/sent/full_ent_lst_remove_atp.txt',
                                   ent_file=f'data/yanbao/{self.domain}/full_ent_lst_remove_atp.txt',
                                   atp_file=f'data/yanbao/{self.domain}/atp_lst.txt',
                                   )
        args.char2id = datareader.char2id
        args.ent2id = datareader.ent2id
        args.tag2id = datareader.tag2id
        args.id2char = args.char2id.get_reverse()
        args.id2tag = args.tag2id.get_reverse()
        args.id2ent = args.ent2id.get_reverse()
        self.datareader = datareader
        self.args = args
        self.init_model()

    def init_model(self):
        # ====cls====

        self.predictor_cls_model = torch.jit.load(f"./model1/{self.domain}_predictor_cls.pth")
        
        # ====span=====
        self.predictor_span_model = torch.jit.load(f"./model1/{self.domain}_predictor_span.pth")

        # ====multimrc====
        self.predictor_multimrc_model = torch.jit.load(f"./model1/{self.domain}_predictor_multimrc.pth")
        
        self.predictor_cls = Predictor_1(self.args, self.conf, datareader=self.datareader, model=self.predictor_cls_model)

        # ====span=====
       
        self.predictor_span = Predictor_1(self.args, self.conf, datareader=self.datareader, model=self.predictor_span_model)
        
        # ====multimrc====
    
        self.predictor_multimrc = Predictor_1(self.args, self.conf, datareader=self.datareader, model=self.predictor_multimrc_model)

    def predict(self, doc_str):
        print(doc_str)
        sent_lst = doc_str.split('。')
        sent_lst = [sent + '。' for sent in sent_lst if sent]
        print(f'doc总句子数:{len(sent_lst)}')

        exm_lst = self.predictor_cls.predict_cls(batch_sents=sent_lst)
        exm_lst = [exm for exm in exm_lst if exm.cls_pred[0] == 1]  # 有指标的样本
        print(f'有指标的句子数:{len(exm_lst)}')

        batch_sents = [exm.text for exm in exm_lst]
        exm_lst = self.predictor_span.predict_span(batch_sents=batch_sents)  # 抽取指标
        for exm in exm_lst:
            print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct', 'source_id']))
            # input()

        mrc_input_lst = []
        for exm in exm_lst:
            sent = exm.text
            for itp_name, pos_lst in exm.pred_ent_dct.items():
                for itp_start, itp_end, prob in pos_lst:
                    mrc_input_lst.append([sent, itp_name.replace('指标-', ''), itp_start, itp_end])

        mrc_batch_sent, mrc_batch_itp = [], []
        for sent, itp_name, itp_start, itp_end in mrc_input_lst:
            mrc_batch_sent.append(sent)
            mrc_batch_itp.append([itp_name, itp_start, itp_end])
        exm_lst = self.predictor_multimrc.predict_multimrc(batch_sents=mrc_batch_sent, batch_itp=mrc_batch_itp)

        # for exm in exm_lst:
        #     print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct', 'source_id']))
        json_str = PipeLine_Infer.print_res(exm_lst)
        print(json_str)
        return json_str

    @staticmethod
    def print_res(exm_lst):
        # multimrc exm_lst
        text_dct = defaultdict(dict)
        for exm in exm_lst:
            text = exm.text
            itp_name = list(exm.ent_dct.keys())[0].replace('指标-', '')
            i_s, i_e, i_prob = list(exm.ent_dct.values())[0][0]
            itp = f'{itp_name} = {text[i_s:i_e]} = {i_s}|{i_e}'
            text_dct[text][itp] = {}
            atp_lst = []
            for atp_name, v in exm.pred_ent_dct.items():
                for a_s, a_e, a_prob in v:
                    atp_lst.append(f'{atp_name.replace("属性-", "")} = {text[a_s: a_e]} = {a_s}|{a_e}')
            atp_lst.sort()
            text_dct[text][itp] = atp_lst
        text_dct = dict(text_dct)
        json_str = json.dumps(text_dct, ensure_ascii=False, indent=2)
        return json_str

    def test(self):
        # exm_lst = utils.NerExample.load_from_jsonl('data/yanbao/sent/doc_exm.jsonl')
        # # print('doc_str:', doc_str)
        # print('=======预测doc')
        # self.predict(exm_lst[26].text)
        # print('=======预测doc')
        # self.predict(exm_lst[49].text)

        exm_lst = utils.NerExample.load_from_jsonl(f'data/yanbao/{self.domain}/test/doc_exm.jsonl')
        for exm in exm_lst:
            print('=======预测doc')
            self.predict(exm.text)

def multimrc_exm_to_json(exm):
    text = exm.text
    text_dct = {
        text: {},
    }
    itp_name = list(exm.ent_dct.keys())[0].replace('指标-', '')
    i_s, i_e, i_prob = list(exm.ent_dct.values())[0][0]
    itp = f'{itp_name} = {text[i_s:i_e]}'
    text_dct[text][itp] = {}
    atp_lst = []
    for atp_name, v in exm.pred_ent_dct.items():
        for a_s, a_e, a_prob in v:
            atp_lst.append(f'{atp_name.replace("属性-", "")} = {text[a_s: a_e]}')
    atp_lst.sort()
    text_dct[text][itp] = atp_lst
    json_str = json.dumps(text_dct, ensure_ascii=False, indent=2)
    # print(json_str)
    return json_str

if __name__ == "__main__":
    import readline

    parser = argparse.ArgumentParser("span_ner_infer")
    parser.add_argument('--info', help='information to distinguish model.', default='')
    parser.add_argument('--gpu', help='information to distinguish model.', default='0')
    parser.add_argument('--domain', help='specify domain', default='')
    args = parser.parse_args()

    domain = 'dianzi'
    # domain = 'yiyao'
    if not args.domain: args.domain = domain

    print(f'domain: {domain}')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用CPU设为'-1'
    print(f'gpu_id:{args.gpu}')

    pipeline_infer = PipeLine_Infer(args)
    pipeline_infer.test()
    exit(0)
    domain = 'dianzi'
    conf = {
        'dropout_rate': 0.2,
        # 'bert_model_dir': '/home/zyn/huggingface_model_resource/chinese-roberta-wwm-ext-large',
        'bert_model_dir': '/home/zyn/huggingface_model_resource/chinese-bert-wwm-ext',
        # 'bert_model_dir': '/home/zyn/huggingface_model_resource/bert-base-uncased',
        # 'bert_model_dir': '/home/zyn/huggingface_model_resource/bert-large-cased',
        # 'bert_model_dir': '/home/zyn/huggingface_model_resource/bert-base-chinese',
        # 'span_layer_type':'biaffine',
        # 'span_layer_type': 'tencent',
        # 'span_layer_type': 'self_attn',
        'span_layer_type': 'self_attn_mask_mean',
    }
    args.ckpt_dir = Path('span_ner_modules/ckpt')
    args.eval_every_step = 0
    args.eval_after_step = 0

    args.dev_dataset = None
    args.batch_size = 16
    datareader = NerDataReader(conf['bert_model_dir'], 512,
                               ent_file=f'data/yanbao/{domain}/full_ent_lst_remove_atp.txt',
                               atp_file=f'data/yanbao/{domain}/atp_lst.txt',
                               )
    args.char2id = datareader.char2id
    args.ent2id = datareader.ent2id
    args.tag2id = datareader.tag2id
    args.id2char = args.char2id.get_reverse()
    args.id2tag = args.tag2id.get_reverse()
    args.id2ent = args.ent2id.get_reverse()
    args.lr = 2e-5  # 原来1e-5
    # args.lr = 1e-3  # 原来1e-5
    args.num_epochs = 60

    # ====cls====
    # ckpt_path = Path('/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_07_21_47_cls_szb_cls_bert/best_test_model_8_3200.ckpt')
    # predictor_cls = Predictor(args, conf, ckpt_path=ckpt_path, datareader=datareader, mode='cls')
    # while True:
    #     sent = input('请输入:')
    #     batch_sents = [sent]
    #     time0 = time.time()
    #     predictor_cls.predict(batch_sents=batch_sents)
    #     print(f'elpased:{time.time() - time0}')

    # ====span=====
    # ckpt_path = Path('/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_10_19_37_span_self_attn_mask_mean_after_mul_szb_cls_bert/best_test_model_34_7446.ckpt')
    # predictor_span = Predictor(args, conf, ckpt_path=ckpt_path, datareader=datareader, mode='span')
    # while True:
    #     sent = input('请输入:')
    #     batch_sents = [sent]
    #     time0 = time.time()
    #     predictor_span.predict(batch_sents=batch_sents)
    #     print(f'elpased:{time.time() - time0}')

    # ====multimrc====
    ckpt_path = Path('/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_09_14_53_multimrc_contiu/best_test_model_6_594.ckpt')
    ckpt_path = Path('/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_09_15_09_multimrc_init/best_test_model_14_2758.ckpt')
    ckpt_path = Path('/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_09_15_09_multimrc_init/best_test_model_19_3743.ckpt')
    ckpt_path = Path('/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_09_15_09_multimrc_init/best_test_model_23_4531.ckpt')
    ckpt_path = Path('/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_09_15_09_multimrc_init/best_test_model_50_9850.ckpt')
    ckpt_path = Path('/home/zyn/MultiQueryMRC/span_ner_modules/dianzi/ckpt/21_06_17_23_12_multimrc_bert/best_test_model_5_2085.ckpt')

    predictor_multimrc = Predictor(args, conf, ckpt_path=ckpt_path, datareader=datareader, mode='multimrc')
    multimrc_test_samples = []

    multimrc_test_samples += [
        '全球智能手表出货量由2016年的2110万台快速增长至2019年9140万台，年复合增长率高达43.71%。||出货量||9140万台',
        '全球智能手表出货量由2016年的2110万台快速增长至2019年9140万台，年复合增长率高达43.71%。||出货量||2110万台',
        '3月下旬LCD价格加速上涨，1H21面板龙头盈利能力有望超预期改善根据WitsView数据，3月32/43/55/65寸面板价格达到75/126/202/252美元/片，环比上涨7%/4%/6%/5%，较20年5月已上涨134%/88%/96%/54%，3月环比增速较2月再提升，面板大厂的盈利能力有望在1Q21创下历史新高。||价格||75',
        '3月下旬LCD价格加速上涨，1H21面板龙头盈利能力有望超预期改善根据WitsView数据，3月32/43/55/65寸面板价格达到75/126/202/252美元/片，环比上涨7%/4%/6%/5%，较20年5月已上涨134%/88%/96%/54%，3月环比增速较2月再提升，面板大厂的盈利能力有望在1Q21创下历史新高。||价格||126',
        '3月下旬LCD价格加速上涨，1H21面板龙头盈利能力有望超预期改善根据WitsView数据，3月32/43/55/65寸面板价格达到75/126/202/252美元/片，环比上涨7%/4%/6%/5%，较20年5月已上涨134%/88%/96%/54%，3月环比增速较2月再提升，面板大厂的盈利能力有望在1Q21创下历史新高。||价格||202',
        '3月下旬LCD价格加速上涨，1H21面板龙头盈利能力有望超预期改善根据WitsView数据，3月32/43/55/65寸面板价格达到75/126/202/252美元/片，环比上涨7%/4%/6%/5%，较20年5月已上涨134%/88%/96%/54%，3月环比增速较2月再提升，面板大厂的盈利能力有望在1Q21创下历史新高。||价格||252美元/片',
        '3月下旬LCD价格加速上涨，1H21面板龙头盈利能力有望超预期改善根据WitsView数据，3月32/43/55/65寸面板价格达到75/126/202/252美元/片，环比上涨7%/4%/6%/5%，较20年5月已上涨134%/88%/96%/54%，3月环比增速较2月再提升，面板大厂的盈利能力有望在1Q21创下历史新高。||价格环比||7%',
        '3月下旬LCD价格加速上涨，1H21面板龙头盈利能力有望超预期改善根据WitsView数据，3月32/43/55/65寸面板价格达到75/126/202/252美元/片，环比上涨7%/4%/6%/5%，较20年5月已上涨134%/88%/96%/54%，3月环比增速较2月再提升，面板大厂的盈利能力有望在1Q21创下历史新高。||价格环比||4%',
        '3月下旬LCD价格加速上涨，1H21面板龙头盈利能力有望超预期改善根据WitsView数据，3月32/43/55/65寸面板价格达到75/126/202/252美元/片，环比上涨7%/4%/6%/5%，较20年5月已上涨134%/88%/96%/54%，3月环比增速较2月再提升，面板大厂的盈利能力有望在1Q21创下历史新高。||价格环比||6%',
        '3月下旬LCD价格加速上涨，1H21面板龙头盈利能力有望超预期改善根据WitsView数据，3月32/43/55/65寸面板价格达到75/126/202/252美元/片，环比上涨7%/4%/6%/5%，较20年5月已上涨134%/88%/96%/54%，3月环比增速较2月再提升，面板大厂的盈利能力有望在1Q21创下历史新高。||价格环比||5%',
        '根据前瞻产业研究院预测，2024年我国人工智能芯片市场规模将达785亿元，2019-2024年的CAGR为45.11%，约占全球市场的16.54%。||市占率||16.54%',
        '根据前瞻产业研究院预测，2024年我国人工智能芯片市场规模将达785亿元，2019-2024年的CAGR为45.11%，约占全球市场的16.54%。||市场规模||785亿元',
        '根据前瞻产业研究院预测，2024年我国人工智能芯片市场规模将达785亿元，2019-2024年的CAGR为45.11%，约占全球市场的16.54%。||市场规模年复合增长率||45.11%',
    ]

    multimrc_test_samples += [
        '我们预计，公司20/21/22年营收增长32%/66%/55%至7.3亿元/12.2亿元/18.8亿元，毛利率高位稳定在92%。||营业收入同比||32%',
        '我们预计，公司20/21/22年营收增长32%/66%/55%至7.3亿元/12.2亿元/18.8亿元，毛利率高位稳定在92%。||营业收入同比||66%',
        '我们预计，公司20/21/22年营收增长32%/66%/55%至7.3亿元/12.2亿元/18.8亿元，毛利率高位稳定在92%。||营业收入同比||55%',
        '我们预计，公司20/21/22年营收增长32%/66%/55%至7.3亿元/12.2亿元/18.8亿元，毛利率高位稳定在92%。||营业收入同比||7.3亿元',
        '我们预计，公司20/21/22年营收增长32%/66%/55%至7.3亿元/12.2亿元/18.8亿元，毛利率高位稳定在92%。||营业收入同比||12.2亿元',
        '我们预计，公司20/21/22年营收增长32%/66%/55%至7.3亿元/12.2亿元/18.8亿元，毛利率高位稳定在92%。||营业收入同比||18.8亿元',
        '我们预计，公司20/21/22年营收增长32%/66%/55%至7.3亿元/12.2亿元/18.8亿元，毛利率高位稳定在92%。||毛利率||92%',
        '2014年-2018年间：中国居民人均可支配收入从20,167元上涨至28,228元，期间年复合增长率为8.8%。||人均可支配收入||20,167元',
        '2014年-2018年间：中国居民人均可支配收入从20,167元上涨至28,228元，期间年复合增长率为8.8%。||人均可支配收入||28,228元',
        '人均消费支出从2014年的14,491元提升至2018年的19,853元，年复合增长率为8.2%。||人均消费支出||14,491元',
        '人均消费支出从2014年的14,491元提升至2018年的19,853元，年复合增长率为8.2%。||人均消费支出||19,853元',
    ]

    for sent in multimrc_test_samples:
        print(sent)
        sent, itp_name, itp_value = sent.split('||')
        itp_start = sent.find(itp_value)
        itp_end = itp_start + len(itp_value)
        batch_sents = [sent]
        batch_itp = [[itp_name, itp_start, itp_end]]
        time0 = time.time()
        predictor_multimrc.predict(batch_sents=batch_sents, batch_itp=batch_itp, verbose=True)
        print(f'elpased:{time.time() - time0}\n')

    # 指标
    while True:
        sent = input('请输入:')
        sent, itp_name, itp_value = sent.split('||')
        itp_start = sent.find(itp_value)
        itp_end = itp_start + len(itp_value)
        batch_sents = [sent]
        batch_itp = [[itp_name, itp_start, itp_end]]
        time0 = time.time()
        predictor_multimrc.predict(batch_sents=batch_sents, batch_itp=batch_itp, verbose=True)
        print(f'elpased:{time.time() - time0}\n')

    # ===pipeline span + mrc
    # while True:
    #     sent = input('请输入:')
    #     batch_sents = [sent]
    #     time0 = time.time()
    #     exm_lst = predictor_span.predict(batch_sents=batch_sents)
    #
    #     mrc_input_lst = []
    #     for exm in exm_lst:
    #         sent = exm.text
    #         for itp_name, pos_lst in exm.pred_ent_dct.items():
    #             for itp_start, itp_end, prob in pos_lst:
    #                 mrc_input_lst.append([sent, itp_name.replace('指标-', ''), itp_start, itp_end])
    #
    #     mrc_batch_sent, mrc_batch_itp = [], []
    #     for sent, itp_name, itp_start, itp_end in mrc_input_lst:
    #         mrc_batch_sent.append(sent)
    #         mrc_batch_itp.append([itp_name, itp_start, itp_end])
    #     predictor_multimrc.predict(batch_sents=mrc_batch_sent, batch_itp=mrc_batch_itp)
    #
    #     print(f'elpased:{time.time() - time0}')

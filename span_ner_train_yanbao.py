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
from span_ner import Trainer

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

from pathlib import Path
import datautils as utils
import argparse
from data_reader import NerDataReader
import sys
sys.path.append('.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser("span_ner_train_yanbao")
    parser.add_argument('--info', help='information to distinguish model.', default='')
    parser.add_argument('--gpu', help='select gpu to use.', default='0')
    parser.add_argument('--domain', help='specify domain', default='')
    parser.add_argument('--model', help='specify model type', default='')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu  # 使用CPU设为'-1'
    print(f'gpu_id:{args.gpu}')

    conf = {
        'dropout_rate': 0.2,
        # 'dropout_rate': 0.5,
        'bert_model_dir': '/home/zyn/huggingface_model_resource/chinese-bert-wwm-ext',
        'span_layer_type': 'self_attn_mask_mean',
        'span_model_negsample_rate': 0.3  # 0.3  # 0不进行负采样 非0 按比例进行负采样
    }
    # yanbao
    domain = 'dianzi'
    domain = 'yiyao'
    if args.domain: domain = args.domain
    print(f'domain: {domain}')

    args.ckpt_dir = Path(f'data/yanbao/{domain}/ckpt')
    # args.ckpt_dir = Path(f'span_ner_modules/{domain}/ckpt')

    def train_cls(args, conf, domain):
        args.eval_every_step = 0
        args.eval_after_step = 0

        datareader = NerDataReader(conf['bert_model_dir'], 512,
                                   ent_file=f'data/yanbao/{domain}/full_ent_lst_remove_atp.txt',
                                   )
        args.char2id = datareader.char2id
        args.ent2id = datareader.ent2id
        args.tag2id = datareader.tag2id
        args.id2char = args.char2id.get_reverse()
        args.id2tag = args.tag2id.get_reverse()
        args.id2ent = args.ent2id.get_reverse()

        train_exm_lst = utils.NerExample.load_from_jsonl(f'data/yanbao/{domain}/train/sent_exm.jsonl', external_attrs=['source_id'])
        test_exm_lst = utils.NerExample.load_from_jsonl(f'data/yanbao/{domain}/test/sent_exm.jsonl', external_attrs=['source_id'])
        args.train_dataset = datareader.build_dataset(train_exm_lst, mode='cls')
        args.test_dataset = datareader.build_dataset(test_exm_lst, mode='cls')
        args.dev_dataset = None
        args.datareader = datareader

        args.batch_size = 16
        args.lr = 2e-5  # 原来1e-5
        args.num_epochs = 40

        trainer = Trainer(args, conf, model_type='cls')

    def eval_cls(args, conf, domain):
        exist_ckpt = list(sorted(list(args.ckpt_dir.glob('*cls*')), key=lambda p: p.stat().st_ctime, reverse=False)[0].glob('*.ckpt'))[0]
        args.eval_every_step = 0
        args.eval_after_step = 0

        datareader = NerDataReader(conf['bert_model_dir'], 512,
                                   ent_file=f'data/yanbao/{domain}/full_ent_lst_remove_atp.txt',
                                   )
        args.char2id = datareader.char2id
        args.ent2id = datareader.ent2id
        args.tag2id = datareader.tag2id
        args.id2char = args.char2id.get_reverse()
        args.id2tag = args.tag2id.get_reverse()
        args.id2ent = args.ent2id.get_reverse()
        args.datareader = datareader

        # pred cls
        args.batch_size = 64
        args.lr = 2e-5  # 原来1e-5
        args.num_epochs = 100

        train_exm_lst = utils.NerExample.load_from_jsonl(f'data/yanbao/{domain}/train/sent_exm.jsonl', external_attrs=['source_id', 'cls_tgt'])
        test_exm_lst = utils.NerExample.load_from_jsonl(f'data/yanbao/{domain}/test/sent_exm.jsonl', external_attrs=['source_id', 'cls_tgt'])
        num_train = len(train_exm_lst)
        total_exm_lst = train_exm_lst + test_exm_lst
        args.infer_dataset = datareader.build_dataset(total_exm_lst, mode='cls')
        trainer = Trainer(args, conf, model_type='cls', exist_ckpt=exist_ckpt, evaluate=True)
        exm_lst = trainer.test_data_loader.dataset.instances
        utils.NerExample.save_to_jsonl(exm_lst[:num_train], f'data/yanbao/{domain}/train/sent_exm_cls_pred.jsonl', external_attrs=['cls_pred', 'cls_tgt', 'source_id'])
        utils.NerExample.save_to_jsonl(exm_lst[num_train:], f'data/yanbao/{domain}/test/sent_exm_cls_pred.jsonl', external_attrs=['cls_pred', 'cls_tgt', 'source_id'])


    def train_span(args, conf, domain):
        args.eval_every_step = 0
        args.eval_after_step = 500

        datareader = NerDataReader(conf['bert_model_dir'], 512,
                                   ent_file=f'data/yanbao/{domain}/full_ent_lst_remove_atp.txt',
                                   )
        args.char2id = datareader.char2id
        args.ent2id = datareader.ent2id
        args.tag2id = datareader.tag2id
        args.id2char = args.char2id.get_reverse()
        args.id2tag = args.tag2id.get_reverse()
        args.id2ent = args.ent2id.get_reverse()
        args.datareader = datareader

        # =====after single_zb cls
        train_exm_lst = utils.NerExample.load_from_jsonl(f'data/yanbao/{domain}/train/sent_exm_cls_pred.jsonl', external_attrs=['cls_pred', 'cls_tgt', 'source_id'])
        train_exm_lst = [exm for exm in train_exm_lst if exm.cls_pred[0] == 1]
        test_exm_lst = utils.NerExample.load_from_jsonl(f'data/yanbao/{domain}/test/sent_exm_cls_pred.jsonl', external_attrs=['cls_pred', 'cls_tgt', 'source_id'])
        test_exm_lst = [exm for exm in test_exm_lst if exm.cls_pred[0] == 1]

        args.train_dataset = datareader.build_dataset(train_exm_lst, mode='span')
        args.test_dataset = datareader.build_dataset(test_exm_lst, mode='span')
        args.dev_dataset = None
        args.datareader = datareader

        args.batch_size = 8
        args.lr = 2e-5  # 原来1e-5
        args.num_epochs = 100

        trainer = Trainer(args, conf, model_type='span')
        # trainer = Trainer(args, conf, model_type='seq')

    def train_multimrc(args, conf, domain):
        args.eval_every_step = 0
        args.eval_after_step = 500

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

        train_exm_lst = utils.NerExample.load_from_jsonl(f'data/yanbao/{domain}/train/doc_exm_w_re_sent_singlezb.jsonl', external_attrs=['itp2atp_dct', 'source_id'])
        test_exm_lst = utils.NerExample.load_from_jsonl(f'data/yanbao/{domain}/test/doc_exm_w_re_sent_singlezb.jsonl', external_attrs=['itp2atp_dct', 'source_id'])
        args.train_dataset = datareader.build_dataset(train_exm_lst, mode='multimrc')
        args.test_dataset = datareader.build_dataset(test_exm_lst, mode='multimrc')
        args.dev_dataset = None
        args.datareader = datareader

        args.batch_size = 8
        args.lr = 2e-5  # 原来1e-5
        args.num_epochs = 100

        trainer = Trainer(args, conf, model_type='multimrc')
        # trainer = Trainer(args, conf, model_type='multimrc', exist_ckpt='/home/zyn/MultiQueryMRC/span_ner_modules/dianzi/ckpt/21_06_17_23_12_multimrc_bert/best_test_model_5_2085.ckpt', evaluate=True)
        # trainer = Trainer(args, conf, model_type='multimrc', exist_ckpt='/home/zyn/MultiQueryMRC/span_ner_modules/ckpt/21_06_08_22_07_multimrc_test/best_test_model_20_1980.ckpt', evaluate=False)

    if args.model == 'cls':#在句子级别上对指标进行二分类，判断该句子是否存在指标
        train_cls(args, conf, domain)
    elif args.model == 'pred_cls':
        eval_cls(args, conf, domain)
    elif args.model == 'span':#对指标进行span-base-ner识别，对存在指标的句子进行实体识别
        train_span(args, conf, domain)
    elif args.model == 'multimrc':#通过阅读理解模型进行属性抽取
        train_multimrc(args, conf, domain)
    else:
        print(f'请检查--model参数是否有误, 当前指定值为:args.model')
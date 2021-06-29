import os
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

import json
from data_reader import NerDataReader


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

        self.predictor_cls = torch.load(f"./model_save/{self.domain}_predictor_cls.pth")

        # ====span=====
        self.predictor_span = torch.load(f"./model_save/{self.domain}_predictor_span.pth")


        # ====multimrc====
        self.predictor_multimrc = torch.load(f"./model_save/{self.domain}_predictor_multimrc.pth")


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
        json_str = PipeLine_Infer_1.print_res(exm_lst)
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

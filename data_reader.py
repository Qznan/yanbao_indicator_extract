#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据处理

Author: v_zhouxiaojin@baidu.com
"""

import json
import time
import os
import copy
from typing import *
import unicodedata
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
# from utils import Dataset, LazyDataset
from transformers import BertTokenizer
from collections import namedtuple, defaultdict
from functools import partial
import random
import datautils as utils
from datautils import NerExample
from prefetch_generator import BackgroundGenerator  # prefetch-generator


def get_my_matrix(batch_size, length):
    matrix_lst = []
    for i in range(length - 1):
        # print('get_my_matrix func for', i)
        mat = np.zeros([length, length], dtype='float')
        mat[:i + 1, i + 1:] = 1.
        matrix_lst.append(mat)
    matrix_lst = np.stack(matrix_lst, 0)  # l-1, l,l
    # matrix_lst = np.tile(matrix_lst[None, ...], (batch_size, 1, 1, 1))
    return matrix_lst


class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def is_whitespace(char):
    """判断是否为空字符"""
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


class CharTokenizer(BertTokenizer):
    """构造字符级tokenizer，添加对空字符的支持"""

    def tokenize(self, text, **kwargs):
        """tokenize by char"""
        token_list = []
        for c in text:
            if c in self.vocab:
                token_list.append(c)
            elif is_whitespace(c):
                token_list.append('[unused1]')
            else:
                token_list.append(self.unk_token)
        return token_list


class Example(object):
    """构造Example"""

    def __init__(self,
                 input_id,
                 sent_id,
                 query_mask,
                 attn_mask,
                 answers=None,
                 ):
        self.input_id = input_id
        self.sent_id = sent_id
        self.query_mask = query_mask
        self.attn_mask = attn_mask
        self.answers = answers


class DataReader(object):
    """数据构造器"""

    def __init__(self, tokenizer_path, max_len):
        self.tokenizer = CharTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len
        self.sep_token = '[SEP]'
        self.cls_token = '[CLS]'
        self.char2id = utils.Any2Id(exist_dict=self.tokenizer.vocab)

    def diff_query_mask(self, query_tokens, query_mask):
        """根据query的diff字符，构建query mask"""
        all_query_tokens = defaultdict(int)
        for tok in query_tokens:
            all_query_tokens[tok] += 1
        all_query_tokens = dict(all_query_tokens)
        for tdx, tok in enumerate(query_tokens):
            if all_query_tokens[tok] < 2:
                query_mask[tdx] = 1

    def build_query_attn_mask(self, query_mask):
        """构造attention mask的query部分"""
        cusum = np.cumsum(query_mask)
        attn_mask = cusum[None, :] == cusum[:, None]  # 还可以这样expand?

        return attn_mask.astype('int')

    def truncate_and_pad(self, array, length, pad_value=0):
        """截断/pad到指定长度"""
        if len(array.shape) == 1:
            array = array[:length]

        elif len(array.shape) == 2:
            array = array[:length, :length]

        array = np.pad(array, [0, length - len(array)], mode='constant', constant_values=pad_value)  # 分别在每一维前后补0和remainlen
        return array

    def wrapper(self, instance, shuffle_query=False, predict=False):
        """warpper"""
        context = instance['context']
        info = instance['event_type']
        queries = instance['queries']
        # 这里可以对query乱序
        if shuffle_query:
            random.shuffle(queries)
        query_mask = np.zeros(self.max_len, dtype=int)
        tokens_a = []
        for q in queries:
            query_mask[len(tokens_a)] = 1
            tokens = self.tokenizer.tokenize(q['query'])
            tokens_a.extend([self.cls_token] + tokens)
        attn_mask = self.build_query_attn_mask(query_mask[:len(tokens_a)])  # query只跟自己attn
        attn_mask = attn_mask[:self.max_len][:, :self.max_len]  # 矩形长宽限制
        # 在第一个片段添加额外的info, 比如事件类型
        tokens_a.extend([self.sep_token] + self.tokenizer.tokenize(info) + [self.sep_token])
        input_id_a = self.tokenizer.convert_tokens_to_ids(tokens_a)
        # 构造正文部分
        tokens_b = self.tokenizer.tokenize(context) + [self.sep_token]
        input_id_b = self.tokenizer.convert_tokens_to_ids(tokens_b)
        sent_id = np.zeros(len(input_id_a), dtype=int)
        input_id = np.array(input_id_a + input_id_b, dtype=int)
        attn_mask = self.truncate_and_pad(attn_mask, len(input_id), 1)

        sent_id = self.truncate_and_pad(sent_id, self.max_len, 1)
        input_id = self.truncate_and_pad(input_id, self.max_len, self.tokenizer.pad_token_id)
        attn_mask = self.truncate_and_pad(attn_mask, self.max_len, 0)
        if predict:
            return Example(input_id, sent_id, query_mask, attn_mask)

        num_query = len(queries)
        answers = np.zeros((num_query, self.max_len, 2), dtype=int)
        sent_a_length = self.max_len - sent_id.sum()
        for i, query in enumerate(queries):
            for answer in query['answers']:
                text = answer['text']
                start_index = answer['start_index'] + sent_a_length
                if start_index >= self.max_len: continue  # start在512之后 不要这个答案
                end_index = start_index + len(text) - 1
                answers[i][start_index][0] = 1
                if end_index >= self.max_len: continue  # 如果没有end即start到结尾都是答案
                answers[i][end_index][1] = 1

        return Example(input_id, sent_id, query_mask, attn_mask, answers)

    def build_dataset(self, data_source, shuffle_query=False, predict=False):
        """构造数据集"""
        instances = []
        if isinstance(data_source, (str, Path)):
            with open(data_source) as f:
                for line in tqdm(f):
                    instance = json.loads(line)
                    instances.append(instance)
        else:
            instances = data_source
        wrapper = partial(self.wrapper, shuffle_query=shuffle_query, predict=predict)

        return LazyDataset(instances, wrapper)


Batch = namedtuple('Batch', ['input_ids', 'sent_ids', 'query_masks', 'attn_masks', 'answers'])


def batcher(device='cpu', status='train'):
    """
    batch构造
    用于DataLoader中的collate_fn
    """

    def numpy_to_tensor(array):
        """numpy转换成torch tensor"""
        return torch.from_numpy(array).to(device)

    def batcher_fn(batch):
        """batcher_fn"""

        batch_input_id = []
        batch_sent_id = []
        batch_query_mask = []
        batch_attn_mask = []
        batch_answer = []
        for instance in batch:
            batch_input_id.append(instance.input_id)
            batch_sent_id.append(instance.sent_id)
            batch_query_mask.append(instance.query_mask)
            batch_attn_mask.append(instance.attn_mask)
            if status == 'decode':
                continue
            else:
                batch_answer.append(instance.answers)
        batch_input_id = numpy_to_tensor(np.stack(batch_input_id).astype('int64'))  # concat 和 stack默认都是第一维dim=0
        batch_sent_id = numpy_to_tensor(np.stack(batch_sent_id).astype('int64'))
        batch_query_mask = numpy_to_tensor(np.stack(batch_query_mask).astype('int64'))
        batch_attn_mask = numpy_to_tensor(np.stack(batch_attn_mask).astype('float32'))
        if batch_answer:
            batch_answer = numpy_to_tensor(np.concatenate(batch_answer).astype('float32'))  # concat not stack
        # print('batch_input_id', batch_input_id.shape)  # bat,len
        # print('batch_sent_id', batch_sent_id.shape)  # bat,len
        # print('batch_query_mask', batch_query_mask.shape)  # bat,len
        # print('batch_attn_mask', batch_attn_mask.shape)  # bat,len,len
        # print('batch_answer', batch_answer.shape)  # bat*num_query,len,2
        batch_pack = Batch(batch_input_id, batch_sent_id, batch_query_mask, batch_attn_mask, batch_answer)
        return batch_pack

    return batcher_fn


class MultiMrc_Postprocessor:
    def __init__(self, tokenizer, max_len, sep_token, cls_token, char2id):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sep_token = sep_token
        self.cls_token = cls_token
        self.char2id = char2id

    def diff_query_mask(self, query_tokens, query_mask):
        """根据query的diff字符，构建query mask"""
        all_query_tokens = defaultdict(int)
        for tok in query_tokens:
            all_query_tokens[tok] += 1
        all_query_tokens = dict(all_query_tokens)
        for tdx, tok in enumerate(query_tokens):
            if all_query_tokens[tok] < 2:
                query_mask[tdx] = 1

    def build_query_attn_mask(self, query_mask):
        """构造attention mask的query部分"""
        cusum = np.cumsum(query_mask)
        attn_mask = cusum[None, :] == cusum[:, None]  # 还可以这样expand?

        return attn_mask.astype('int')

    def truncate_and_pad(self, array, length, pad_value=0):
        """截断/pad到指定长度"""
        if len(array.shape) == 1:
            array = array[:length]

        elif len(array.shape) == 2:
            array = array[:length, :length]

        array = np.pad(array, [0, length - len(array)], mode='constant', constant_values=pad_value)  # 分别在每一维前后补0和remainlen
        return array

    def process(self, exm: NerExample, all_atps, shuffle_query=False, train=True):
        # # 根据itp2atp_dct 先把无关的实体都过滤了 根据itp2atp_dct也仅保留当前句子存在的实体
        # # print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct','source_id']))
        # ent_lst = exm.get_ent_lst()
        # itp2atp_dct_keep_ent_ids = set()
        # for itp_id, atps_ids in exm.itp2atp_dct.items():
        #     itp2atp_dct_keep_ent_ids.add(int(itp_id))
        #     for atp_id in atps_ids:
        #         itp2atp_dct_keep_ent_ids.add(atp_id)
        # ent_lst = [e for e in ent_lst if e[3] in itp2atp_dct_keep_ent_ids]
        # exm.ent_dct = NerExample.ent_lst_to_ent_dct(ent_lst)
        #
        # # 已经过滤了
        # ent_lst_kepp_ent_ids = set(e[3] for e in ent_lst)
        # for itp_id, atps_ids in exm.itp2atp_dct.items():
        #     exm.itp2atp_dct[itp_id] = [atp_id for atp_id in atps_ids if atp_id in ent_lst_kepp_ent_ids]
        # # print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct', 'source_id']))
        # # input()
        # # ==根据itp2atp_dct 先把无关的实体都过滤了
        #
        # context = exm.text
        # itp2atp_dct = exm.itp2atp_dct
        # itp_atps_lst = []  # [[指标名,start,end,id],[属性名,start,end,id],[属性名,start,end,id]...]
        # ent_lst = exm.get_ent_lst()
        # # print(exm)
        # # print(ent_lst)s
        # ent_id_2_ent = {e[3]: e for e in ent_lst}
        # # print(ent_id_2_ent)
        # # print(itp2atp_dct)
        # for itp_id, atps_ids in itp2atp_dct.items():
        #     itp_id = int(itp_id)
        #     if itp_id in ent_id_2_ent:
        #         itp_atps = [ent_id_2_ent[itp_id]]
        #         for atp_id in atps_ids:  # 可能有些属性不在当前句子
        #             if atp_id in ent_id_2_ent:
        #                 itp_atps.append(ent_id_2_ent[atp_id])
        #             else:
        #                 # print('该属性在当前句子中不存在')
        #                 pass
        #         itp_atps_lst.append(itp_atps)
        # # print(itp_atps_lst)

        # for itp_atps in itp_atps_lst:  # 只会循环一次 因为每个样本只有1个指标
        # input()
        # context = exm.text
        # ent_lst = exm.get_ent_lst()
        # itp_lst = [e for e in ent_lst if e[0].startswith('指标')]
        # assert len(itp_lst) == 1  # 每个exm只放1个指标
        # itp_name, itp_start, itp_end, *_ = itp_lst[0]
        # itp_name = itp_name.replace('指标-', '')
        #
        # atp_lst = [e for e in ent_lst if e[0].startswith('属性')]
        #
        # itp2atp_dct = exm.itp2atp_dct
        # itp_name, itp_start, itp_end, *_ = itp_atps[0]
        # itp_value = context[itp_start:itp_end]
        # atp_name_2_start_end = {atp_name.replace('属性-', ''): [atp_start, atp_end] for atp_name, atp_start, atp_end, *_ in itp_atps[1:]}

        context = exm.text
        ent_lst = exm.get_ent_lst()
        itp_lst = [e for e in ent_lst if e[0].startswith('指标')]
        atp_lst = [e for e in ent_lst if e[0].startswith('属性')]
        assert len(itp_lst) == 1  # 每个exm只放1个指标
        itp_name, itp_start, itp_end, *_ = itp_lst[0]
        itp_name = itp_name.replace('指标-', '')
        itp_value = context[itp_start:itp_end]
        queries = {atp_name: [] for atp_name in all_atps}  # {'来源':[101, 104], '项目':[], ...}
        for atp_name, atp_start, atp_end, *_ in atp_lst:
            atp_name = atp_name.replace('属性-', '')
            queries[atp_name] = [atp_start, atp_end]

        # print(exm.to_json_str(for_human_read=True, external_attrs=['itp2atp_dct', 'source_id']))
        # print(queries)
        query_mask = np.zeros(self.max_len, dtype='int')
        # 构造query和其他信息部分 tokens_a
        tokens_a = []  # 放query,如: [cls]行业[cls]项目[sep]增长率60%[sep]
        for atp_name in queries:
            query_mask[len(tokens_a)] = 1
            tokens_a.extend([self.cls_token] + self.tokenizer.tokenize(atp_name))

        attn_mask = self.build_query_attn_mask(query_mask[:len(tokens_a)])  # query只跟自己attn
        attn_mask = attn_mask[:self.max_len][:, :self.max_len]  # 矩形长宽限制

        # 在第一个片段添加额外的info, 如:指标名及指标值
        tokens_a.extend([self.sep_token] + self.tokenizer.tokenize(itp_name) + self.tokenizer.tokenize(itp_value) + [self.sep_token])
        # print(''.join(tokens_a))
        input_id_a = self.tokenizer.convert_tokens_to_ids(tokens_a)

        # 构造正文部分 token_b 如: text[sep]
        tokens_b = self.tokenizer.tokenize(context) + [self.sep_token]
        # print(''.join(tokens_b))
        input_id_b = self.tokenizer.convert_tokens_to_ids(tokens_b)

        # 总的
        input_id = np.array(input_id_a + input_id_b, dtype='int')
        attn_mask = self.truncate_and_pad(attn_mask, len(input_id), 1)  # 补1 与context进行attn
        token_type_id = np.zeros(len(input_id_a), dtype='int')  # query是0 context是1
        token_type_id = self.truncate_and_pad(token_type_id, self.max_len, 1)
        input_id = self.truncate_and_pad(input_id, self.max_len, self.tokenizer.pad_token_id)
        attn_mask = self.truncate_and_pad(attn_mask, self.max_len, 0)

        answers = None
        if train:
            num_query = len(all_atps)
            answers = np.zeros((num_query, self.max_len, 2), dtype='int')
            sent_a_length = self.max_len - token_type_id.sum()
            for i, atp_name in enumerate(all_atps):
                # for answer in query['answers']:  # 每个指标属性值只有1个
                if queries[atp_name]:
                    start, end = queries[atp_name]
                    start = start + sent_a_length
                    if start >= self.max_len: continue  # start在512之后 不要这个答案
                    end = end + sent_a_length
                    end = end - 1  # 闭区间
                    answers[i][start][0] = 1
                    if end >= self.max_len: continue  # 如果没有end即start到结尾都是答案
                    answers[i][end][1] = 1
            # print(answers.tolist())

            atp_res = []
            for prob_, atp_name in zip(answers, all_atps):
                # prob_ [len,2]
                if sum(np.max(prob_, axis=0) >= 0.5) == 2:  # 有start和end
                    pos = np.argmax(prob_, axis=0)  # [2]
                    start, end = pos.tolist()
                    if start <= end:
                        atp_res.append([atp_name, start, end + 1])
            pred_ent_dct = {e[0]: [e[1], e[2]] for e in atp_res}
            # print(pred_ent_dct)

        # print(input_id, token_type_id, attn_mask, query_mask, answers)
        return input_id, token_type_id, attn_mask, query_mask, answers


def yb_ent_convert(ent_type):
    if ent_type.startswith('指标-'):
        ent_type = '指标'
    return ent_type


class NerDataReader():
    def __init__(self, tokenizer_path, max_len, ent_file, atp_file=None):
        self.tokenizer = CharTokenizer.from_pretrained(tokenizer_path)
        self.max_len = max_len
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.pad_token = self.tokenizer.pad_token
        self.sep_id = self.tokenizer.sep_token_id
        self.cls_id = self.tokenizer.cls_token_id
        self.pad_id = self.tokenizer.pad_token_id

        self.char2id = utils.Any2Id(exist_dict=self.tokenizer.vocab)
        self.ent2id = utils.Any2Id.from_file(ent_file, use_line_no=True)
        print('ent2id', self.ent2id)
        tag2id = {'[PAD]': 0, 'O': 1}
        # tag2id = {}
        for ent_tp in self.ent2id:
            if ent_tp not in ['[PAD]', 'O']:
                tag2id[f'B-{ent_tp}'] = len(tag2id)
                tag2id[f'I-{ent_tp}'] = len(tag2id)
        self.tag2id = utils.Any2Id(exist_dict=tag2id)
        # print(self.tag2id)

        if atp_file is not None:
            self.all_atps = [atp.replace('属性-', '') for atp in utils.file2list(atp_file)]
        else:
            self.all_atps = ['上游', '下游', '业务', '产品', '公司', '区域', '品牌', '客户', '市场', '年龄', '性别', '时间', '机构', '来源', '渠道', '行业', '项目']  # yimei
        print('all_atps', self.all_atps)

    def multimrc_post_process(self, exm: NerExample, train=True):
        if not hasattr(self, 'multi_mrc_postprocessor'):
            self.multimrc_postprocessor = MultiMrc_Postprocessor(self.tokenizer, self.max_len, self.sep_token, self.cls_token, self.char2id)
        if not hasattr(exm, 'train_cache'):
            input_id, token_type_id, attn_mask, query_mask, answers = self.multimrc_postprocessor.process(exm, all_atps=self.all_atps, train=train)
            exm.train_cache = {
                'input_ids': input_id,
                'token_type_id': token_type_id,
                'attn_mask': attn_mask,
                'query_mask': query_mask,
                'answers': answers,
            }
        return dict(
            {
                'ner_exm': exm,
            }, **exm.train_cache)

    def cls_post_process(self, exm: NerExample, train=True):
        if not hasattr(exm, 'cls_tgt'):
            # exm.cls_tgt = int(not exm.is_neg())
            exm.cls_tgt = int(exm.has_ent_type_startswith('指标'))
        if not hasattr(exm, 'train_cache'):
            input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(exm.char_lst))
            input_ids = [self.cls_id] + input_ids + [self.sep_id]
            exm.train_cache = {
                'input_ids': input_ids,
                'len': len(input_ids),
            }
        return dict(
            {
                'cls_tgt': exm.cls_tgt,
                'ner_exm': exm,
            }, **exm.train_cache)

    def span_post_process_softmax(self, exm: NerExample, train=True):
        # softmax
        input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(exm.char_lst))
        input_ids = [self.cls_id] + input_ids + [self.sep_id]
        if train:
            # ===数据特殊处理
            tmp_exm = copy.deepcopy(exm)
            tmp_exm.filter_ent_by_startswith('指标', mode='keep')
            # tmp_exm.ent_type_convert(yb_ent_convert)
            span_ner_tgt_lst = tmp_exm.get_span_level_ner_tgt_lst(neg_symbol='O')
            # ===数据特殊处理
            # span_ner_tgt_lst = exm.get_span_level_ner_tgt_lst(neg_symbol='O')
            span_ner_tgt_lst = [self.ent2id[e] for e in span_ner_tgt_lst]
        return {
            'input_ids': input_ids,
            'len': len(input_ids),
            'span_ner_tgt_lst': span_ner_tgt_lst if train else [],
            'ner_exm': exm,
        }

    def span_post_process_softmax_ENG(self, exm: NerExample, train=True):
        # softmax
        input_ids = self.tokenizer.convert_tokens_to_ids(exm.char_lst)
        input_ids = [self.cls_id] + input_ids + [self.sep_id]
        if train:
            span_ner_tgt_lst = exm.get_span_level_ner_tgt_lst(neg_symbol='O')
            span_ner_tgt_lst = [self.ent2id[e] for e in span_ner_tgt_lst]
        return {
            'input_ids': input_ids,
            'len': len(input_ids),
            'span_ner_tgt_lst': span_ner_tgt_lst if train else [],
            'ner_exm': exm,
        }

    def span_post_process_sigmoid(self, exm: NerExample, train=True):
        # onehot
        if not hasattr(exm, 'train_cache'):
            input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(exm.char_lst))
            input_ids = [self.cls_id] + input_ids + [self.sep_id]
            if train:
                span_ner_tgt_lst = exm.get_span_level_ner_tgt_lst(neg_symbol='O')
                # use onehot
                span_ner_onehot_tgt_lst = []
                for tag in span_ner_tgt_lst:
                    if tag == 'O':
                        span_ner_onehot_tgt_lst.append(np.zeros([len(self.ent2id)], dtype='float'))
                    else:
                        span_ner_onehot_tgt_lst.append(np.eye(len(self.ent2id), dtype='float')[self.ent2id[tag]])
            exm.train_cache = {
                'input_ids': input_ids,
                'len': len(input_ids),
                'span_ner_tgt_lst': span_ner_onehot_tgt_lst if train else [],
            }
        return {
            'input_ids': exm.train_cache['input_ids'],
            'len': exm.train_cache['len'],
            'span_ner_tgt_lst': exm.train_cache['span_ner_tgt_lst'],
            'ner_exm': exm,
        }

    def seq_post_process(self, exm: NerExample, train=True):
        if not hasattr(exm, 'train_cache'):
            input_ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(exm.char_lst))
            input_ids = [self.cls_id] + input_ids + [self.sep_id]
            if train:
                # ===数据特殊处理
                tmp_exm = copy.deepcopy(exm)
                tmp_exm.ent_type_convert(yb_ent_convert)
                filter_ent_dct = tmp_exm.get_filter_ent_dct_by_startswith('指标', mode='keep')
                tag_lst = NerExample.to_tag_lst(exm.char_lst, filter_ent_dct)
                # ===数据特殊处理
                # tag_lst = NerExample.to_tag_lst(exm.char_lst, exm.ent_dct)
                tag_ids = [self.tag2id[tag] for tag in tag_lst]
            exm.train_cache = {
                'input_ids': input_ids,
                'len': len(input_ids),
                'tag_ids': tag_ids if train else [],
                'ner_exm': exm,
            }
        return dict(
            {
                'ner_exm': exm,
            }, **exm.train_cache)

    def get_batcher_fn(self, gpu=False, mode='span', train=True):

        def tensorize(array, dtype='int'):
            if isinstance(array, np.ndarray):
                ret = torch.from_numpy(array)
            else:
                if dtype == 'int':
                    ret = torch.LongTensor(array)
                elif dtype == 'float':
                    ret = torch.FloatTensor(array)
                elif dtype == 'double':
                    ret = torch.DoubleTensor(array)
                else:
                    raise NotImplementedError
            if gpu:
                ret = ret.cuda()
            return ret

        def span_batcher(batch_e):
            # batch_start_time = time.time()
            max_len = max([e['len'] for e in batch_e])
            batch_seq_len = []
            batch_input_ids = []
            batch_span_ner_tgt_lst = []
            batch_neg_span_mask = []
            batch_bert_token_type_ids = []
            batch_bert_attention_mask = []
            batch_ner_exm = []
            for e in batch_e:
                batch_seq_len.append(e['len'] - 2)  # 去除cls和sep后的长度
                batch_input_ids.append(e['input_ids'] + [self.pad_id] * (max_len - e['len']))
                batch_bert_token_type_ids.append([0] * max_len)  # seg0
                batch_bert_attention_mask.append([1] * e['len'] + [0] * (max_len - e['len']))
                batch_ner_exm.append(e['ner_exm'])
                if train:
                    batch_span_ner_tgt_lst.extend(e['span_ner_tgt_lst'])
                    # neg_sample mask
                    # span_mask = (np.sum(e['span_ner_tgt_lst'], axis=-1) != 0.).astype('float')  # 负样本mask=0
                    # sample_neg_indices = [i for i, e in enumerate(span_mask) if e == 0.]
                    # num_sample = int(e['len'] - 2 * 0.35 + 0.5)
                    # sample_neg_indices = random.sample(sample_neg_indices, num_sample)
                    # span_mask[sample_neg_indices] = 1.  # 采样的负样本mask变为1
                    # batch_neg_span_mask.extend(span_mask)
            # print('batcher time', time.time() - batch_start_time)
            return {
                'seq_len': tensorize(batch_seq_len),
                'input_ids': tensorize(batch_input_ids),
                'span_ner_tgt_lst': tensorize(batch_span_ner_tgt_lst, dtype='int'),  # softmax
                # 'span_ner_tgt_lst': tensorize(batch_span_ner_tgt_lst, dtype='float'),  # sigmoid
                # 'neg_span_mask': tensorize(batch_neg_span_mask, dtype='float'),
                'bert_token_type_ids': tensorize(batch_bert_token_type_ids),
                'bert_attention_mask': tensorize(batch_bert_attention_mask),
                'batch_ner_exm': batch_ner_exm,
                # 'mask_matrix': tensorize(get_my_matrix(len(batch_e), max_len-2), dtype='float'),
            }

        def seq_batcher(batch_e):
            max_len = max([e['len'] for e in batch_e])
            batch_seq_len = []
            batch_input_ids = []
            batch_tag_ids = []
            batch_bert_token_type_ids = []
            batch_bert_attention_mask = []
            batch_ner_exm = []
            for e in batch_e:
                batch_seq_len.append(e['len'] - 2)
                batch_input_ids.append(e['input_ids'] + [self.pad_id] * (max_len - e['len']))
                batch_bert_token_type_ids.append([0] * max_len)  # seg0
                batch_bert_attention_mask.append([1] * e['len'] + [0] * (max_len - e['len']))
                batch_ner_exm.append(e['ner_exm'])
                if train:
                    # batch_tag_ids.append(e['tag_ids'] + [self.tag2id['[PAD]']] * (max_len - 2 - e['len']))  # len本来已有2 不需要再减 应该是下面形式
                    batch_tag_ids.append(e['tag_ids'] + [self.tag2id['[PAD]']] * (max_len - e['len']))

            return {
                'seq_len': tensorize(batch_seq_len),
                'input_ids': tensorize(batch_input_ids),
                'tag_ids': tensorize(batch_tag_ids),
                'bert_token_type_ids': tensorize(batch_bert_token_type_ids),
                'bert_attention_mask': tensorize(batch_bert_attention_mask),
                'batch_ner_exm': batch_ner_exm,
            }

        def cls_batcher(batch_e):
            max_len = max([e['len'] for e in batch_e])
            batch_seq_len = []
            batch_input_ids = []
            batch_cls_tgt = []
            batch_bert_token_type_ids = []
            batch_bert_attention_mask = []
            batch_ner_exm = []
            for e in batch_e:
                batch_seq_len.append(e['len'] - 2)
                batch_input_ids.append(e['input_ids'] + [self.pad_id] * (max_len - e['len']))
                batch_bert_token_type_ids.append([0] * max_len)  # seg0
                batch_bert_attention_mask.append([1] * e['len'] + [0] * (max_len - e['len']))
                batch_ner_exm.append(e['ner_exm'])
                if train:
                    batch_cls_tgt.append(e['cls_tgt'])

            return {
                'seq_len': tensorize(batch_seq_len),
                'input_ids': tensorize(batch_input_ids),
                'cls_tgt': tensorize(batch_cls_tgt),
                'bert_token_type_ids': tensorize(batch_bert_token_type_ids),
                'bert_attention_mask': tensorize(batch_bert_attention_mask),
                'batch_ner_exm': batch_ner_exm,
            }

        def multimrc_batcher(batch_e):
            batch_input_ids = []
            batch_token_type_id = []
            batch_query_mask = []
            batch_attn_mask = []
            batch_answer = []
            batch_ner_exm = []
            for e in batch_e:
                batch_input_ids.append(e['input_ids'])
                batch_token_type_id.append(e['token_type_id'])
                batch_attn_mask.append(e['attn_mask'])
                batch_query_mask.append(e['query_mask'])
                batch_ner_exm.append(e['ner_exm'])
                if train:
                    batch_answer.extend(e['answers'])  # concat not stack
            batch_input_ids = tensorize(batch_input_ids)
            batch_token_type_id = tensorize(batch_token_type_id)
            batch_query_mask = tensorize(batch_query_mask)
            batch_attn_mask = tensorize(batch_attn_mask)
            batch_answer = tensorize(batch_answer, 'float')
            return {
                'input_ids': batch_input_ids,
                'token_type_id': batch_token_type_id,
                'attn_mask': batch_attn_mask,
                'query_mask': batch_query_mask,
                'answers': batch_answer,
                'batch_ner_exm': batch_ner_exm,
            }

        return {'span': span_batcher,
                'seq': seq_batcher,
                'cls': cls_batcher,
                'multimrc': multimrc_batcher,
                }.get(mode, None)

    def build_dataset(self, data_source, mode='span'):
        """构造数据集"""
        if isinstance(data_source, (str, Path)):
            instances = NerExample.load_from_jsonl(data_source)
        else:
            instances = data_source

        def yb_ent_convert(ent_type):
            if ent_type.startswith('指标-'):
                ent_type = '指标'
            return ent_type

        for exm in instances:
            exm.truncate(max_size=self.max_len - 2, direction='tail')

            # exm.ent_type_convert(yb_ent_convert)  # yb

        instances = instances

        post_process = {
            # 'span': self.span_post_process_sigmoid,
            'span': self.span_post_process_softmax,
            # 'span': self.span_post_process_softmax_ENG,
            'seq': self.seq_post_process,
            'cls': self.cls_post_process,
            'multimrc': self.multimrc_post_process,
        }.get(mode, None)

        return LazyDataset(instances, post_process)


class Dataset(torch.utils.data.Dataset):
    """Dataset"""

    def __init__(self, instances):
        self.instances = instances

    def __getitem__(self, idx):
        """Get the instance with index idx"""
        return self.instances[idx]

    def __len__(self):
        return len(self.instances)


# class LazyDataset(DataLoaderX):
class LazyDataset(torch.utils.data.Dataset):
    """LazyDataset"""

    def __init__(self, instances, wrapper):
        self.instances = instances
        self.wrapper = wrapper

    def __getitem__(self, idx):
        """Get the instance with index idx"""
        return self.wrapper(self.instances[idx])  # 在DataLoader的时候才对输入进行处理(wrapper) 所以叫Lazy

    def __len__(self):
        return len(self.instances)


def main1():
    demo_data = [
        {
            "queries": [
                {'query': "发布产品", 'answers': [{'text': '吉利', "start_index": 0}]},
                {'query': "时间", 'answers': []},
                {'query': "发布方", 'answers': [{'text': "轿跑SUV", 'start_index': 8}]}
            ],
            "context": "吉利要火了！首款轿跑SUV即将上市，颜值媲美宝马X6，仅13.58万起",
            "event_type": "产品行为-发布"
        }
    ]

    tokenizer_path = '/home/zyn/huggingface_model_resource/chinese-bert-wwm-ext/'
    reader = DataReader(tokenizer_path, 256)
    dataset = reader.build_dataset(demo_data)
    expample = dataset[0]

    # 根据指针去索引答案
    sent_id = expample.sent_id
    text = demo_data[0]['context']
    sent_a_length = len(sent_id) - sent_id.sum()
    point = expample.answers
    answer = []
    for x in point:
        tmp = []
        l = -1
        r = tail = sent_a_length
        while tail < len(sent_id):

            if x[tail][0] > 0:
                l = tail

            if x[tail][1] > 0:
                r = tail
                if l != -1:
                    tmp.append({'text': text[l - sent_a_length: r - sent_a_length + 1]})
                    l = -1
            tail += 1
        answer.append(tmp)
    print(answer)


if __name__ == "__main__":
    tokenizer_path = '/home/zyn/huggingface_model_resource/chinese-bert-wwm-ext/'
    tokenizer_path = r'E:\huggingface_bert_model\bert-base-uncased'
    tokenizer_path = r'E:\huggingface_bert_model\bert-base-chinese'
    datareader = NerDataReader(tokenizer_path, 512, ent_file='data/yanbao/sent/ent_lst_addO.txt')

    # exm_lst = NerExample.load_from_jsonl('data/yanbao/exm/indi_nerexm_lst_maxlen512.jsonl')
    exm_lst = NerExample.load_from_jsonl('data/yanbao/sent/doc_exm_w_re_sent.jsonl', external_attrs=['itp2atp_dct', 'source_id'])

    dataset = datareader.build_dataset(
        exm_lst,
        mode='multimrc'
    )
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=16,
                                              shuffle=False,
                                              collate_fn=datareader.get_batcher_fn(gpu=False, mode='multimrc'))

    for e in data_loader:
        print(e)
        break

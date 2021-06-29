#!/usr/bin/env python
# coding=utf-8
"""
author: yunanzhang
model: span-level ner model
"""
import torch
import torch.nn as nn
from typing import List, Optional
from transformers import BertConfig, BertModel
import math
import numpy as np


def get_my_matrix(batch_size, length):
    matrix_lst = []
    for i in range(length - 1):
        mat = np.zeros([length, length], dtype='float')
        mat[:i + 1, i + 1:] = 1.
        matrix_lst.append(mat)
    matrix_lst = np.stack(matrix_lst, 0)
    matrix_lst = np.tile(matrix_lst[None, ...], (batch_size, 1, 1, 1))
    return matrix_lst


class Tencent_layer(nn.Module):
    def __init__(self,
                 input_hidden_size,
                 score_layer_size,
                 ent_size,
                 dropout_rate):
        super(Tencent_layer, self).__init__()
        self.mlp_layer1 = nn.Linear(input_hidden_size * 4, score_layer_size)
        self.mlp_layer2 = nn.Linear(score_layer_size, ent_size)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, input_hidden1, input_hidden2):
        m, i = input_hidden1.shape[1:]  # t1_len t1_hid (call t1)
        n, j = input_hidden2.shape[1:]  # t2_len t2_hid (call t2)

        # Concat
        # b,m,i concat b,n,j (need m = n)
        # b,m,1,i -> b,m,m,i
        # b,1,n,j -> b,n,n,j
        # b,m,m,i concat b,n,n,j

        # 不用repeat 用expand节省内存
        # t1_expand = input_hidden1.unsqueeze(2).repeat((1, 1, m, 1))
        # t2_expand = input_hidden2.unsqueeze(1).repeat((1, n, 1, 1))

        t1_expand = input_hidden1.unsqueeze(2).expand((-1, -1, m, -1))
        t2_expand = input_hidden2.unsqueeze(1).expand((-1, n, -1, -1))

        t1multit2 = t1_expand * t2_expand  # bmn(iorj)
        t1minust2 = t1_expand - t2_expand  # bmn(iorj)

        t1concatt2 = torch.cat([t1_expand, t2_expand, t1multit2, t1minust2], -1)  # bmn 4*iorj

        # MLP
        linear_out = self.mlp_layer1(t1concatt2)
        linear_out = torch.tanh(linear_out)
        span_ner_mat_tensor = self.mlp_layer2(linear_out)  # bmnt

        return span_ner_mat_tensor


class Biaffine_layer(nn.Module):
    def __init__(self,
                 input_hidden_size,
                 start_size,
                 end_size,
                 ent_size,
                 dropout_rate,
                 add_bias=True):
        super(Biaffine_layer, self).__init__()
        self.add_bias = add_bias
        self.start_size = 150
        self.end_size = 150
        self.hidden2start_layer = nn.Linear(400, self.start_size)
        self.hidden2end_layer = nn.Linear(400, self.end_size)
        self.dropout_layer = nn.Dropout(p=0.2)
        self.dropout_2d_layer = nn.Dropout2d(p=0.2)

        self.w = nn.Parameter(torch.Tensor(self.start_size + int(self.add_bias), ent_size, self.end_size + int(self.add_bias)))
        # self.w.data.uniform_(-0.1, 0.1)
        self.w.data.zero_()
        # self.hidden2ent_layer = nn.Linear(start_size + int(self.add_bias) + end_size + int(self.add_bias) , ent_size)

        self.bilstm_layer = nn.LSTM(
            input_size=input_hidden_size,
            hidden_size=200,
            num_layers=3,
            bidirectional=True,
            batch_first=True,
            dropout=0.4
        )

    def forward(self, input_hidden, seq_len):
        # BiLSTM
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(input_hidden, list(seq_len), batch_first=True, enforce_sorted=False)
        pack_out, _ = self.bilstm_layer(pack_embed)
        bilstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
        # bilstm_out = self.dropout_layer(bilstm_out)

        start = self.hidden2start_layer(bilstm_out)  # [bat,len,hid]
        end = self.hidden2end_layer(bilstm_out)  # [bat,len,hid]

        # b,l,h -> b,l,h,1 -> dropout2d(impact l) -> b,l,h
        start = self.dropout_2d_layer(start[..., None]).squeeze(-1)
        end = self.dropout_2d_layer(end[..., None]).squeeze(-1)

        if self.add_bias:
            start = torch.cat([start, torch.ones_like(start[..., :1])], dim=-1)
            end = torch.cat([end, torch.ones_like(end[..., :1])], dim=-1)

        m, i = start.shape[1:]  # start_len start_hid (call t1)
        n, j = end.shape[1:]  # end_len end_hid (call t2)

        # bilinear
        # w_shape: i,tag,j
        # t1_shape: b,m,i
        # t2_shape: b,n,j
        # t1*w*t2 -> b,m,n,t
        o1 = torch.einsum('itj, bnj -> bitn', self.w, end)  # w *t2
        o2 = torch.einsum('bmi, bitn -> bmtn', start, o1)  # t1 * (w * t2)
        o2 = torch.transpose(o2, 2, 3)  # bmnt

        # o2 = torch.einsum('bmi, itj, bnj -> bmnt', start, self.w, end)

        # linear
        # b,m,i concat b,n,j (need m = n)
        # b,m,1,i -> b,m,m,i
        # b,1,n,j -> b,n,n,j
        # b,m,m,i concat b,n,n,j

        # 不用repeat 用expand节省内存
        # t1_expand = start.unsqueeze(2).repeat((1, 1, m, 1))
        # t2_expand = end.unsqueeze(1).repeat((1, n, 1, 1))

        # t1_expand = start.unsqueeze(2).expand((-1, -1, m, -1))
        # t2_expand = end.unsqueeze(1).expand((-1, n, -1, -1))

        # t1concatt2 = torch.cat([t1_expand, t2_expand], -1)  # bmn(i+j)

        # linear_out = self.hidden2ent_layer(t1concatt2)  # bmnt

        # span_ner_mat_tensor = o2 + linear_out  # bmnt
        span_ner_mat_tensor = o2
        return span_ner_mat_tensor


class Conj_layer(nn.Module):
    def __init__(self, input_hidden_size, conj_size, ent_size):
        super(Conj_layer, self).__init__()

        self.num_ent = ent_size
        self.conj_size = conj_size

        self.conj_start_layer = nn.Linear(input_hidden_size, self.conj_size)
        self.conj_end_layer = nn.Linear(input_hidden_size, self.conj_size)

    def forward(self, input_hidden):
        conj_start_hidden = self.conj_start_layer(input_hidden)  # b,l,h
        conj_end_hidden = self.conj_end_layer(input_hidden)  # b,l,h

        conj_dot_product_score = torch.matmul(conj_start_hidden, conj_end_hidden.transpose(-1, -2))  # b,l,l

        conj_dot_product_score = conj_dot_product_score / math.sqrt(self.conj_size)  # b,l,l

        conj_dot_product_score = torch.diagonal(conj_dot_product_score, offset=1, dim1=-2, dim2=-1)  # b,l-1

        # softmax
        mask_matrix = get_mask_for_scale(conj_dot_product_score, mode='min')[..., None]  # b,l-1,l-1,1
        final_mask = torch.nn.functional.pad(mask_matrix, pad=(0, 0, 1, 0, 0, 1), mode="constant", value=0)  # b,l,l,1
        final_mask = torch.nn.functional.pad(final_mask, pad=(1, self.num_ent - 2), mode="constant", value=0)  # b,l,l,l  # 只加在O上

        return final_mask


class SelfAttention_with_mask(nn.Module):
    def __init__(self, input_hidden_size, ent_size):
        super(SelfAttention_with_mask, self).__init__()

        self.num_ent = ent_size
        self.hidden_size_per_ent = 50

        total_hidden_size = (self.num_ent + 1) * self.hidden_size_per_ent

        self.start_layer = nn.Linear(input_hidden_size, total_hidden_size)
        self.end_layer = nn.Linear(input_hidden_size, total_hidden_size)

    def forward(self, input_hidden, mask_matrix):
        # input_hidden b,l,h
        sinusoidal_pos = self.get_sinusoidal_position_embeddings(input_hidden, output_dim=self.hidden_size_per_ent)  # [l,h]

        start_hidden = self.start_layer(input_hidden)
        end_hidden = self.end_layer(input_hidden)

        start_hidden = self.transpose_for_scores(start_hidden)
        end_hidden = self.transpose_for_scores(end_hidden)

        conj_start_hidden = start_hidden[:, -1, :, :]  # b,l,h
        conj_end_hidden = end_hidden[:, -1, :, :]  # b,l,h

        start_hidden = start_hidden[:, :-1, :, :]
        end_hidden = end_hidden[:, :-1, :, :]

        # Rotary pos
        query_layer = start_hidden
        key_layer = end_hidden

        cos_pos = torch.repeat_interleave(sinusoidal_pos[..., 1::2], 2, dim=-1)

        sin_pos = torch.repeat_interleave(sinusoidal_pos[..., ::2], 2, dim=-1)
        # query_layer b h l d
        qw2 = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(query_layer)

        query_layer = query_layer * cos_pos + qw2 * sin_pos

        kw2 = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        key_layer = key_layer * cos_pos + kw2 * sin_pos

        start_hidden = query_layer
        end_hidden = key_layer
        # end Rotary pos

        attention_scores = torch.matmul(start_hidden, end_hidden.transpose(-1, -2))  # [bat,num_ent,len,hid] * [bat,num_ent,hid,len] = [bat,num_ent,len,len]

        attention_scores = attention_scores / math.sqrt(self.hidden_size_per_ent)  # b,e,l,l

        span_ner_mat_tensor = attention_scores.permute(0, 2, 3, 1)  # b,l,l,e

        # conj
        batch_size, length = input_hidden.shape[:2]
        conj_dot_product_score = torch.matmul(conj_start_hidden, conj_end_hidden.transpose(-1, -2))  # b,l,l
        conj_dot_product_score /= math.sqrt(self.hidden_size_per_ent)  # b,l,l
        tmp_lst = torch.split(conj_dot_product_score, 1, dim=0)  # List:b 1,l,l
        tmp_lst = [torch.diag(t.squeeze(0), diagonal=1) for t in tmp_lst]  # 取对角上1行的
        conj_dot_product_score = torch.stack(tmp_lst, 0)  # b,l-1
        # print('conj_dot_product_score', conj_dot_product_score.shape)
        conj_dot_product_score = conj_dot_product_score.unsqueeze(-1).unsqueeze(-1)  # b,l-1,1,1
        # mask_matrix b,l-1,l,l
        # mask_matrix = torch.FloatTensor(get_my_matrix(batch_size, length)).cuda()
        mask_matrix = mask_matrix.unsqueeze(0)  # 1, l-1, l,l
        final_mask = mask_matrix * conj_dot_product_score  # b,l-1,l,l
        final_mask = torch.sum(final_mask, dim=1)  # b,l,l
        final_mask = final_mask.unsqueeze(-1)  # b,l,l,1
        # print('final_mask', final_mask.shape)
        span_ner_mat_tensor = span_ner_mat_tensor + final_mask

        return span_ner_mat_tensor

    def transpose_for_scores(self, x):
        # x: [bat,len,totalhid]
        new_x_shape = x.size()[:-1] + (self.num_ent + 1, self.hidden_size_per_ent)  # [bat,len,num_ent,hid]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [bat,num_ent,len,hid]

    def get_sinusoidal_position_embeddings(self, inputs, output_dim):
        # 要获取句子长度length 和最细粒度的hidden size
        # output_dim = self.config.hidden_size // self.config.num_attention_heads
        # output_dim = inputs.size(2)
        seq_len = inputs.size(1)

        position_ids = torch.arange(
            0, seq_len, dtype=torch.float32, device=inputs.device)

        indices = torch.arange(
            0, output_dim // 2, dtype=torch.float32, device=inputs.device)
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = torch.einsum('n,d->nd', position_ids, indices)
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], dim=-1)
        embeddings = torch.reshape(embeddings, (seq_len, output_dim))
        return embeddings[None, None, :, :]


class SelfAttention_with_mask_mean(nn.Module):
    def __init__(self, input_hidden_size, ent_size):
        super(SelfAttention_with_mask_mean, self).__init__()

        self.num_ent = ent_size
        self.hidden_size_per_ent = 50
        self.dropout_layer = nn.Dropout(p=0.2)
        total_hidden_size = (self.num_ent + 1) * self.hidden_size_per_ent
        # total_hidden_size = (self.num_ent * 2 + 1) * self.hidden_size_per_ent

        self.start_layer = nn.Linear(input_hidden_size, total_hidden_size)
        self.end_layer = nn.Linear(input_hidden_size, total_hidden_size)

    def forward(self, input_hidden, loss_type='softmax', userope=False):
        # input_hidden b,l,h
        batch_size, length = input_hidden.shape[:2]
        # start_hidden = torch.nn.functional.relu(self.start_layer(input_hidden))
        # end_hidden = torch.nn.functional.relu(self.end_layer(input_hidden))

        start_hidden = self.start_layer(input_hidden)
        end_hidden = self.end_layer(input_hidden)

        # start_hidden = self.dropout_layer(start_hidden)
        # end_hidden = self.dropout_layer(end_hidden)

        start_hidden = self.transpose_for_scores(start_hidden)  # b,e,l,h
        end_hidden = self.transpose_for_scores(end_hidden)  # b,e,l,h

        conj_start_hidden = start_hidden[:, -1, :, :]  # b,l,h
        conj_end_hidden = end_hidden[:, -1, :, :]  # b,l,h

        start_hidden = start_hidden[:, :-1, :, :]  # b,e,l,h
        end_hidden = end_hidden[:, :-1, :, :]  # b,e,l,h

        if userope:
            sinusoidal_pos = self.get_sinusoidal_position_embeddings(input_hidden, output_dim=self.hidden_size_per_ent)  # [l,h]
            # Rotary pos
            query_layer = start_hidden
            key_layer = end_hidden

            cos_pos = torch.repeat_interleave(sinusoidal_pos[..., 1::2], 2, dim=-1)

            sin_pos = torch.repeat_interleave(sinusoidal_pos[..., ::2], 2, dim=-1)
            # query_layer b h l d
            qw2 = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(query_layer)

            query_layer = query_layer * cos_pos + qw2 * sin_pos

            kw2 = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
            key_layer = key_layer * cos_pos + kw2 * sin_pos

            start_hidden = query_layer
            end_hidden = key_layer
            # end Rotary pos

        attention_scores = torch.matmul(start_hidden, end_hidden.transpose(-1, -2))  # [bat,num_ent,len,hid] * [bat,num_ent,hid,len] = [bat,num_ent,len,len]
        attention_scores = attention_scores / math.sqrt(self.hidden_size_per_ent)  # b,e,l,l
        span_ner_mat_tensor = attention_scores.permute(0, 2, 3, 1)  # b,l,l,e

        # # 翻转
        # low_tri_mask = torch.tril(torch.ones([length, length]).to(input_hidden.device), diagonal=0)[None, ..., None]  # 1,l,l,1
        # span_ner_mat_tensor_low_tri = span_ner_mat_tensor * low_tri_mask
        # span_ner_mat_tensor_low_tri = span_ner_mat_tensor_low_tri.transpose(-2, -3)  # 翻转下三角到上三角
        # span_ner_mat_tensor = span_ner_mat_tensor + span_ner_mat_tensor_low_tri

        # # 2transpose
        # forward_start_hidden, backward_start_hidden = torch.chunk(start_hidden, 2, dim=1)
        # forward_end_hidden, backward_end_hidden = torch.chunk(end_hidden, 2, dim=1)
        # forward_span_mat_tensor = (torch.matmul(forward_start_hidden, forward_end_hidden.transpose(-1, -2)) / self.hidden_size_per_ent ** 0.5).permute(0, 2, 3, 1)  # b,l,l,e
        # backward_span_mat_tensor = (torch.matmul(backward_start_hidden, backward_end_hidden.transpose(-1, -2)) / self.hidden_size_per_ent ** 0.5).permute(0, 2, 3, 1)  # b,l,l,e
        #
        # # 翻转
        # low_tri_mask = torch.tril(torch.ones([length, length]).to(input_hidden.device), diagonal=0)[None, ..., None]  # 1,l,l,1
        # span_ner_mat_tensor_low_tri = backward_span_mat_tensor * low_tri_mask
        # span_ner_mat_tensor_low_tri = span_ner_mat_tensor_low_tri.transpose(-2, -3)  # 翻转下三角到上三角
        # span_ner_mat_tensor = forward_span_mat_tensor + span_ner_mat_tensor_low_tri

        # conj
        conj_dot_product_score = torch.matmul(conj_start_hidden, conj_end_hidden.transpose(-1, -2))  # b,l,l
        conj_dot_product_score /= math.sqrt(self.hidden_size_per_ent)  # b,l,l
        conj_dot_product_score = torch.diagonal(conj_dot_product_score, offset=1, dim1=-2, dim2=-1)  # b,l-1

        # print('conj_dot_product_score', conj_dot_product_score)
        # print('conj_dot_product_score', torch.sigmoid(conj_dot_product_score))

        if loss_type == 'softmax':
            mask_matrix = get_mask_for_scale(conj_dot_product_score, mode='min')[..., None]  # b,l-1,l-1,1
            final_mask = torch.nn.functional.pad(mask_matrix, pad=(0, 0, 1, 0, 0, 1), mode="constant", value=0)  # b,l,l,1
            final_mask = torch.nn.functional.pad(final_mask, pad=(1, self.num_ent - 2), mode="constant", value=0)  # b,l,l,l  # 只加在O上

            span_ner_mat_tensor = span_ner_mat_tensor - final_mask
            # span_ner_mat_tensor = span_ner_mat_tensor

        if loss_type == 'sigmoid':
            # mask_matrix = get_windows_sum(conj_dot_product_score[..., None], mean=False)  # b,l-1,l-1,1
            # mask_matrix = get_windows_sum_min_pool1(conj_dot_product_score[..., None])  # b,l-1,l-1,1
            # mask_matrix = - get_windows_sum_max_pool(conj_dot_product_score[..., None])  # b,l-1,l-1,1
            # mask_matrix = get_windows_sum_max_pool(conj_dot_product_score[..., None])  # b,l-1,l-1,1

            # mask_matrix = get_mask_for_scale_sum(conj_dot_product_score, mean=True)[..., None]  # b,l-1,l-1,1
            mask_matrix = get_mask_for_scale(conj_dot_product_score, mode='min')[..., None]  # b,l-1,l-1,1
            final_mask = torch.nn.functional.pad(mask_matrix, pad=(0, 0, 1, 0, 0, 1), mode="constant", value=0)  # b,l,l,1

            span_ner_mat_tensor = span_ner_mat_tensor + final_mask
            # span_ner_mat_tensor = span_ner_mat_tensor

        # Note
        # triu的作用最多是3维b,l,l 且是直接取树不是生成0-1矩阵 吃亏！！
        # final_mask = mask_matrix * torch.triu(mask_matrix, diagonal=1)  # 对角线以上的为1 保留  b,l,l,1

        return span_ner_mat_tensor, torch.sigmoid(conj_dot_product_score)

        # return span_ner_mat_tensor, conj_dot_product_score

    def transpose_for_scores(self, x):
        # x: [bat,len,totalhid]
        new_x_shape = x.size()[:-1] + (self.num_ent + 1, self.hidden_size_per_ent)  # [bat,len,num_ent,hid]
        # new_x_shape = x.size()[:-1] + (self.num_ent*2 + 1, self.hidden_size_per_ent)  # [bat,len,num_ent,hid]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [bat,num_ent,len,hid]

    def get_sinusoidal_position_embeddings(self, inputs, output_dim):
        # 要获取句子长度length 和最细粒度的hidden size
        # output_dim = self.config.hidden_size // self.config.num_attention_heads
        # output_dim = inputs.size(2)
        seq_len = inputs.size(1)

        position_ids = torch.arange(
            0, seq_len, dtype=torch.float32, device=inputs.device)

        indices = torch.arange(
            0, output_dim // 2, dtype=torch.float32, device=inputs.device)
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = torch.einsum('n,d->nd', position_ids, indices)
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], dim=-1)
        embeddings = torch.reshape(embeddings, (seq_len, output_dim))
        return embeddings[None, None, :, :]


class SelfAttention(nn.Module):
    def __init__(self, input_hidden_size, ent_size):
        super(SelfAttention, self).__init__()

        self.num_ent = ent_size
        self.hidden_size_per_ent = 50

        total_hidden_size = self.num_ent * self.hidden_size_per_ent

        self.start_layer = nn.Linear(input_hidden_size, total_hidden_size)
        self.end_layer = nn.Linear(input_hidden_size, total_hidden_size)

    def forward(self, input_hidden):
        # input_hidden b,l,h
        # sinusoidal_pos = self.get_sinusoidal_position_embeddings(input_hidden, output_dim=self.hidden_size_per_ent)  # [l,h]

        start_hidden = self.start_layer(input_hidden)
        end_hidden = self.end_layer(input_hidden)

        start_hidden = self.transpose_for_scores(start_hidden)
        end_hidden = self.transpose_for_scores(end_hidden)

        # Rotary pos
        # query_layer = start_hidden
        # key_layer = end_hidden
        #
        # cos_pos = torch.repeat_interleave(sinusoidal_pos[..., 1::2], 2, dim=-1)
        #
        # sin_pos = torch.repeat_interleave(sinusoidal_pos[..., ::2], 2, dim=-1)
        # # query_layer b h l d
        # qw2 = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(query_layer)
        #
        # query_layer = query_layer * cos_pos + qw2 * sin_pos
        #
        # kw2 = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
        # key_layer = key_layer * cos_pos + kw2 * sin_pos
        #
        # start_hidden = query_layer
        # end_hidden = key_layer
        # end Rotary pos

        attention_scores = torch.matmul(start_hidden, end_hidden.transpose(-1, -2))  # [bat,num_ent,len,hid] * [bat,num_ent,hid,len] = [bat,num_ent,len,len]

        attention_scores = attention_scores / math.sqrt(self.hidden_size_per_ent)

        span_ner_mat_tensor = attention_scores.permute(0, 2, 3, 1)

        return span_ner_mat_tensor

    def transpose_for_scores(self, x):
        # x: [bat,len,totalhid]
        new_x_shape = x.size()[:-1] + (self.num_ent, self.hidden_size_per_ent)  # [bat,len,num_ent,hid]
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)  # [bat,num_ent,len,hid]

    def get_sinusoidal_position_embeddings(self, inputs, output_dim):
        # output_dim = self.config.hidden_size // self.config.num_attention_heads
        # output_dim = inputs.size(2)
        seq_len = inputs.size(1)

        position_ids = torch.arange(
            0, seq_len, dtype=torch.float32, device=inputs.device)

        indices = torch.arange(
            0, output_dim // 2, dtype=torch.float32, device=inputs.device)
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        embeddings = torch.einsum('n,d->nd', position_ids, indices)
        embeddings = torch.stack([embeddings.sin(), embeddings.cos()], dim=-1)
        embeddings = torch.reshape(embeddings, (seq_len, output_dim))
        return embeddings[None, None, :, :]


class Bert_Span(nn.Module):
    def __init__(self,
                 conf,
                 tok2id,
                 pretrain_embed=None):
        super(Bert_Span, self).__init__()
        self.char2id = tok2id['char2id']
        self.id2char = tok2id['id2char']
        self.tag2id = tok2id['tag2id']
        self.id2tag = tok2id['id2tag']
        self.ent2id = tok2id['ent2id']
        self.id2ent = tok2id['id2ent']

        bert_model_name = conf['bert_model_dir']
        self.bert_layer = BertModel.from_pretrained(bert_model_name)
        self.bert_conf = BertConfig.from_pretrained(bert_model_name)

        self.dropout_rate = conf['dropout_rate']
        self.negsample_rate = conf['span_model_negsample_rate']
        print('span_model_negsample_rate', self.negsample_rate)
        self.vocab_size = len(self.char2id)
        # self.tag_size = len(self.tag2id)
        self.ent_size = len(self.ent2id)
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

        self.span_layer_type = conf['span_layer_type']
        if self.span_layer_type == 'biaffine':
            self.biaffine_layer = Biaffine_layer(input_hidden_size=self.bert_conf.hidden_size,
                                                 start_size=self.bert_conf.hidden_size,
                                                 end_size=self.bert_conf.hidden_size,
                                                 ent_size=self.ent_size,
                                                 dropout_rate=self.dropout_rate)
            self.biaffine_conj_layer = Conj_layer(input_hidden_size=self.bert_conf.hidden_size,
                                                  conj_size=50,
                                                  ent_size=self.ent_size)

        elif self.span_layer_type == 'tencent':
            self.tencent_layer = Tencent_layer(input_hidden_size=self.bert_conf.hidden_size,
                                               score_layer_size=256,
                                               ent_size=self.ent_size,
                                               dropout_rate=self.dropout_rate)
        elif self.span_layer_type == 'self_attn':
            self.self_attn_layer = SelfAttention(input_hidden_size=self.bert_conf.hidden_size,
                                                 ent_size=self.ent_size)
        # elif self.span_layer_type == 'self_attn_mask':
        #     self.self_attn_layer = SelfAttention_with_mask(input_hidden_size=self.bert_conf.hidden_size,
        #                                                    ent_size=self.ent_size)
        elif self.span_layer_type == 'self_attn_mask_mean':
            self.self_attn_layer = SelfAttention_with_mask_mean(input_hidden_size=self.bert_conf.hidden_size,
                                                                ent_size=self.ent_size)

        else:

            raise NotImplementedError()

        self.ce_loss_layer = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        self.bce_loss_layer = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, bert_chars_ids, bert_token_type_ids, bert_attention_mask, seq_len, tag=0,*span_ner_tgt_lst):
        # seq_len [bat]
        # span_ner_tgt_lst [len*(len+1)/2]
        batch_size, length = bert_chars_ids.shape[:2]
        seq_len_lst = seq_len.tolist()

        # with torch.no_grad():
        bert_outputs = self.bert_layer(input_ids=bert_chars_ids,
                                       token_type_ids=bert_token_type_ids,
                                       attention_mask=bert_attention_mask)

        # bert_outputs = process_long_input(self.bert_layer,
        #                               input_ids=bert_chars_ids,
        #                               token_type_ids=bert_token_type_ids,
        #                               attention_mask=bert_attention_mask,
        #                               seq_len_lst=seq_len_lst,
        #                               bert_max_len=256,
        #                               )

        bert_out = bert_outputs.last_hidden_state

        # 去除bert_output[CLS]和[SEP]
        bert_out_lst = [t for t in bert_out]
        for i, t in enumerate(bert_out_lst):  # iter through batch
            # tensor [len, hid]
            bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
        bert_out = torch.stack(bert_out_lst, 0)
        bert_out = self.dropout_layer(bert_out)

        conj_dot_product_score = None
        if self.span_layer_type == 'biaffine':
            span_ner_mat_tensor = self.biaffine_layer(bert_out, seq_len)  # [bat,len,len,tag]spa
            finnal_mask = self.biaffine_conj_layer(bert_out)
            span_ner_mat_tensor = span_ner_mat_tensor - finnal_mask
        elif self.span_layer_type == 'tencent':
            span_ner_mat_tensor = self.tencent_layer(bert_out, bert_out)  # [bat,len,len,tag]
        elif self.span_layer_type == 'self_attn':
            span_ner_mat_tensor = self.self_attn_layer(bert_out)  # [bat,len,len,tag]
        # elif self.span_layer_type == 'self_attn_mask':
        #     span_ner_mat_tensor = self.self_attn_layer(bert_out, mask_matrix)  # [bat,len,len,tag]
        elif self.span_layer_type == 'self_attn_mask_mean':
            span_ner_mat_tensor, conj_dot_product_score = self.self_attn_layer(bert_out, loss_type='softmax', userope=False)  # [bat,len,len,tag]
        else:
            raise NotImplementedError()

        # 构造下三角mask 考虑了pad 考虑了下三角为0
        len_mask = sequence_mask(seq_len)  # b,l
        matrix_mask = torch.logical_and(torch.unsqueeze(len_mask, 1), torch.unsqueeze(len_mask, 2))  # b,l,l  # 小正方形mask
        score_mat_mask = torch.triu(matrix_mask, diagonal=0)  # b,l,l  # 取出对角线及上三角，其余为0
        score_mat_mask = torch.unsqueeze(score_mat_mask, -1).expand(-1, -1, -1, self.ent_size)  # b,l,l,t
        span_ner_pred_lst = torch.masked_select(span_ner_mat_tensor, score_mat_mask)  # 只取True或1组成列表
        span_ner_pred_lst = span_ner_pred_lst.view(-1, len(self.ent2id))  # [*,ent]

        
        
        span_ner_pred_prob_lst = torch.softmax(span_ner_pred_lst, -1)
        if tag==1:
            # print("------------------------")
            # print(type(span_ner_mat_tensor))
            # print(type(span_ner_pred_prob_lst))
            # print(type(conj_dot_product_score))
            # print("------------------------")
            return span_ner_mat_tensor, span_ner_pred_prob_lst, conj_dot_product_score  # 返回的是prob不是logit
        # # sigmoid
        # span_loss = self.bce_loss_layer(span_ner_pred_lst, span_ner_tgt_lst)  # 传入[batch,num_class] - [batch, num_class] target已是onehot
        # # span_loss = torch.sum(span_loss, -1)
        # span_loss = torch.mean(span_loss)
        # span_ner_pred_prob_lst = torch.sigmoid(span_ner_pred_lst)

        # # Su
        # span_loss = multilabel_categorical_crossentropy(span_ner_pred_lst, span_ner_tgt_lst)  # SuJianlin old
        # span_loss = multilabel_categorical_crossentropy(span_ner_pred_lst.transpose(0, 1), span_ner_tgt_lst.transpose(0, 1))  # SuJianlin
        # span_loss = torch.mean(span_loss)
        # span_ner_pred_prob_lst = torch.sigmoid(span_ner_pred_lst)
        else:
        # f1 metrics
            if self.negsample_rate:
                assert 0 < self.negsample_rate <= 1
                r = self.negsample_rate
                # ===负采样(适用softmax)
                # r = 0.35  # neg_sample_ratio 按句子长度比率r保留负样本 r=0.35  seq_len * 0.35
                span_len = seq_len * (seq_len + 1) // 2  # b
                # print(seq_len)
                # print(span_len)
                batch_span_ner_tag_lst = torch.split(span_ner_tgt_lst, span_len.tolist(), dim=0)  # tensor列表 batch中的每一个
                # print(batch_span_ner_tag_lst)
                # 每个正样本数:
                batch_num_pos = [torch.sum(t > 1) for t in batch_span_ner_tag_lst]  # 0和1不是正样本
                # print(batch_num_pos)
                # 转换成在全体负样本中保留概率 2rn / n(n+1)-2p  p为正样本个数
                batch_neg_keep_prob = [2 * r * n / (n * (n + 1) - 2 * p) for n, p in zip(seq_len, batch_num_pos)]
                # print(batch_neg_keep_prob)
                batch_neg_mask = []
                for keep_prob, tgt_lst in zip(batch_neg_keep_prob, batch_span_ner_tag_lst):
                    rnd = torch.rand_like(tgt_lst.float())
                    neg_sample_mask = torch.where(torch.logical_and(tgt_lst == 1, rnd > keep_prob), 0, 1)  # 没有采样出来的负样本为0. 要丢弃的
                    batch_neg_mask.append(neg_sample_mask)
                neg_sample_mask = torch.cat(batch_neg_mask, dim=0)
                span_ner_tgt_lst = span_ner_tgt_lst * neg_sample_mask
                # ===负采样完毕

            span_loss = self.ce_loss_layer(span_ner_pred_lst, span_ner_tgt_lst)  # 传入[batch,num_class] - [batch] target还没变成one-hot
            span_loss = torch.sum(span_loss)
            span_ner_tgt_lst_onehot = torch.nn.functional.one_hot(span_ner_tgt_lst, self.ent_size)  # [*,ent]
            span_ner_perd_lst_onehot = torch.nn.functional.one_hot(torch.argmax(span_ner_pred_lst, dim=-1), self.ent_size)  # [*,ent]
            span_ner_tgt_lst_onehot = span_ner_tgt_lst_onehot[:, 2:]  # 不要pad和O
            span_ner_perd_lst_onehot = span_ner_perd_lst_onehot[:, 2:]  # 不要pad和O
            num_gold = torch.sum(span_ner_tgt_lst_onehot)
            num_pred = torch.sum(span_ner_perd_lst_onehot)
            tp = torch.sum(span_ner_tgt_lst_onehot * span_ner_perd_lst_onehot)
            f1 = 2 * tp / (num_gold + num_pred + 1e-12)

            return span_loss, span_ner_mat_tensor, span_ner_pred_prob_lst, conj_dot_product_score, f1

    def predict(self, bert_chars_ids, bert_token_type_ids, bert_attention_mask, seq_len):
        # seq_len 去除cls和sep的真实长度
        chars_ids = bert_chars_ids
        seq_len_lst = seq_len.tolist()
        bert_outputs = self.bert_layer(input_ids=chars_ids,
                                       token_type_ids=bert_token_type_ids,
                                       attention_mask=bert_attention_mask)
        bert_out = bert_outputs.last_hidden_state

        # 去除bert_output[CLS]和[SEP]
        bert_out_lst = [t for t in bert_out]
        for i, t in enumerate(bert_out_lst):  # iter through batch
            # tensor [len, hid]
            bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
        bert_out = torch.stack(bert_out_lst, 0)
        bert_out = self.dropout_layer(bert_out)

        conj_dot_product_score = None
        if self.span_layer_type == 'biaffine':
            span_ner_mat_tensor = self.biaffine_layer(bert_out, seq_len)  # [bat,len,len,tag]spa
            finnal_mask = self.biaffine_conj_layer(bert_out)
            span_ner_mat_tensor = span_ner_mat_tensor - finnal_mask
        elif self.span_layer_type == 'tencent':
            span_ner_mat_tensor = self.tencent_layer(bert_out, bert_out)  # [bat,len,len,tag]
        elif self.span_layer_type == 'self_attn':
            span_ner_mat_tensor = self.self_attn_layer(bert_out)  # [bat,len,len,tag]
        # elif self.span_layer_type == 'self_attn_mask':
        #     span_ner_mat_tensor = self.self_attn_layer(bert_out, mask_matrix)  # [bat,len,len,tag]
        elif self.span_layer_type == 'self_attn_mask_mean':
            span_ner_mat_tensor, conj_dot_product_score = self.self_attn_layer(bert_out, loss_type='softmax', userope=False)  # [bat,len,len,tag]
        else:
            raise NotImplementedError()

        # 构造下三角mask 考虑了pad 考虑了下三角为0
        len_mask = sequence_mask(seq_len)  # b,l
        matrix_mask = torch.logical_and(torch.unsqueeze(len_mask, 1), torch.unsqueeze(len_mask, 2))  # b,l,l  # 小正方形mask
        score_mat_mask = torch.triu(matrix_mask, diagonal=0)  # b,l,l  # 取出对角线及上三角，其余为0
        score_mat_mask = torch.unsqueeze(score_mat_mask, -1).expand(-1, -1, -1, self.ent_size)  # b,l,l,t
        span_ner_pred_lst = torch.masked_select(span_ner_mat_tensor, score_mat_mask)  # 只取True或1组成列表
        span_ner_pred_lst = span_ner_pred_lst.view(-1, len(self.ent2id))  # [*,ent]

        span_ner_pred_prob_lst = torch.softmax(span_ner_pred_lst, -1)

        return span_ner_mat_tensor, span_ner_pred_prob_lst, conj_dot_product_score  # 返回的是prob不是logit


class Bert_Cls(nn.Module):
    def __init__(self,
                 conf,
                 tok2id,
                 pretrain_embed=None):
        super(Bert_Cls, self).__init__()
        self.char2id = tok2id['char2id']
        self.id2char = tok2id['id2char']
        self.tag2id = tok2id['tag2id']
        self.id2tag = tok2id['id2tag']
        self.ent2id = tok2id['ent2id']
        self.id2ent = tok2id['id2ent']

        bert_model_name = conf['bert_model_dir']
        self.bert_layer = BertModel.from_pretrained(bert_model_name)
        self.bert_conf = BertConfig.from_pretrained(bert_model_name)

        self.dropout_rate = conf['dropout_rate']
        self.vocab_size = len(self.char2id)
        # self.tag_size = len(self.tag2id)
        self.ent_size = len(self.ent2id)
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

        self.linear_layer = nn.Linear(self.bert_conf.hidden_size, 1)  #

        self.ce_loss_layer = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
        self.bce_loss_layer = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.FloatTensor([10]))
        # self.focal_loss_layer = BCEFocalLoss()
        self.focal_loss_layer = FocalLoss()

    def forward(self, bert_chars_ids, bert_token_type_ids, bert_attention_mask, seq_len,tag=0, *cls_tgt):
        # seq_len [bat]
        # cls_tgt [bat]
        batch_size = bert_chars_ids.shape[0]
        # with torch.no_grad():
        bert_outputs = self.bert_layer(input_ids=bert_chars_ids,
                                       token_type_ids=bert_token_type_ids,
                                       attention_mask=bert_attention_mask)

        # bert_outputs = process_long_input(self.bert_layer,
        #                               input_ids=bert_chars_ids,
        #                               token_type_ids=bert_token_type_ids,
        #                               attention_mask=bert_attention_mask,
        #                               seq_len_lst=seq_len_lst,
        #                               bert_max_len=256,
        #                               )

        bert_out = bert_outputs.last_hidden_state

        # 提取出CLS
        cls_out = bert_out[:, 0, :]  # b,h
        cls_out = self.dropout_layer(cls_out)  # b
        cls_logit = self.linear_layer(cls_out)  # b,1

        
        # focal loss
        # cls_loss = self.focal_loss_layer(cls_logit, cls_tgt)
        # cls_loss = torch.mean(cls_loss)

        # f1 metrics
        cls_pred_prob = torch.sigmoid(cls_logit.squeeze(-1))  # sigmoid binary
        if tag:
            return cls_pred_prob  # b
        else:
        # cls_pred_prob = torch.softmax(cls_logit, dim=-1)[:,1]  # softmax binary
            cls_loss = self.bce_loss_layer(cls_logit.squeeze(-1), cls_tgt.float())
            # print(cls_logit.squeeze(-1))
            # print(cls_tgt.float())
            # print(cls_loss)
            cls_loss = torch.sum(cls_loss)
            cls_pred = (cls_pred_prob >= 0.5).int()
            num_gold = torch.sum(cls_tgt)
            num_pred = torch.sum(cls_pred)
            tp = torch.sum(cls_tgt * cls_pred)
            f1 = 2 * tp / (num_gold + num_pred + 1e-12)
            acc = torch.sum(torch.eq(cls_pred, cls_tgt)) / batch_size

            return cls_loss, cls_pred_prob, f1, acc

    def predict(self, bert_chars_ids, bert_token_type_ids, bert_attention_mask, seq_len):
        # seq_len [bat]
        # cls_tgt [bat]
        batch_size = bert_chars_ids.shape[0]
        # with torch.no_grad():
        bert_outputs = self.bert_layer(input_ids=bert_chars_ids,
                                       token_type_ids=bert_token_type_ids,
                                       attention_mask=bert_attention_mask)

        # bert_outputs = process_long_input(self.bert_layer,
        #                               input_ids=bert_chars_ids,
        #                               token_type_ids=bert_token_type_ids,
        #                               attention_mask=bert_attention_mask,
        #                               seq_len_lst=seq_len_lst,
        #                               bert_max_len=256,
        #                               )

        bert_out = bert_outputs.last_hidden_state

        # 提取出CLS
        cls_out = bert_out[:, 0, :]  # b,h
        cls_out = self.dropout_layer(cls_out)  # b
        cls_logit = self.linear_layer(cls_out)  # b,1

        cls_pred_prob = torch.sigmoid(cls_logit.squeeze(-1))  # sigmoid binary
        return cls_pred_prob  # b


class Bert_MultiMrc(nn.Module):
    def __init__(self,
                 conf,
                 tok2id,
                 pretrain_embed=None):
        super(Bert_MultiMrc, self).__init__()
        self.char2id = tok2id['char2id']
        self.id2char = tok2id['id2char']
        self.tag2id = tok2id['tag2id']
        self.id2tag = tok2id['id2tag']
        self.ent2id = tok2id['ent2id']
        self.id2ent = tok2id['id2ent']

        bert_model_name = conf['bert_model_dir']
        self.bert_layer = BertModel.from_pretrained(bert_model_name)
        self.bert_conf = BertConfig.from_pretrained(bert_model_name)

        self.dropout_rate = conf['dropout_rate']
        self.vocab_size = len(self.char2id)
        # self.tag_size = len(self.tag2id)
        self.ent_size = len(self.ent2id)
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.input_size = self.bert_conf.hidden_size
        self.bias = nn.Linear(self.bert_conf.hidden_size, 1)  #
        self.fc = nn.Linear(self.bert_conf.hidden_size, 2)  # start and end

        self.bce_loss_layer = nn.BCELoss(reduction='sum')

    def forward(self, bert_chars_ids, bert_token_type_ids, bert_attention_mask, query_masks,tag=0, *answers):
        # with torch.no_grad():
        bert_outputs = self.bert_layer(input_ids=bert_chars_ids,
                                       token_type_ids=bert_token_type_ids,
                                       attention_mask=bert_attention_mask)

        # bert_outputs = process_long_input(self.bert_layer,
        #                               input_ids=bert_chars_ids,
        #                               token_type_ids=bert_token_type_ids,
        #                               attention_mask=bert_attention_mask,
        #                               seq_len_lst=seq_len_lst,
        #                               bert_max_len=256,
        #                               )

        bert_out = bert_outputs.last_hidden_state

        encoder_feature = bert_out  # bat,len,hid
        query_indexes = torch.nonzero(query_masks.view(-1), as_tuple=False).squeeze()
        query_feature = encoder_feature.view(-1, self.input_size).index_select(0, query_indexes)  # bat*nq,hid  # 取出每个query对应的cls向量
        query_bias = self.bias(query_feature).unsqueeze(-2)  # bat*nq,1,1
        query_nums = query_masks.sum(-1)  # bat  每个句子有多少个query
        expand_feature = self.sequence_expand(encoder_feature, query_nums)  # batch中每个句子有几个query就复制几份 bat*nq,len,hid
        query_feature = query_feature.unsqueeze(-2)  # bat*nq,1,hid
        compact = query_feature * expand_feature + query_bias  # bat*nq,len,hid
        out = self.fc(compact)  # bat*nq,len,2
        prob = torch.sigmoid(out)  # bat*nq,len,2
        mask = bert_token_type_ids * bert_attention_mask[:, 0]  # bat,len * bat,len  只要context的内容mask
        expand_mask = self.sequence_expand(mask, query_nums)  # bat*nq,len
        prob = prob * expand_mask.unsqueeze(-1)  # bat*nq,len,2 * bat*nq,len,1
        # print(prob.shape)


        context_start_index = bert_token_type_ids.shape[1] - bert_token_type_ids.sum(-1)  # bat
        if tag:
            mrc_loss = torch.tensor(0)
        else:
            mrc_loss = self.bce_loss_layer(prob, answers) / prob.shape[0]  # bat*nq,len,2
        return mrc_loss, prob, context_start_index

    def sequence_expand(self, tensor, seq_list):
        """参考paddle sequence_expand功能实现
           其实就是batch中每个句子，按不同的倍数复制。倍数列表为seq_list
        """
        _, *dims = tensor.shape
        out = []
        for l, batch in zip(seq_list, tensor):
            if not l: continue
            batch_expand = batch.unsqueeze(0).expand(l, *dims)
            out.append(batch_expand)
        return torch.cat(out)

    def predict(self, bert_chars_ids, bert_token_type_ids, bert_attention_mask, query_masks, answers):
        # with torch.no_grad():
        bert_outputs = self.bert_layer(input_ids=bert_chars_ids,
                                       token_type_ids=bert_token_type_ids,
                                       attention_mask=bert_attention_mask)

        # bert_outputs = process_long_input(self.bert_layer,
        #                               input_ids=bert_chars_ids,
        #                               token_type_ids=bert_token_type_ids,
        #                               attention_mask=bert_attention_mask,
        #                               seq_len_lst=seq_len_lst,
        #                               bert_max_len=256,
        #                               )

        bert_out = bert_outputs.last_hidden_state

        encoder_feature = bert_out  # bat,len,hid
        query_indexes = torch.nonzero(query_masks.view(-1), as_tuple=False).squeeze()
        query_feature = encoder_feature.view(-1, self.input_size).index_select(0, query_indexes)  # bat*nq,hid  # 取出每个query对应的cls向量
        query_bias = self.bias(query_feature).unsqueeze(-2)  # bat*nq,1,1
        query_nums = query_masks.sum(-1)  # bat  每个句子有多少个query
        expand_feature = self.sequence_expand(encoder_feature, query_nums)  # batch中每个句子有几个query就复制几份 bat*nq,len,hid
        query_feature = query_feature.unsqueeze(-2)  # bat*nq,1,hid
        compact = query_feature * expand_feature + query_bias  # bat*nq,len,hid
        out = self.fc(compact)  # bat*nq,len,2
        prob = torch.sigmoid(out)  # bat*nq,len,2
        mask = bert_token_type_ids * bert_attention_mask[:, 0]  # bat,len * bat,len  只要context的内容mask
        expand_mask = self.sequence_expand(mask, query_nums)  # bat*nq,len
        prob = prob * expand_mask.unsqueeze(-1)  # bat*nq,len,2 * bat*nq,len,1
        # print(prob.shape)
        # mrc_loss = self.bce_loss_layer(prob, answers) / prob.shape[0]  # bat*nq,len,2
        mrc_loss = 0

        context_start_index = bert_token_type_ids.shape[1] - bert_token_type_ids.sum(-1)  # bat
        return mrc_loss, prob, context_start_index


class BiLSTM_Span(nn.Module):
    def __init__(self,
                 conf,
                 tok2id,
                 span_layer_type='biaffine',
                 pretrain_embed=None, ):
        super(BiLSTM_Span, self).__init__()
        self.char2id = tok2id['char2id']
        self.id2char = tok2id['id2char']
        self.tag2id = tok2id['tag2id']
        self.id2tag = tok2id['id2tag']
        self.ent2id = tok2id['ent2id']
        self.id2ent = tok2id['id2ent']

        self.embed_size = conf['embed_size']
        self.rnn_hidden_size = conf['rnn_hidden_size']
        self.dropout_rate = conf['dropout_rate']
        self.vocab_size = len(self.char2id)
        self.tag_size = len(self.tag2id)
        self.ent_size = len(self.ent2id)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)
        if pretrain_embed is not None:
            self.embedding_layer.weight.data.copy_(pretrain_embed)  # self.embeddings.weight.requires_grad = False
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.bilstm_layer = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=conf['rnn_num_layers'],
            bidirectional=True,
            batch_first=True,
            dropout=0
        )
        self.span_layer_type = span_layer_type
        if self.span_layer_type == 'biaffine':
            self.biaffine_layer = Biaffine_layer(input_hidden_size=self.rnn_hidden_size * 2,
                                                 start_size=self.rnn_hidden_size * 2,
                                                 end_size=self.rnn_hidden_size * 2,
                                                 ent_size=self.ent_size,
                                                 dropout_rate=self.dropout_rate)
        elif self.span_layer_type == 'tencent':
            self.tencent_layer = Tencent_layer(input_hidden_size=self.rnn_hidden_size * 2,
                                               score_layer_size=256,
                                               ent_size=self.ent_size,
                                               dropout_rate=self.dropout_rate)
        else:
            raise NotImplementedError()

        self.ce_loss_layer = nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    def forward(self, chars_ids, tags_ids, seq_len,tag=0, *span_ner_tgt_lst):
        # chars_ids [bat,len]
        # tags_ids [bat,len]
        # seq_len [bat]
        # span_ner_tgt_lst [len*(len+1)/2]
        batch_size, length = chars_ids.shape[:2]
        chars_embed = self.embedding_layer(chars_ids)  # [bat,len,emb]
        chars_embed = self.dropout_layer(chars_embed)

        # BiLSTM
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(chars_embed, list(seq_len), batch_first=True, enforce_sorted=False)
        pack_out, _ = self.bilstm_layer(pack_embed)
        bilstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
        bilstm_out = self.dropout_layer(bilstm_out)

        if self.span_layer_type == 'biaffine':
            span_ner_mat_tensor = self.biaffine_layer(bilstm_out)  # [bat,len,len,tag]
        elif self.span_layer_type == 'tencent':
            span_ner_mat_tensor = self.tencent_layer(bilstm_out, bilstm_out)  # [bat,len,len,tag]
        else:
            raise NotImplementedError()

        # 构造下三角mask 考虑了pad 考虑了下三角为0
        len_mask = sequence_mask(seq_len)  # b,l
        matrix_mask = torch.logical_and(torch.unsqueeze(len_mask, 1), torch.unsqueeze(len_mask, 2))  # b,l,l  # 小正方形mask
        score_mat_mask = torch.triu(matrix_mask, diagonal=0)  # b,l,l  # 对角线mask
        score_mat_mask = torch.unsqueeze(score_mat_mask, -1).repeat(1, 1, 1, self.ent_size)  # b,l,l,t

        span_ner_pred_lst = torch.masked_select(span_ner_mat_tensor, score_mat_mask)  # 只取True或1组成列表
        span_ner_pred_lst = span_ner_pred_lst.view(-1, len(self.ent2id))  # [None,ent]
        if tag:
            return span_ner_mat_tensor, span_ner_pred_lst  # 返回的是prob不是logit
        else:

            span_loss = self.ce_loss_layer(span_ner_pred_lst, span_ner_tgt_lst)  # 传入[batch,num_class] - [batch] target还没变成one-hot
            span_loss = torch.sum(span_loss)

            return span_loss, span_ner_mat_tensor, span_ner_pred_lst

    def predict(self, chars_ids, seq_len):
        chars_embed = self.embedding_layer(chars_ids)  # [bat,len,emb]
        chars_embed = self.dropout_layer(chars_embed)

        # BiLSTM
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(chars_embed, list(seq_len), batch_first=True, enforce_sorted=False)
        pack_out, _ = self.bilstm_layer(pack_embed)
        bilstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
        bilstm_out = self.dropout_layer(bilstm_out)

        if self.span_layer_type == 'biaffine':
            span_ner_mat_tensor = self.biaffine_layer(bilstm_out)  # [bat,len,len,tag]
        elif self.span_layer_type == 'tencent':
            span_ner_mat_tensor = self.tencent_layer(bilstm_out, bilstm_out)  # [bat,len,len,tag]
        else:
            raise NotImplementedError()

        # 构造下三角mask 考虑了pad 考虑了下三角为0
        len_mask = sequence_mask(seq_len)  # b,l
        matrix_mask = torch.logical_and(torch.unsqueeze(len_mask, 1), torch.unsqueeze(len_mask, 2))  # b,l,l  # 小正方形mask
        score_mat_mask = torch.triu(matrix_mask, diagonal=0)  # b,l,l  # 对角线mask
        score_mat_mask = torch.unsqueeze(score_mat_mask, -1).repeat(1, 1, 1, self.ent_size)  # b,l,l,t

        span_ner_pred_lst = torch.masked_select(span_ner_mat_tensor, score_mat_mask)  # 只取True或1组成列表
        span_ner_pred_lst = span_ner_pred_lst.view(-1, len(self.ent2id))  # [None,ent]

        span_ner_mat_tensor = torch.softmax(span_ner_mat_tensor, -1)
        span_ner_pred_lst = torch.softmax(span_ner_pred_lst, -1)

        return span_ner_mat_tensor, span_ner_pred_lst  # 返回的是prob不是logit


def process_long_input(model, input_ids, token_type_ids, attention_mask, seq_len_lst,
                       bert_max_len=512,
                       input_start_id=2,
                       input_end_id=3,
                       input_pad_id=0,
                       token_type_pad_id=0,
                       attention_mask_pad_id=0):
    # seq_len 包括了cls和sep的
    # Split the input to 2 overlapping chunks. Now BERT can encode inputs of which the length are up to 1024.
    batch_size, length = input_ids.shape
    start_tokens = torch.tensor([input_start_id]).to(input_ids)
    end_tokens = torch.tensor([input_end_id]).to(input_ids)
    len_start_tokens = start_tokens.shape[0]
    len_end_tokens = end_tokens.shape[0]

    input_pad_id = torch.tensor([input_pad_id]).to(input_ids)
    token_type_pad_id = torch.tensor([token_type_pad_id]).to(input_ids)
    attention_mask_pad_id = torch.tensor([attention_mask_pad_id]).to(input_ids)

    zeros = torch.tensor([0]).to(input_ids)
    ones = torch.tensor([1]).to(input_ids)

    if length <= bert_max_len:
        bert_outputs = model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return bert_outputs

    # 有超过bert_max_len CLS + Text + SEP + PAD
    new_input_ids, new_attention_mask, new_token_type_ids, new_seq_len, num_seg = [], [], [], [], []
    bdx_id_records = []
    # seq_len = attention_mask.sum(1).cpu().numpy().astype(np.int32).tolist()
    for bdx, seq_len_bdx in enumerate(seq_len_lst):
        if seq_len_bdx <= bert_max_len:
            new_input_ids.append(input_ids[bdx, :bert_max_len])
            new_attention_mask.append(attention_mask[bdx, :bert_max_len])
            new_token_type_ids.append(token_type_ids[bdx, :bert_max_len])
            new_seq_len.append(seq_len_bdx)
            bdx_id_records.append(bdx)
        else:
            text_input_ids = input_ids[bdx, len_start_tokens: seq_len_bdx - len_end_tokens]
            text_attention_mask = attention_mask[bdx, len_start_tokens: seq_len_bdx - len_end_tokens]
            text_token_type_ids = token_type_ids[bdx, len_start_tokens: seq_len_bdx - len_end_tokens]

            # start_tokens_attention_mask =

            text_length = len(text_input_ids)
            max_text_length = bert_max_len - len_start_tokens - len_end_tokens

            for idx in range(0, text_length, max_text_length):
                curr_text_input_ids = text_input_ids[idx: idx + max_text_length]
                curr_text_attention_mask = text_attention_mask[idx: idx + max_text_length]
                curr_text_token_type_ids = text_token_type_ids[idx: idx + max_text_length]

                curr_input_ids = torch.cat([start_tokens, curr_text_input_ids, end_tokens], dim=-1)
                curr_attention_mask = torch.cat([ones, curr_text_attention_mask, ones], dim=-1)
                curr_token_type_ids = torch.cat([zeros, curr_text_token_type_ids, zeros], dim=-1)

                new_input_ids.append(curr_input_ids)
                new_attention_mask.append(curr_attention_mask)
                new_token_type_ids.append(curr_token_type_ids)
                new_seq_len.append(len(curr_input_ids))
                bdx_id_records.append(bdx)

            if new_seq_len[-1] < bert_max_len:  # 最后一个要补pad
                new_input_ids[-1] = torch.cat([new_input_ids[-1], torch.tensor([input_pad_id] * (bert_max_len - new_seq_len[-1])).to(input_ids)], dim=-1)
                new_attention_mask[-1] = torch.cat([new_attention_mask[-1], torch.tensor([token_type_pad_id] * (bert_max_len - new_seq_len[-1])).to(input_ids)], dim=-1)
                new_token_type_ids[-1] = torch.cat([new_token_type_ids[-1], torch.tensor([attention_mask_pad_id] * (bert_max_len - new_seq_len[-1])).to(input_ids)], dim=-1)

    new_input_ids = torch.stack(new_input_ids, dim=0)
    new_attention_mask = torch.stack(new_attention_mask, dim=0)
    new_token_type_ids = torch.stack(new_token_type_ids, dim=0)
    # new_seq_len = torch.stack(new_seq_len, dim=0)

    bert_outputs = model(
        input_ids=new_input_ids,
        token_type_ids=new_attention_mask,
        attention_mask=new_token_type_ids,
    )

    new_last_hidden_state = bert_outputs.last_hidden_state
    resume_last_hidden_state = []
    # print('new_last_hidden_state', new_last_hidden_state.shape)
    # length_diff = length - new_last_hidden_state.shape[1]

    for bdx in range(batch_size):
        new_bdx_lst = [i for i, e in enumerate(bdx_id_records) if e == bdx]
        if len(new_bdx_lst) == 1:
            tmp_hidden_state = new_last_hidden_state[new_bdx_lst[0]]
            cur_len = len(tmp_hidden_state)
            tmp_hidden_state = torch.nn.functional.pad(tmp_hidden_state, (0, 0, 0, length - cur_len))
            resume_last_hidden_state.append(tmp_hidden_state)
        else:
            start_tmp_hidden_state = new_last_hidden_state[new_bdx_lst[0]]
            tmp_hidden_state = start_tmp_hidden_state[:-1, :]  # 去除sep
            for new_bdx in new_bdx_lst[1:-1]:
                curr_tmp_hidden_state = new_last_hidden_state[new_bdx][1:-1, :]  # 去除cls和sep
                tmp_hidden_state = torch.cat([tmp_hidden_state, curr_tmp_hidden_state], dim=0)
            last_tmp_hidden_state = new_last_hidden_state[new_bdx_lst[-1]][1:]  # 去除cls
            tmp_hidden_state = torch.cat([tmp_hidden_state, last_tmp_hidden_state], dim=0)
            cur_len = len(tmp_hidden_state)
            tmp_hidden_state = torch.nn.functional.pad(tmp_hidden_state, (0, 0, 0, length - cur_len))
            resume_last_hidden_state.append(tmp_hidden_state)

    resume_last_hidden_state = torch.stack(resume_last_hidden_state, dim=0)

    bert_outputs.last_hidden_state = resume_last_hidden_state

    return bert_outputs


class BiLSTM_CRF(nn.Module):
    def __init__(self,
                 conf,
                 tok2id,
                 pretrain_embed=None):
        super(BiLSTM_CRF, self).__init__()
        self.char2id = tok2id['char2id']
        self.id2char = tok2id['id2char']
        self.tag2id = tok2id['tag2id']
        self.id2tag = tok2id['id2tag']

        self.embed_size = conf['embed_size']
        self.rnn_hidden_size = conf['rnn_hidden_size']
        self.dropout_rate = conf['dropout_rate']
        self.vocab_size = len(self.char2id)
        self.tag_size = len(self.tag2id)

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embed_size, padding_idx=0)
        if pretrain_embed is not None:
            self.embedding_layer.weight.data.copy_(pretrain_embed)  # self.embeddings.weight.requires_grad = False
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.bilstm_layer = nn.LSTM(
            input_size=self.embed_size,
            hidden_size=self.rnn_hidden_size,
            num_layers=conf['rnn_num_layers'],
            bidirectional=True,
            batch_first=True,
            dropout=0
        )
        self.hidden2tag_layer = nn.Linear(self.rnn_hidden_size * 2, self.tag_size)
        self.crf_layer = CRF(self.tag_size, batch_first=True)

    def forward(self, chars_ids, tags_ids, seq_len, *args):
        # chars_ids [bat,len]
        # tags_ids [bat,len]
        # seq_len [bat]
        # span_ner_tgt_lst [len*(len+1)/2]
        batch_size, length = chars_ids.shape[:2]
        chars_embed = self.embedding_layer(chars_ids)  # [bat,len,emb]
        chars_embed = self.dropout_layer(chars_embed)

        # BiLSTM
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(chars_embed, list(seq_len), batch_first=True, enforce_sorted=False)
        pack_out, _ = self.bilstm_layer(pack_embed)
        bilstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
        bilstm_out = self.dropout_layer(bilstm_out)

        emission = self.hidden2tag_layer(bilstm_out)  # [bat,len,tag]
        mask = sequence_mask(seq_len, dtype=torch.uint8)
        crf_log_likelihood = self.crf_layer(emission, tags_ids, mask)
        crf_loss = -crf_log_likelihood
        decode_ids = self.crf_layer.decode(emission, mask)

        return crf_loss, emission, decode_ids

    def predict(self, chars_ids, seq_len, *args):
        chars_embed = self.embedding_layer(chars_ids)  # [bat,len,emb]
        chars_embed = self.dropout_layer(chars_embed)

        # BiLSTM
        pack_embed = torch.nn.utils.rnn.pack_padded_sequence(chars_embed, list(seq_len), batch_first=True, enforce_sorted=False)
        pack_out, _ = self.bilstm_layer(pack_embed)
        bilstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(pack_out, batch_first=True)  # [bat,len,hid]
        bilstm_out = self.dropout_layer(bilstm_out)

        emission = self.hidden2tag_layer(bilstm_out)  # [bat,len,tag]
        mask = sequence_mask(seq_len, dtype=torch.uint8)
        decode_ids = self.crf_layer.decode(emission, mask)

        return emission, decode_ids


class Bert_Seq(nn.Module):
    def __init__(self,
                 conf,
                 tok2id,
                 pretrain_embed=None):
        super(Bert_Seq, self).__init__()
        self.char2id = tok2id['char2id']
        self.id2char = tok2id['id2char']
        self.tag2id = tok2id['tag2id']
        self.id2tag = tok2id['id2tag']

        bert_model_name = conf['bert_model_dir']  # 'hfl/chinese-bert-wwm-ext'
        # bert_model_name = 'hfl/chinese-bert-wwm-ext'
        # bert_model_name = 'bert-base-chinese'
        self.bert_layer = BertModel.from_pretrained(bert_model_name)
        self.bert_conf = BertConfig.from_pretrained(bert_model_name)

        self.dropout_rate = conf['dropout_rate']
        self.vocab_size = len(self.char2id)
        self.tag_size = len(self.tag2id)

        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.hidden2tag_layer = nn.Linear(self.bert_conf.hidden_size, self.tag_size)
        self.crf_layer = CRF(self.tag_size, batch_first=True)

    def forward(self, bert_chars_ids, bert_token_type_ids, bert_attention_mask, tags_ids, seq_len, ):
        # chars_ids [bat,len]
        # tags_ids [bat,len]
        # seq_len [bat]
        chars_ids = bert_chars_ids  # use bert
        batch_size, length = chars_ids.shape[:2]
        seq_len_lst = seq_len.tolist()

        bert_outputs = self.bert_layer(input_ids=chars_ids,
                                       token_type_ids=bert_token_type_ids,
                                       attention_mask=bert_attention_mask)
        bert_out = bert_outputs.last_hidden_state

        # bert_output[CLS]和[SEP]
        bert_out_lst = [t for t in bert_out]
        for i, t in enumerate(bert_out_lst):  # iter through batch
            # tensor [len, hid]
            bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
        bert_out = torch.stack(bert_out_lst, 0)

        bert_out = self.dropout_layer(bert_out)

        emission = self.hidden2tag_layer(bert_out)  # [bat,len,tag]
        mask = sequence_mask(seq_len, dtype=torch.uint8)
        crf_log_likelihood = self.crf_layer(emission, tags_ids, mask)
        crf_loss = -crf_log_likelihood
        decode_ids = self.crf_layer.decode(emission, mask)

        return crf_loss, emission, decode_ids

    def predict(self, bert_chars_ids, bert_token_type_ids, bert_attention_mask, seq_len):
        chars_ids = bert_chars_ids
        seq_len_lst = seq_len.tolist()
        bert_outputs = self.bert_layer(input_ids=chars_ids,
                                       token_type_ids=bert_token_type_ids,
                                       attention_mask=bert_attention_mask)
        bert_out = bert_outputs.last_hidden_state

        # bert_output[CLS]和[SEP]
        bert_out_lst = [t for t in bert_out]
        for i, t in enumerate(bert_out_lst):  # iter through batch
            # tensor [len, hid]
            bert_out_lst[i] = torch.cat([t[1: 1 + seq_len_lst[i]], t[2 + seq_len_lst[i]:]], 0)
        bert_out = torch.stack(bert_out_lst, 0)

        bert_out = self.dropout_layer(bert_out)

        emission = self.hidden2tag_layer(bert_out)  # [bat,len,tag]
        mask = sequence_mask(seq_len, dtype=torch.uint8)
        decode_ids = self.crf_layer.decode(emission, mask)

        return emission, decode_ids


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
    """ mask 句子非pad部分为 1"""
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).cuda() if lengths.is_cuda else torch.arange(0, maxlen, 1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask.type(dtype)
    return mask


class CRF(nn.Module):
    """Conditional random field.

    This module implements a conditional random field [LMP01]_. The forward computation
    of this class computes the log likelihood of the given sequence of tags and
    emission score tensor. This class also has `~CRF.decode` method which finds
    the best tag sequence given an emission score tensor using `Viterbi algorithm`_.

    Args:
        num_tags: Number of tags.
        batch_first: Whether the first dimension corresponds to the size of a minibatch.

    Attributes:
        start_transitions (`~torch.nn.Parameter`): Start transition score tensor of size
            ``(num_tags,)``.
        end_transitions (`~torch.nn.Parameter`): End transition score tensor of size
            ``(num_tags,)``.
        transitions (`~torch.nn.Parameter`): Transition score tensor of size
            ``(num_tags, num_tags)``.


    .. [LMP01] Lafferty, J., McCallum, A., Pereira, F. (2001).
       "Conditional random fields: Probabilistic models for segmenting and
       labeling sequence data". *Proc. 18th International Conf. on Machine
       Learning*. Morgan Kaufmann. pp. 282–289.

    .. _Viterbi algorithm: https://en.wikipedia.org/wiki/Viterbi_algorithm
    """

    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        """Compute the conditional log likelihood of a sequence of tags given emission scores.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            tags (`~torch.LongTensor`): Sequence of tags tensor of size
                ``(seq_length, batch_size)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.
            reduction: Specifies  the reduction to apply to the output:
                ``none|sum|mean|token_mean``. ``none``: no reduction will be applied.
                ``sum``: the output will be summed over batches. ``mean``: the output will be
                averaged over batches. ``token_mean``: the output will be averaged over tokens.

        Returns:
            `~torch.Tensor`: The log likelihood. This will have size ``(batch_size,)`` if
            reduction is ``none``, ``()`` otherwise.
        """
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.type_as(emissions).sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        """Find the most likely tag sequence using Viterbi algorithm.

        Args:
            emissions (`~torch.Tensor`): Emission score tensor of size
                ``(seq_length, batch_size, num_tags)`` if ``batch_first`` is ``False``,
                ``(batch_size, seq_length, num_tags)`` otherwise.
            mask (`~torch.ByteTensor`): Mask tensor of size ``(seq_length, batch_size)``
                if ``batch_first`` is ``False``, ``(batch_size, seq_length)`` otherwise.

        Returns:
            List of list containing the best tag sequence for each batch.
        """
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.type_as(emissions)

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            # Broadcast score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emissions = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the sum of scores of all
            # possible tag sequences so far that end with transitioning from tag i to tag j
            # and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emissions

            # Sum over all possible current tags, but we're in score space, so a sum
            # becomes a log-sum-exp: for each sample, entry i stores the sum of scores of
            # all possible tag sequences so far, that end in tag i
            # shape: (batch_size, num_tags)
            next_score = torch.logsumexp(next_score, dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Sum (log-sum-exp) over all possible tags
        # shape: (batch_size,)
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape

        # Start transition and first emission
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]
        history = []

        # score is a tensor of size (batch_size, num_tags) where for every batch,
        # value at column j stores the score of the best tag sequence so far that ends
        # with tag j
        # history saves where the best tags candidate transitioned from; this is used
        # when we trace back the best tag sequence

        # Viterbi algorithm recursive case: we compute the score of the best tag sequence
        # for every possible next tag
        for i in range(1, seq_length):
            # Broadcast viterbi score for every possible next tag
            # shape: (batch_size, num_tags, 1)
            broadcast_score = score.unsqueeze(2)

            # Broadcast emission score for every possible current tag
            # shape: (batch_size, 1, num_tags)
            broadcast_emission = emissions[i].unsqueeze(1)

            # Compute the score tensor of size (batch_size, num_tags, num_tags) where
            # for each sample, entry at row i and column j stores the score of the best
            # tag sequence so far that ends with transitioning from tag i to tag j and emitting
            # shape: (batch_size, num_tags, num_tags)
            next_score = broadcast_score + self.transitions + broadcast_emission

            # Find the maximum score over all possible current tag
            # shape: (batch_size, num_tags)
            next_score, indices = next_score.max(dim=1)

            # Set score to the next score if this timestep is valid (mask == 1)
            # and save the index that produces the next score
            # shape: (batch_size, num_tags)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        # End transition score
        # shape: (batch_size, num_tags)
        score += self.end_transitions

        # Now, compute the best path for each sample

        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            # Find the tag which maximizes the score at the last timestep; this is our best tag
            # for the last timestep
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            # We trace back where the best last tag comes from, append that to our best tag
            # sequence, and trace it back again, and so on
            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list


def get_windows_sum(tensor1, mean=True):
    # tensor [batch,len,h]
    # e.g. [1,2,3]
    batch_size, length, hidden_size = tensor1.shape

    diag_mask = torch.diag_embed(torch.ones([length]), offset=0)  # [l,l]
    diag_mask = diag_mask[None, ..., None]  # [1,l,l,1]

    torch.diag_embed(tensor1, )


def get_mask_for_scale_sum(tensor1, mean=True):
    # tensor [batch,len]
    # e.g. [[1,2,3]]
    batch_size, length = tensor1.shape

    # diag_mask = torch.diag_embed(torch.ones([length]), offset=0)  # [l,l]
    # diag_mask = diag_mask[None, ..., None]  # [1,l,l,1]
    # torch.diag_embed(tensor1, )

    diag_t = torch.diag_embed(tensor1, offset=0)  # [b,l,l]
    """diag_t
    [1., 0., 0.]
    [0., 2., 0.]
    [0., 0., 3.]
    """

    cum_t = torch.cumsum(diag_t, dim=-1)  # [b,l,l]
    """cum_t
    [1., 1., 1.]
    [0., 2., 2.]
    [0., 0., 3.]
    """

    cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
    """cum_t
    [0., 0., 3.]
    [0., 2., 2.]
    [1., 1., 1.]
    """

    cum_t = torch.cumsum(cum_t, dim=-2)  # [b,l,l]
    """cum_t
    [0., 0., 3.]
    [0., 2., 2+3.]
    [1., 1+2., 1+2+3.]
    """

    cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
    """cum_t
    [1., 1+2., 1+2+3.]
    [0., 2., 2+3.]
    [0., 0., 3.]
    """
    sum_t = cum_t

    """构造相关mask矩阵"""
    ones_matrix = torch.ones(length, length).to(tensor1.device)
    triu_mask = torch.triu(ones_matrix, 0)[None, ...]  # 1,l,l  # 上三角包括对角线为1 其余为0
    ignore_mask = 1. - triu_mask

    if mean:
        # 求平均逻辑
        # 分母： 要除以来求平均
        # e.g. length=3
        heng = torch.arange(1, length + 1).to(tensor1.device)  # [1,2,3]
        heng = heng.unsqueeze(0).repeat((batch_size, 1))  # b,l
        heng = heng.unsqueeze(1).repeat((1, length, 1))  # b,l,l
        """
        [1,2,3]
        [1,2,3]
        [1,2,3]
        """
        shu = torch.arange(0, length).to(tensor1.device)  # [0,1,2]
        shu = shu.unsqueeze(0).repeat((batch_size, 1))  # b,l
        shu = shu.unsqueeze(1).repeat((1, length, 1))  # b,l,l
        shu = shu.transpose(1, 2)
        shu = - shu
        """
        [-0,-0,-0]
        [-1,-1,-1]
        [-2,-2,-2]
        """
        count = heng + shu  # 这里一开始竟然用了- --得正 日
        """
        [1,2,3]
        [0,1,2]
        [-1,0,1]  # 下三角会被mask掉不用管  Note:但是除以不能为0！
        """

        # 把下三角强制变为1 避免计算溢常 因为后面会mask 没关系
        count = count * triu_mask + ignore_mask

        sum_t = sum_t / count

    # 再把下三角强制变为0
    sum_t = sum_t * triu_mask
    return sum_t


def get_mask_for_scale(tensor1, mode='max'):
    # tensor [batch,len]
    # e.g. [[1,2,3]]
    batch_size, length = tensor1.shape

    diag_t = torch.diag_embed(tensor1, offset=0)  # [b,l,l]
    """diag_t
    [1., 0., 0.]
    [0., 2., 0.]
    [0., 0., 3.]
    """

    cum_t = torch.cumsum(diag_t, dim=-1)  # [b,l,l]
    """cum_t
    [1., 1., 1.]
    [0., 2., 2.]
    [0., 0., 3.]
    """

    cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
    """cum_t
    [0., 0., 3.]
    [0., 2., 2.]
    [1., 1., 1.]
    """

    if mode in ['max', 'min']:
        triu_mask = torch.triu(torch.ones([length, length]), diagonal=1).to(tensor1.device)[None, ...]  # 1,l,l
        """triu_mask
        [0., 1., 1.]
        [0., 0., 1.]
        [0., 0., 0.]
        """
        inv_triu_mask = torch.flip(triu_mask, dims=[-1])
        """inv_triu_mask
        [1., 1., 0.]
        [1., 0., 0.]
        [0., 0., 0.]
        """
        if mode == 'max':
            inv_triu_mask = inv_triu_mask * -1e12
            cum_t = cum_t + inv_triu_mask
            """cum_t
            [-inf., -inf., 3.]
            [-inf., 2., 2.]
            [1., 1., 1.]
            """

            cum_t, _ = torch.cummax(cum_t, dim=-2)  # [b,l,l]
            """cum_t 0: denote -inf
            [max0., max0., max3.]
            [max0+0., max2+0., max2+3.]
            [max1+0+0., max1+2+0., max1+2+3.]
            """

            cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
            """cum_t
            [max1., max1+2., max1+2+3.]
            [max0., max2., max2+3.]
            [max0., max0., max3.]
            """

            cum_t = torch.triu(cum_t, diagonal=0)  # [b,l,l]
            """cum_t
            [max1., max1+2., max1+2+3.]
            [0., max2., max2+3.]
            [0., 0., max3.]
            """


        elif mode == 'min':
            inv_triu_mask = inv_triu_mask * 1e12
            cum_t = cum_t + inv_triu_mask
            """cum_t
            [inf., inf., 3.]
            [inf., 2., 2.]
            [1., 1., 1.]
            """

            cum_t, _ = torch.cummin(cum_t, dim=-2)  # [b,l,l]
            """cum_t 0: denote inf
            [min0., min0., min3.]
            [min0+0., min2+0., min2+3.]
            [min1+0+0., min1+2+0., min1+2+3.]
            """

            cum_t = torch.flip(cum_t, dims=[-2])  # [b,l,l]
            """cum_t
            [min1., min1+2., min1+2+3.]
            [min0., min2., min2+3.]
            [min0., min0., min3.]
            """

            cum_t = torch.triu(cum_t, diagonal=0)  # [b,l,l]
            """cum_t
            [min1., min1+2., min1+2+3.]
            [0., min2., min2+3.]
            [0., 0., min3.]
            """

    return cum_t


def get_windows_sum(tensor1, mean=True):
    # tensor [batch,len,h]
    # e.g. [1,2,3]
    batch_size, length, hidden_size = tensor1.shape
    cum_t = torch.cumsum(tensor1, dim=1)  # b,l,h
    """
    [1,1+2,1+2+3]
    """

    minus_t = torch.unsqueeze(tensor1, 1)  # b,1,l,h
    minus_t = minus_t.repeat((1, length, 1, 1))  # b,l,l,h
    minus_t = minus_t.transpose(1, 2)  # b,l,l,h
    minus_t = - minus_t
    """
    [-1,-1,-1]
    [-2,-2,-2]
    [-3,-3,-3]
    """

    cum_t = torch.unsqueeze(cum_t, 1)  # b,1,l,h
    mat_t = torch.cat([cum_t, minus_t], dim=1)[:, :-1, :, :]  # b,l,l,h
    mat_t = torch.cumsum(mat_t, dim=1)  # b,l,l,h
    """
    [1,1+2,1+2+3]
    [-1,-1,-1]
    [-2,-2,-2]
    run cumsum in colum axis
    = 
    [1,1+2,1+2+3]
    [N,2,2+3]
    [N,N,3]
    """

    """构造相关mask矩阵"""
    ones_matrix = torch.ones(length, length).to(tensor1.device)
    triu_mask = torch.triu(ones_matrix, 0)[None, ..., None]  # 1,l,l,1  # 上三角包括对角线为1 其余为0
    ignore_mask = 1. - triu_mask

    if mean:
        # 求平均逻辑
        # 分母： 要除以来求平均
        # e.g. length=3
        heng = torch.arange(1, length + 1).to(tensor1.device)  # [1,2,3]
        heng = heng.unsqueeze(0).repeat((batch_size, 1))  # b,l
        heng = heng.unsqueeze(1).repeat((1, length, 1))  # b,l,l
        """
        [1,2,3]
        [1,2,3]
        [1,2,3]
        """
        shu = torch.arange(0, length).to(tensor1.device)  # [0,1,2]
        shu = shu.unsqueeze(0).repeat((batch_size, 1))  # b,l
        shu = shu.unsqueeze(1).repeat((1, length, 1))  # b,l,l
        shu = shu.transpose(1, 2)
        shu = - shu
        """
        [-0,-0,-0]
        [-1,-1,-1]
        [-2,-2,-2]
        """
        count = heng + shu  # 这里一开始竟然用了- --得正 日
        """
        [1,2,3]
        [0,1,2]
        [-1,0,1]  # 下三角会被mask掉不用管  Note:但是除以不能为0！
        """
        count = count.unsqueeze(-1)  # b,l,l,1

        # 把下三角强制变为1 避免计算溢常 因为后面会mask 没关系
        count = count * triu_mask + ignore_mask

        mat_t = mat_t / count

    # 再把下三角强制变为0
    mat_t = mat_t * triu_mask
    return mat_t


def get_windows_sum_max_pool(tensor1):
    # tensor [batch,len,h]
    # e.g. [1,2,3]
    batch_size, length, hidden_size = tensor1.shape

    max_num = torch.max(tensor1)
    tensor1 = tensor1 - max_num

    tensor1 = torch.exp(tensor1)
    # print('exp', tensor1)
    cum_t = torch.cumsum(tensor1, dim=1)  # b,l,h
    """
    [1,1+2,1+2+3]
    """

    minus_t = torch.unsqueeze(tensor1, 1)  # b,1,l,h
    minus_t = minus_t.repeat((1, length, 1, 1))  # b,l,l,h
    minus_t = minus_t.transpose(1, 2)  # b,l,l,h
    minus_t = - minus_t
    """
    [-1,-1,-1]
    [-2,-2,-2]
    [-3,-3,-3]
    """

    cum_t = torch.unsqueeze(cum_t, 1)  # b,1,l,h
    mat_t = torch.cat([cum_t, minus_t], dim=1)[:, :-1, :, :]  # b,l,l,h
    mat_t = torch.cumsum(mat_t, dim=1)  # b,l,l,h
    """
    [1,1+2,1+2+3]
    [-1,-1,-1]
    [-2,-2,-2]
    run cumsum in colum axis
    = 
    [1,1+2,1+2+3]
    [N,2,2+3]
    [N,N,3]
    """
    # 把下三角强制变为1 避免计算溢常 因为后面会mask 没关系
    ones_matrix = torch.ones(length, length).to(tensor1.device)
    triu_mask = torch.triu(ones_matrix, 0)[None, ..., None]  # 1,l,l,1  # 上三角包括对角线为1 其余为0
    ignore_mask = 1. - triu_mask
    # mat_t = mat_t * triu_mask + ignore_mask
    mat_t = (torch.relu(mat_t) + 1e-12) * triu_mask + ignore_mask

    # print('before_log', mat_t)
    mat_t = torch.log(mat_t)
    # print('log', mat_t)
    mat_t = mat_t + max_num

    # 再把下三角强制变为0
    mat_t = mat_t * triu_mask
    return mat_t


def get_windows_sum_min_pool1(tensor1):
    record_ = tensor1
    # tensor [batch,len,h]
    # e.g. [1,2,3]
    batch_size, length, hidden_size = tensor1.shape

    tensor1 = -tensor1

    max_num = torch.max(tensor1)
    tensor1 = tensor1 - max_num

    tensor1 = torch.exp(tensor1)

    cum_t = torch.cumsum(tensor1, dim=1)  # b,l,h
    """
    [1,1+2,1+2+3]
    """

    minus_t = torch.unsqueeze(tensor1, 1)  # b,1,l,h
    minus_t = minus_t.repeat((1, length, 1, 1))  # b,l,l,h
    minus_t = minus_t.transpose(1, 2)  # b,l,l,h
    minus_t = - minus_t
    """
    [-1,-1,-1]
    [-2,-2,-2]
    [-3,-3,-3]
    """

    cum_t = torch.unsqueeze(cum_t, 1)  # b,1,l,h
    mat_t = torch.cat([cum_t, minus_t], dim=1)[:, :-1, :, :]  # b,l,l,h
    mat_t = torch.cumsum(mat_t, dim=1)  # b,l,l,h
    """
    [1,1+2,1+2+3]
    [-1,-1,-1]
    [-2,-2,-2]
    run cumsum in colum axis
    = 
    [1,1+2,1+2+3]
    [N,2,2+3]
    [N,N,3]
    """
    # 把下三角强制变为1 避免计算溢常 因为后面会mask 没关系
    ones_matrix = torch.ones(length, length).to(tensor1.device)
    triu_mask = torch.triu(ones_matrix, 0)[None, ..., None]  # 1,l,l,1  # 上三角包括对角线为1 其余为0
    ignore_mask = 1. - triu_mask
    mat_t = mat_t * triu_mask + ignore_mask
    mat_t = torch.maximum(mat_t, torch.Tensor([1e-12]).to(tensor1.device))
    # mat_t = (torch.relu(mat_t) + 1e-12)
    mat_t = mat_t * triu_mask + ignore_mask

    # print('before_log', mat_t)
    if torch.isnan(mat_t).sum() != 0:
        print('before log')
        print(mat_t)
        print(record_)
        np.save('mat.npz', mat_t.cpu().detach().numpy())
        np.save('input.npz', record_.cpu().detach().numpy())
        exit(0)

    mat_t = torch.log(mat_t)
    if torch.isnan(mat_t).sum() != 0:
        print('after log')
        print(mat_t)
        print(record_)
        np.save('mat.npz', mat_t.cpu().detach().numpy())
        np.save('input.npz', record_.cpu().detach().numpy())
        exit(0)
    # print('log', mat_t)
    mat_t = mat_t + max_num
    # 再把下三角强制变为0
    mat_t = mat_t * triu_mask

    mat_t = -mat_t

    return mat_t


def get_windows_sum_min_pool(tensor1):
    mat_t = get_windows_sum_max_pool(-tensor1)
    mat_t = -mat_t
    return mat_t


def multilabel_categorical_crossentropy(y_pred, y_true):
    """多标签分类的交叉熵
    说明：y_true和y_pred的shape一致，y_true的元素非0即1，
         1表示对应的类为目标类，0表示对应的类为非目标类。
    警告：请保证y_pred的值域是全体实数，换言之一般情况下y_pred
         不用加激活函数，尤其是不能加sigmoid或者softmax！预测
         阶段则输出y_pred大于0的类。如有疑问，请仔细阅读并理解
         本文。
    """
    # y_pred [*, num_class]
    # y_true [*, num_class]
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return neg_loss + pos_loss


class FocalLoss1(nn.Module):
    """Multi-class Focal loss implementation"""

    def __init__(self, gamma=2, weight=None, ignore_index=-100):
        super(FocalLoss1, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = torch.nn.functional.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1 - pt) ** self.gamma * logpt
        loss = torch.nn.functional.nll_loss(logpt, target, self.weight, ignore_index=self.ignore_index)
        return loss


class BCEFocalLoss(torch.nn.Module):
    """
    二分类的Focalloss alpha 固定
    """

    def __init__(self, gamma=2, alpha=0.25, reduction='elementwise_mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, _input, target):
        pt = torch.sigmoid(_input)
        alpha = self.alpha
        loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
               (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.reduction == 'elementwise_mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, size_avg=False, episode=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_avg = size_avg
        self.episode = episode
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha], dtype=torch.float)
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha, dtype=torch.float)

    def forward(self, logits, labels):
        batch_size = logits.size(0)
        num_labels = logits.size(1)
        softmax_probobilities = torch.nn.functional.softmax(logits, dim=-1)
        probobilities = softmax_probobilities.gather(dim=1, index=labels.view((batch_size, 1))).view(-1)  # [batch, nlabels] > [batch, 1] > [batch]
        if self.episode is not None:
            probobilities = probobilities + self.episode
        log_probobilities = torch.log(probobilities)

        if self.alpha is not None:
            self.alpha = self.alpha.to(logits)
            alpha_weights = self.alpha.gather(dim=0, index=labels)
            log_probobilities = log_probobilities * alpha_weights

        loss = -1.0 * (1.0 - probobilities) ** self.gamma * log_probobilities
        if self.size_avg:
            loss = loss.mean()
        return loss


class ClassBalancedLoss(nn.Module):
    def __init__(self, loss_type='cross_entropy',
                 gamma=None,
                 alpha=None,
                 weights=None,
                 episode=None):
        super(ClassBalancedLoss, self).__init__()
        '''
          Args:
            loss_type: ['cross_entropy', 'focal_loss', 'cb_focal_loss', 'cb_cross_entropy']
            gamma: value for gamma parameter of focal loss
            alpha: value for alpha parameter of focal loss
            weight: value for weight parameter of class balanced loss
            size_avg:
          Return:
            loss
        '''
        self.loss_type = loss_type
        self.gamma = gamma
        self.alpha = alpha
        self.weights = weights
        self.episode = episode

    def forward(self, logits, labels, mask=None):
        num_labels = logits.size(-1)
        if logits.dim() > 2:
            logits = logits.view((-1, num_labels))
        if labels.dim() > 1:
            labels = labels.view(-1)
        if mask is not None and mask.dim() > 1:
            mask = mask.view(-1)
        if mask is not None:
            active_tokens = mask == 1
            labels = labels[active_tokens]
            logits = logits[active_tokens]

        if self.loss_type == 'cross_entropy':
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(input=logits, target=labels)
        elif self.loss_type == 'cb_cross_entropy':
            if self.weights is not None:
                self.weights = torch.tensor(self.weights, dtype=torch.float).to(logits)
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.weights)
            loss = loss_fct(input=logits, target=labels)
        elif self.loss_type == 'focal_loss':
            loss_fct = FocalLoss(gamma=self.gamma, alpha=self.alpha, size_avg=True, episode=self.episode)
            loss = loss_fct(logits=logits, labels=labels)
        else:  # 'cb_focal_loss'
            loss_fct = FocalLoss(gamma=self.gamma, alpha=self.alpha, size_avg=False, episode=self.episode)
            losses = loss_fct(logits=logits, labels=labels)
            if self.weights is not None:
                self.weights = torch.tensor(self.weights, dtype=torch.float).to(logits)
                picked_weights = self.weights.gather(dim=0, index=labels)
                losses = losses * picked_weights
            loss = losses.mean()
        return loss


if __name__ == '__main__':
    # inp = np.load('input.npz.npy')
    # tinp = torch.FloatTensor(inp)
    #
    # mat = np.load('mat.npz.npy')
    # tmat = torch.FloatTensor(mat)
    #
    # get_windows_sum_min_pool1(tinp[:1])
    #
    # exit(0)

    x = torch.Tensor([1, 2, 3])
    # x = -torch.Tensor([1,2,3])
    x = x[None, ...]
    x = x[..., None]
    #
    print('min', get_windows_sum_min_pool(x))
    print('max', get_windows_sum_max_pool(x))
    print('mean', get_windows_sum(x, mean=True))
    print('sum', get_windows_sum(x, mean=False))

    x = torch.Tensor([-1, -2, -3])
    x = x[None, ...]
    print('sum', get_mask_for_scale_sum(x, mean=False))
    print('mean', get_mask_for_scale_sum(x, mean=True))
    print('max', get_mask_for_scale(x, mode='max'))
    print('min', get_mask_for_scale(x, mode='min'))

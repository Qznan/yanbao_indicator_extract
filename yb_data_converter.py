import json
from pathlib import Path
import numpy as np
import copy
import datautils as utils
import random
from datautils import NerExample
import argparse


def gen_alias_map():
    # 指标名标准化
    items = utils.file2items('data/yanbao/标准化信息.txt')
    alias_map_dct = {}
    for item in items:
        stand = item[0]
        # alias_map_dct[stand] = stand
        for cand in item[1:]:
            if cand and cand != ' ':
                alias_map_dct[cand] = stand
    utils.save_json(alias_map_dct, 'data/yanbao/indi_alias_map.json')

    # data_dir = Path('data/yanbao/data')
    # files = data_dir.glob('*.json')
    # for f in files:
    #     jdata = utils.load_json_file(f)
    #     data = jdata['data']


def norm_indi_fn_yimei():
    alias_map_dct = utils.load_json('data/yanbao/yimei/indi_alias_map.json')

    def norm_indi(itp: str):
        itp = itp.strip()
        itp = itp.replace(' ', '')
        itp = itp.replace('\n', '')
        itp = itp.replace('\r', '')
        itp = alias_map_dct.get(itp, itp)
        return itp

    return norm_indi

def norm_atp_fn_yimei():
    alias_map_dct = {
        '医美': '行业',
        '地区': '区域',
        '地域': '区域',
        '年龄分布': '年龄',
        '非手术类项目': '项目',
        '非手术类项目注射项目': '项目',
        '非项目': '项目',
    }

    def norm_atp(atp: str):
        atp = atp.strip()
        atp = atp.strip(';；')
        atp = atp.replace(' ', '')
        atp = atp.replace('\n', '')
        atp = atp.replace('\r', '')
        atp = alias_map_dct.get(atp, atp)

        start_chars = ['业务',
                       '产品',
                       '公司',
                       '区域',
                       '时间',
                       '行业',
                       '渠道',
                       ]

        for chars in start_chars:
            if atp[:2] == chars:
                atp = chars
        return atp

    return norm_atp


def norm_indi_fn_normal():
    def norm_indi(itp: str):
        itp = itp.strip()
        itp = itp.replace(' ', '')
        itp = itp.replace('\n', '')
        itp = itp.replace('\r', '')
        return itp
    return norm_indi



def get_atp():
    norm_atp = norm_atp_fn_yimei()
    schema = utils.load_json('data/yanbao/schema.json')
    atps = set()
    for k, v in schema.items():
        for atp in v:
            atps.add(norm_atp(atp))
    atps = sorted(atps)
    # utils.list2file(atps, 'data/yanbao/schema_apts.json')
    utils.save_json(atps, 'data/yanbao/schema_apts.json')


def anal_exm():
    indi_exm_lst = utils.load_jsonl('data/yanbao/exm/indi_exm_lst.jsonl')
    new_indi_exm_lst = []
    print('total', len(indi_exm_lst))
    lens = []
    min1024 = 0
    for exm in indi_exm_lst:
        lens.append(len(exm['ctx']))
        if len(exm['ctx']) < 1000:
            new_indi_exm_lst.append(exm)
        if len(exm['ctx']) < 1024:
            min1024 += 1
    print(min1024)

    print(len(new_indi_exm_lst))
    # utils.save_jsonl(new_indi_exm_lst, 'data/yanbao/exm/indi_exm_lst_maxlen1000.jsonl')

    print(utils.stats_lst(lens))


def convert2nerexm():
    itp2id = utils.file2list('data\yanbao\indi_lst.txt')
    itp2id = {itp: i for i, itp in enumerate(itp2id)}
    exm_lst = []
    inst_lst = utils.load_jsonl('data/yanbao/exm/indi_exm_lst_maxlen1000.jsonl')
    for inst in inst_lst:
        text = inst['ctx']
        itp = inst['indicator_name']
        ivl = inst['indicator_value']
        v, start, end = ivl
        ctx_start, ctx_end = inst['ctx_offset']
        start = int(start) - int(ctx_start)
        end = int(end) - int(ctx_start)
        assert (text[start:end]) == v
        ent_dct = {itp2id[itp]: [[start, end]]}
        exm = NerExample(list(text), ent_dct=ent_dct)
        exm.truncate(max_size=512, direction='tail')
        exm_lst.append(exm)

    print(len(exm_lst))
    exm_lst = NerExample.combine_by_text(exm_lst)
    print(len(exm_lst))
    NerExample.save_to_jsonl(exm_lst, 'data/yanbao/exm/indi_nerexm_lst_maxlen512.jsonl')


def gen_schema(json_files, out_schema_file):
    schema = {}
    norm_indi = norm_indi_fn_yimei()
    norm_atp = norm_atp_fn_yimei()
    for json_file in json_files:
        data = utils.load_json(json_file)
        source = data['source_content']
        data_lst = data['data']

        for exm in data_lst:
            for indi in exm['indicators']:
                itp = indi['indicator_name']
                itp = norm_indi(itp)
                if itp == '':
                    print(exm)
                    raise Exception
                if itp not in schema:
                    schema[itp] = set()

                for atp in indi['indicator_element']:
                    schema[itp].add(norm_atp(atp))

    for itp in schema:
        schema[itp] = sorted(schema[itp])

    utils.save_json(schema, out_schema_file)
    print(sorted(schema.keys()))
    print(len(schema.keys()))

    utils.list2file(sorted(schema.keys()), 'data/yanbao/indi_lst.txt')

    atps = set()
    for k, v in schema.items():
        for apt in v:
            atps.add(apt)
    atps = sorted(atps)
    print('apts', atps)
    print(len(atps))
    utils.list2file(atps, 'data/yanbao/atp_lst.txt')


def gen_schema0617(json_files, out_schema_file_dir, itp_norm_fn=None, atp_norm_fn=None):
    utils.make_sure_dir_exist(out_schema_file_dir)
    schema = {}
    for doc_id, json_file in json_files:
        data = utils.load_json(json_file)
        source = data['source_content']
        data_lst = data['data']
        # print(data_lst)

        for exm in data_lst:
            for indi in exm['indicators']:
                itp_name = indi['indicator_name']
                if itp_norm_fn is not None: itp_name = itp_norm_fn(itp_name)
                if itp_name == '':
                    print('Except:', exm)
                    raise Exception
                schema.setdefault(itp_name, set())
                for atp_name in indi['indicator_element']:
                    if atp_norm_fn is not None: atp_name = atp_norm_fn(atp_name)
                    schema[itp_name].add(atp_name)

    for itp_name in schema:
        schema[itp_name] = sorted(schema[itp_name])

    utils.save_json(schema, f'{out_schema_file_dir}/scheam.json')

    itp_name_lst = sorted(schema.keys())
    atp_name_lst = set()
    for atp_set in schema.values():
        atp_name_lst.update(atp_set)
    atp_name_lst = sorted(atp_name_lst)

    utils.list2file([f'属性-{atp_name}' for atp_name in atp_name_lst], f'{out_schema_file_dir}/atp_lst.txt')
    utils.list2file([f'指标-{itp_name}' for itp_name in itp_name_lst], f'{out_schema_file_dir}/itp_lst.txt')

    utils.list2file(['[PAD]', 'O'] +
                    [f'指标-{itp_name}' for itp_name in itp_name_lst],
                    f'{out_schema_file_dir}/full_ent_lst_remove_atp.txt')

    utils.list2file(['[PAD]', 'O'] +
                    [f'属性-{atp_name}' for atp_name in atp_name_lst] +
                    [f'指标-{itp_name}' for itp_name in itp_name_lst]
                    ,
                    f'{out_schema_file_dir}/full_ent_lst.txt')

    utils.save_json(schema, f'{out_schema_file_dir}/scheam.json')
    print(f'指标数量:{len(itp_name_lst)}\n'
          f'属性数量:{len(atp_name_lst)}\n'
          f'指标一览:{itp_name_lst}\n'
          f'属性一览:{atp_name_lst}\n'
          )


def get_len(json_files):
    lens = []
    num_sent = []
    len_per_sent = []
    for json_file in json_files:
        data = utils.load_json(json_file)
        source = data['source_content']
        data_lst = data['data']
        lens.append(len(source))
        sent_lst = source.split('。')
        if sent_lst[-1] == '':
            sent_lst = sent_lst[:-1]
        num_sent.append(len(sent_lst))
        len_per_sent.extend([len(s) for s in sent_lst])

    print('\nlens')
    print(lens)
    print(np.mean(lens))
    print(np.max(lens))
    print(np.min(lens))

    print('\nnum_sent')
    print(num_sent)
    print(np.mean(num_sent))
    print(np.max(num_sent))
    print(np.min(num_sent))

    print('\nlen_per_sent')
    print(len_per_sent)
    print(np.mean(len_per_sent))
    print(np.max(len_per_sent))
    print(np.min(len_per_sent))


def anan_ori_data(files):
    all_sent_lst = []
    for json_file in files:
        data = utils.load_json(json_file)
        source = data['source_content']
        data_lst = data['data']

        sent_lst = source.split('。')
        if sent_lst[-1] == '':
            sent_lst = sent_lst[:-1]
        all_sent_lst.extend(sent_lst)

    utils.list2file(all_sent_lst, 'data/yanbao/sent/all_sent_lst.txt')
    utils.list2file([e for e in all_sent_lst if len(e) > 5000], 'data/yanbao/sent/all_sent_lst_5000.txt')
    sent_len_lst = [len(e) for e in sent_lst]

    print(utils.stats_lst(sent_len_lst))


def convert_data(files):
    norm_indi = norm_indi_fn()
    norm_atp = norm_atp_fn()
    indi_exm_lst = []
    unmatch_count = 0
    match_count = 0
    for json_file in files:
        data = utils.load_json(json_file)
        source = data['source_content']
        data_lst = data['data']

        for exm in data_lst:
            content_start, content_end = exm['content_offset']
            content_start, content_end = int(content_start), int(content_end)

            for indi in exm['indicators']:  # 遍历每个样本
                indi_info = copy.deepcopy(indi)
                min_start = content_start
                max_end = content_end
                indi_info['indicator_name'] = norm_indi(indi_info['indicator_name'])
                indi_info.pop('indicator_supplement')

                v, start, end = indi_info['indicator_value']  # 有-1 -1  也有start=end
                start, end = int(start), int(end)
                offset_v = source[start:end]
                if offset_v != v or start == end:
                    # print('file', json_file)
                    # print(start, end)
                    # print('offset_v', repr(offset_v))
                    # print('v', repr(v))
                    # print()
                    unmatch_count += 1
                    continue
                else:
                    match_count += 1

                if start < min_start: min_start = start
                if end > max_end: max_end = end

                ele_unmatch_flag = False

                # norm atp
                new_dct = {}
                for k, v in indi_info['indicator_element'].items():
                    new_dct[norm_atp(k)] = v
                indi_info['indicator_element'] = new_dct

                for v, start, end in indi_info['indicator_element'].values():

                    start, end = int(start), int(end)  # 有-1 -1  也有start=end

                    offset_v = source[start:end]
                    if offset_v != v or start == end:
                        # print('file', json_file)
                        # print(start, end)
                        # print('offset_v', repr(offset_v))
                        # print('v', repr(v))
                        # print()
                        unmatch_count += 1
                        ele_unmatch_flag = True
                        continue
                    else:
                        match_count += 1
                    if start < min_start: min_start = start
                    if end > max_end: max_end = end

                if ele_unmatch_flag:  # 有1个不匹配
                    continue

                indi_info['ctx'] = source[min_start: max_end]
                indi_info['ctx_offset'] = [min_start, max_end]

                indi_exm_lst.append(indi_info)
    print(unmatch_count)
    print(match_count)
    utils.save_jsonl(indi_exm_lst, 'data/yanbao/exm/indi_exm_lst.jsonl')


def convert_data1(files):
    norm_indi = norm_indi_fn()
    norm_atp = norm_atp_fn()
    indi_exm_lst = []
    unmatch_count = 0
    match_count = 0
    source_dct = {}
    sent_exm = []

    for json_file in files:
        data = utils.load_json(json_file)
        source = data['source_content']
        data_lst = data['data']

        source_id = len(source_dct)
        source_dct[source_id] = source
        source_info = {'source_id': source_id}
        ent_lst = []
        itp2atp_lst = []

        for exm in data_lst:
            content_start, content_end = exm['content_offset']
            content_start, content_end = int(content_start), int(content_end)

            for indi in exm['indicators']:  # 遍历每个样本
                indi_info = copy.deepcopy(indi)
                min_start = content_start
                max_end = content_end
                indi_info['indicator_name'] = norm_indi(indi_info['indicator_name'])
                indi_info.pop('indicator_supplement')

                v, start, end = indi_info['indicator_value']  # 有-1 -1  也有start=end
                start, end = int(start), int(end)
                offset_v = source[start:end]
                if offset_v != v or start == end:
                    # print('file', json_file)
                    # print(start, end)
                    # print('offset_v', repr(offset_v))
                    # print('v', repr(v))
                    # print()
                    unmatch_count += 1
                    continue
                else:
                    match_count += 1

                if start < min_start: min_start = start
                if end > max_end: max_end = end

                ele_unmatch_flag = False

                # norm atp
                new_dct = {}
                for k, v in indi_info['indicator_element'].items():
                    new_dct[norm_atp(k)] = v
                indi_info['indicator_element'] = new_dct

                for v, start, end in indi_info['indicator_element'].values():

                    start, end = int(start), int(end)  # 有-1 -1  也有start=end

                    offset_v = source[start:end]
                    if offset_v != v or start == end:
                        # print('file', json_file)
                        # print(start, end)
                        # print('offset_v', repr(offset_v))
                        # print('v', repr(v))
                        # print()
                        unmatch_count += 1
                        ele_unmatch_flag = True
                        continue
                    else:
                        match_count += 1
                    if start < min_start: min_start = start
                    if end > max_end: max_end = end

                if ele_unmatch_flag:  # 有1个不匹配
                    continue

                indi_info['ctx'] = source[min_start: max_end]
                indi_info['ctx_offset'] = [min_start, max_end]

                indi_exm_lst.append(indi_info)

                itp = indi_info['indicator_name']
                itp_val, start, end = indi_info['indicator_value']
                ent_lst.append([f'指标-{itp}', start, end])
                itp2atp = [[f'指标-{itp}', start, end, itp_val]]
                for atp, (atp_val, start, end) in indi_info['indicator_element'].items():
                    ent_lst.append([f'属性-{atp}', start, end])
                    itp2atp.append([f'属性-{atp}', start, end, atp_val])

                itp2atp_lst.append(itp2atp)

        source_info['ent_lst'] = ent_lst
        source_info['itp2atp_lst'] = itp2atp_lst
        sent_exm.append(source_info)

    print(unmatch_count)
    print(match_count)
    # utils.save_jsonl(indi_exm_lst, 'data/yanbao/exm/indi_exm_lst.jsonl')
    utils.save_jsonl(sent_exm, 'data/yanbao/sent/sent_exm.jsonl')
    utils.save_json(source_dct, 'data/yanbao/sent/sent_dct.json')

    ner_exm_lst = []
    for item in sent_exm:
        char_lst = list(source_dct[item['source_id']])
        exm = NerExample.from_ent_lst(char_lst, item['ent_lst'])
        # print(exm)
        exm.source_id = item['source_id']
        exm.itp2atp_lst = item['itp2atp_lst']
        ner_exm_lst.append(exm)
    NerExample.save_to_jsonl(ner_exm_lst, 'data/yanbao/sent/doc_exm.jsonl', for_human_read=True, external_attrs=['itp2atp_lst', 'source_id'])

    # process_doc_re()  # 生成属性抽取mrc数据

    # 开始按句号分样本
    new_ner_exm_lst = []
    for idx, dco_exm in enumerate(ner_exm_lst):
        split_exm_lst = NerExample.split_exm_by_deli(dco_exm)  # TODO
        # split_exm_lst = NerExample.split_exm_by_deli_multi_sent(dco_exm)
        for sent_idx, split_exm in enumerate(split_exm_lst):
            split_exm.source_id = f'{dco_exm.source_id}_{sent_idx}'
        new_ner_exm_lst.extend(split_exm_lst)

    # 按是否有实体来分
    # pos_exm_lst = [exm for exm in new_ner_exm_lst if not exm.is_neg()]
    # neg_exm_lst = [exm for exm in new_ner_exm_lst if exm.is_neg()]

    # 只按指标来分
    pos_exm_lst = [exm for exm in new_ner_exm_lst if exm.has_ent_type_startswith('指标')]
    neg_exm_lst = [exm for exm in new_ner_exm_lst if not exm.has_ent_type_startswith('指标')]

    print('total', len(new_ner_exm_lst))
    print('num_neg', len(neg_exm_lst))
    print('num_pos', len(pos_exm_lst))
    print('负样本是正样本的几倍:', len(neg_exm_lst) / len(new_ner_exm_lst))
    dir_name = 'sent_mzb'
    dir_name = 'sent_szb'
    NerExample.save_to_jsonl(new_ner_exm_lst, f'data/yanbao/{dir_name}/sent_exm_final.jsonl', for_human_read=True, external_attrs=['source_id'])
    NerExample.save_to_jsonl(pos_exm_lst, f'data/yanbao/{dir_name}/sent_exm_final_pos.jsonl', for_human_read=True, external_attrs=['source_id'])
    NerExample.save_to_jsonl(neg_exm_lst, f'data/yanbao/{dir_name}/sent_exm_final_neg.jsonl', for_human_read=True, external_attrs=['source_id'])
    exit(0)

    random.seed = 1234
    combine_exm_lst = pos_exm_lst + random.sample(neg_exm_lst, len(pos_exm_lst))
    random.shuffle(combine_exm_lst)
    NerExample.save_to_jsonl(combine_exm_lst, 'data/yanbao/sent/sent_exm_final_combine.jsonl', for_human_read=True, external_attrs=['source_id'])
    NerExample.stats(combine_exm_lst)

    # NerExample.stats(new_ner_exm_lst)
    # NerExample.stats(pos_exm_lst)
    # NerExample.stats(neg_exm_lst)


def convert_data0617(files, out_file_dir, itp_norm_fn=None, atp_norm_fn=None):
    indi_exm_lst = []
    unmatch_count = 0
    match_count = 0
    doc_level_item_lst = []

    for doc_id, json_file in files:
        data = utils.load_json(json_file)
        text = data['source_content']

        ent_lst = []
        itp2atp_lst = []
        doc_level_item = {'source_id': doc_id, 'text': text}

        for exm in data['data']:
            content_start, content_end = exm['content_offset']
            content_start, content_end = int(content_start), int(content_end)

            for indi in exm['indicators']:  # 遍历每个样本
                indi_info = copy.deepcopy(indi)
                min_start = content_start
                max_end = content_end
                if itp_norm_fn is not None:
                    indi_info['indicator_name'] = itp_norm_fn(indi_info['indicator_name'])
                indi_info.pop('indicator_supplement')

                v, start, end = indi_info['indicator_value']  # 有-1 -1  也有start=end
                start, end = int(start), int(end)
                offset_v = text[start:end]
                if offset_v != v or start == end:
                    # print('file', json_file)
                    # print(start, end)
                    # print('offset_v', repr(offset_v))
                    # print('v', repr(v))
                    # print()
                    unmatch_count += 1
                    continue
                else:
                    match_count += 1

                if start < min_start: min_start = start
                if end > max_end: max_end = end

                ele_unmatch_flag = False

                if atp_norm_fn is not None:  # norm atp
                    new_dct = {}
                    for k, v in indi_info['indicator_element'].items():
                        new_dct[atp_norm_fn(k)] = v
                    indi_info['indicator_element'] = new_dct

                for v, start, end in indi_info['indicator_element'].values():
                    start, end = int(start), int(end)  # 有-1 -1  也有start=end
                    offset_v = text[start:end]
                    if offset_v != v or start == end:
                        # print('file', json_file)
                        # print(start, end)
                        # print('offset_v', repr(offset_v))
                        # print('v', repr(v))
                        # print()
                        unmatch_count += 1
                        ele_unmatch_flag = True
                        continue
                    else:
                        match_count += 1
                    if start < min_start: min_start = start
                    if end > max_end: max_end = end

                if ele_unmatch_flag:  # 有1个不匹配
                    continue

                indi_info['ctx'] = text[min_start: max_end]
                indi_info['ctx_offset'] = [min_start, max_end]

                indi_exm_lst.append(indi_info)

                itp = indi_info['indicator_name']
                itp_val, start, end = indi_info['indicator_value']
                ent_lst.append([f'指标-{itp}', start, end])
                itp2atp = [[f'指标-{itp}', start, end, itp_val]]
                for atp, (atp_val, start, end) in indi_info['indicator_element'].items():
                    ent_lst.append([f'属性-{atp}', start, end])
                    itp2atp.append([f'属性-{atp}', start, end, atp_val])

                itp2atp_lst.append(itp2atp)

        doc_level_item['ent_lst'] = ent_lst
        doc_level_item['itp2atp_lst'] = itp2atp_lst
        doc_level_item_lst.append(doc_level_item)

    print('unmatch_count:', unmatch_count)
    print('match_count:', match_count)

    # ==保存文档集exm_lst
    doc_level_exm_lst = []
    for item in doc_level_item_lst:
        exm = NerExample.from_ent_lst(char_lst=list(item['text']), ent_lst=item['ent_lst'])
        # print(exm)
        exm.source_id = item['source_id']
        exm.itp2atp_lst = item['itp2atp_lst']
        doc_level_exm_lst.append(exm)
    NerExample.save_to_jsonl(doc_level_exm_lst, f'{out_file_dir}/doc_exm.jsonl', for_human_read=True, external_attrs=['itp2atp_lst', 'source_id'])

    # ==生成属性抽取mrc数据
    gen_doc_re(doc_level_exm_lst, out_file_dir=out_file_dir)

    # ==开始按句号分样本
    sent_level_exm_lst = []
    for doc_idx, doc_exm in enumerate(doc_level_exm_lst):
        split_exm_lst = NerExample.split_exm_by_deli(doc_exm)  # 单句分  # TODO
        # split_exm_lst = NerExample.split_exm_by_deli_multi_sent(dco_exm)  # 多句分
        for sent_idx, split_exm in enumerate(split_exm_lst):
            split_exm.source_id = f'{doc_exm.source_id}_{sent_idx}'
        sent_level_exm_lst.extend(split_exm_lst)

    # 按是否有实体来分
    # pos_exm_lst = [exm for exm in sent_level_exm_lst if not exm.is_neg()]
    # neg_exm_lst = [exm for exm in sent_level_exm_lst if exm.is_neg()]

    # 只按指标来分
    pos_exm_lst = [exm for exm in sent_level_exm_lst if exm.has_ent_type_startswith('指标')]
    neg_exm_lst = [exm for exm in sent_level_exm_lst if not exm.has_ent_type_startswith('指标')]

    print('句子级别样本信息:')
    print(f'num_total: {len(sent_level_exm_lst)}')
    print(f'num_pos: {len(pos_exm_lst)}')
    print(f'num_neg: {len(neg_exm_lst)}')
    print('负样本是正样本的几倍:', len(neg_exm_lst) / len(pos_exm_lst))

    dir_name = 'sent_mzb'  # multi 指标
    dir_name = 'sent_szb'  # single 指标
    NerExample.save_to_jsonl(sent_level_exm_lst, f'{out_file_dir}/sent_exm.jsonl', for_human_read=True, external_attrs=['source_id'])
    NerExample.save_to_jsonl(pos_exm_lst, f'{out_file_dir}/sent_exm_zb_pos.jsonl', for_human_read=True, external_attrs=['source_id'])
    NerExample.save_to_jsonl(neg_exm_lst, f'{out_file_dir}/sent_exm_zb_neg.jsonl', for_human_read=True, external_attrs=['source_id'])

    # NerExample.stats(sent_level_exm_lst)
    # NerExample.stats(pos_exm_lst)
    # NerExample.stats(neg_exm_lst)


def gen_doc_re(doc_level_exm_lst, out_file_dir):
    """ 生成文档及指标属性关系信息 """
    # doc_level_exm_lst = NerExample.load_from_jsonl('data/yanbao/sent/doc_exm.jsonl', external_attrs=['itp2atp_lst', 'source_id'])
    for exm in doc_level_exm_lst:
        ent_lst = exm.get_ent_lst(for_human=False)
        ent_ids_dct = {f'{ent_type}||{start}||{end}': edx for edx, (ent_type, start, end) in enumerate(ent_lst)}
        itp2atp_dct = {}
        for itp2atp in exm.itp2atp_lst:
            ent_type, start, end, *_ = itp2atp[0]
            itp_id = ent_ids_dct[f'{ent_type}||{start}||{end}']
            itp2atp_dct[itp_id] = []
            atps = itp2atp[1:]
            for ent_type, start, end, *_ in atps:
                atp_id = ent_ids_dct[f'{ent_type}||{start}||{end}']
                itp2atp_dct[itp_id].append(atp_id)

        exm.ent_dct_wid = copy.deepcopy(exm.ent_dct)  # ent_dct with id
        for ent_type, pos_lst in exm.ent_dct_wid.items():
            for pos in pos_lst:
                start, end = pos
                ent_id = ent_ids_dct[f'{ent_type}||{start}||{end}']
                pos.append(ent_id)

        exm.itp2atp_dct = itp2atp_dct
        exm.ent_ids_dct = ent_ids_dct
        exm.ent_dct = exm.ent_dct_wid
    NerExample.save_to_jsonl(doc_level_exm_lst, f'{out_file_dir}/doc_exm_w_re.jsonl', for_human_read=True, external_attrs=['itp2atp_dct', 'ent_ids_dct', 'source_id'])

    new_exm_lst = []
    miss_atp_lst = []
    for doc_level_exm in doc_level_exm_lst:
        itp2atp_dct = doc_level_exm.itp2atp_dct
        split_exm_lst = NerExample.split_exm_by_deli(doc_level_exm)  # 单句分
        # split_exm_lst = NerExample.split_exm_by_deli_multi_sent(doc_level_exm)  # 多句分  # TODO 是否要尝试多句的，更多上下文

        for exm in split_exm_lst:
            exm.truncate(max_size=512 - 2, direction='tail')  # TODO 这里截断是否合适

        for sid, exm in enumerate(split_exm_lst):
            miss_atp_count = 0
            ent_lst = exm.get_ent_lst()
            ent_ids_lst = [e[3] for e in ent_lst]

            curr_itp2atp_dct = {}
            for itp, atps in itp2atp_dct.items():
                if int(itp) in ent_ids_lst:
                    # new_atps = [atp if atp in ent_ids_lst else -1 for atp in atps ]
                    # miss_atp_count += sum(1 for e in new_atps if e == -1)
                    # curr_itp2atp_dct[itp] = new_atps
                    curr_itp2atp_dct[itp] = atps
            exm.itp2atp_dct = curr_itp2atp_dct

            miss_atp_lst.append(miss_atp_count)
            exm.source_id = f'{doc_level_exm.source_id}_{sid}'
        new_exm_lst.extend(split_exm_lst)
    # print(sorted(miss_atp_lst, reverse=True))
    print('整体指标属性关系缺失率:', np.mean(miss_atp_lst))
    NerExample.save_to_jsonl(new_exm_lst, f'{out_file_dir}/doc_exm_w_re_sent_multizb.jsonl', for_human_read=True, external_attrs=['itp2atp_dct', 'ent_ids_dct', 'source_id'])

    # 每个指标分1个样本。没有指标的样本不要  # TODO 后期没有指标的也要，增加些负样本
    single_zb_exm_lst = []
    for exm in new_exm_lst:
        for itp_id, atps_ids in exm.itp2atp_dct.items():
            # itp_id = int(itp_id)
            new_itp2atp_dct = {itp_id: atps_ids}
            new_exm = copy.deepcopy(exm)
            new_exm.itp2atp_dct = new_itp2atp_dct
            # 把无关指标和属性去除 保证每个样本只有1个指标及其对应的属性
            ent_lst = exm.get_ent_lst()
            exist_ids = [int(itp_id)] + atps_ids  # 存在的指标或属性id
            ent_lst = sorted([e for e in ent_lst if e[3] in exist_ids], reverse=True)  # 争取将指标排在属性前面
            new_exm.ent_dct = NerExample.ent_lst_to_ent_dct(ent_lst)
            new_exm.update(anchor='ent_dct')

            single_zb_exm_lst.append(new_exm)
    NerExample.save_to_jsonl(single_zb_exm_lst, f'{out_file_dir}/doc_exm_w_re_sent_singlezb.jsonl', for_human_read=True, external_attrs=['itp2atp_dct', 'ent_ids_dct', 'source_id'])


def anal_exm():
    indi_exm_lst = utils.load_jsonl('data/yanbao/exm/indi_exm_lst.jsonl')
    new_indi_exm_lst = []
    print('total', len(indi_exm_lst))
    lens = []
    min1024 = 0
    for exm in indi_exm_lst:
        lens.append(len(exm['ctx']))
        if len(exm['ctx']) < 1000:
            new_indi_exm_lst.append(exm)
        if len(exm['ctx']) < 1024:
            min1024 += 1
    print(min1024)

    print(len(new_indi_exm_lst))
    # utils.save_jsonl(new_indi_exm_lst, 'data/yanbao/exm/indi_exm_lst_maxlen1000.jsonl')

    print(utils.stats_lst(lens))


def convert2nerexm():
    itp2id = utils.file2list('data\yanbao\indi_lst.txt')
    itp2id = {itp: i for i, itp in enumerate(itp2id)}
    exm_lst = []
    inst_lst = utils.load_jsonl('data/yanbao/exm/indi_exm_lst_maxlen1000.jsonl')
    for inst in inst_lst:
        text = inst['ctx']
        itp = inst['indicator_name']
        ivl = inst['indicator_value']
        v, start, end = ivl
        ctx_start, ctx_end = inst['ctx_offset']
        start = int(start) - int(ctx_start)
        end = int(end) - int(ctx_start)
        assert (text[start:end]) == v
        ent_dct = {itp2id[itp]: [[start, end]]}
        exm = NerExample(list(text), ent_dct=ent_dct)
        exm.truncate(max_size=512, direction='tail')
        exm_lst.append(exm)

    print(len(exm_lst))
    exm_lst = NerExample.combine_by_text(exm_lst)
    print(len(exm_lst))
    NerExample.save_to_jsonl(exm_lst, 'data/yanbao/exm/indi_nerexm_lst_maxlen512.jsonl')


def process_data(domain, raw_json_data_dir, save_data_dir):
    itp_norm_fn = norm_indi_fn_normal()
    atp_norm_fn = None
    if domain == 'yimei':  # 医美的指标和属性需要进一步清洗
        itp_norm_fn = norm_indi_fn_yimei()
        atp_norm_fn = norm_atp_fn_yimei()

    files = list(Path(raw_json_data_dir).glob('*.json'))
    files = [[doc_id, file] for doc_id, file in enumerate(files)]  # doc_id, filename
    num_files = len(files)
    print(f'json文件总数:{num_files}')
    doc_id_2_pdf_name_map = {doc_id: Path(file).name for doc_id, file in files}
    utils.save_json(doc_id_2_pdf_name_map, f'{save_data_dir}/source_id_2_pdf_name_map.jsonl')

    # gen schema
    gen_schema0617(files, out_schema_file_dir=save_data_dir, itp_norm_fn=itp_norm_fn, atp_norm_fn=atp_norm_fn)

    # split train test
    random.seed(1234)
    doc_ids = [doc_id for doc_id, file in files]
    num_train = int(num_files * 0.9)
    num_test = num_files - num_train
    train_doc_ids = random.sample(doc_ids, num_train)
    train_files = [e for e in files if e[0] in train_doc_ids]
    test_files = [e for e in files if e[0] not in train_doc_ids]
    print(f'num_train: {num_train}')
    print(f'num_test: {num_test}')
    print(f'train_doc_ids: {train_doc_ids}')

    # convert
    print('\n====处理训练集=====')
    convert_data0617(train_files, out_file_dir=f'{save_data_dir}/train', itp_norm_fn=itp_norm_fn, atp_norm_fn=atp_norm_fn)
    print('\n====处理测试集=====')
    convert_data0617(test_files, out_file_dir=f'{save_data_dir}/test', itp_norm_fn=itp_norm_fn, atp_norm_fn=atp_norm_fn)

    exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("yb_data_converter")
    parser.add_argument('--domain', help='specify domain.', default='')
    args = parser.parse_args()

    domain = 'dianzi'
    domain = 'yiyao'
    domain = 'yimei'

    if args.domain:
        domain = args.domain

    process_data(
        domain=domain,
        raw_json_data_dir=f'data/yanbao/{domain}/raw_json_data',
        save_data_dir=f'data/yanbao/{domain}',
    )

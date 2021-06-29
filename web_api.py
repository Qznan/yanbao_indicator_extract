#!/usr/bin/env python
# coding=utf-8

from flask.globals import request

from load_model import PipeLine_Infer_1
import argparse
from flask import Flask
import json
import time
import os


app = Flask(__name__)
# gpu_args = argparse.ArgumentParser("gpu")
# gpu_args.add_argument("--gpu",help='select gpu to use',default = "0")

# args = gpu_args.parse_args()
# print(args)
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


dianzi_infer_args = argparse.ArgumentParser("dianzi").parse_args()
xuezhipin_infer_args = argparse.ArgumentParser("xuezhipin").parse_args()
yimei_infer_args = argparse.ArgumentParser("yimei").parse_args()
yiliaoqixie_infer_args = argparse.ArgumentParser("yiliaoqixie").parse_args()
yiyao_infer_args = argparse.ArgumentParser("yiyao").parse_args()


dianzi_infer_args.domain = "dianzi"
xuezhipin_infer_args.domain = "xuezhipin"
yimei_infer_args.domain = "yimei"
yiliaoqixie_infer_args.domain = "yiliaoqixie"
yiyao_infer_args.domain = "yiyao"

dianzi_pipeline_infer = PipeLine_Infer_1(dianzi_infer_args)
xuezhipin_pipeline_infer = PipeLine_Infer_1(xuezhipin_infer_args)
yimei_pipeline_infer = PipeLine_Infer_1(yimei_infer_args)
yiliaoqixie_pipeline_infer = PipeLine_Infer_1(yiliaoqixie_infer_args)
yiyao_pipeline_infer = PipeLine_Infer_1(yiyao_infer_args)

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/infer',methods=['POST',"GET"])
def infer():
    if not request.data:
        return "no document",400
    infer_args = argparse.ArgumentParser().parse_args()
    re_json = request.data
    re_json = json.loads(re_json)
    doc = re_json["doc"]
    domain = re_json["domain"]
    if domain not in ["xuezhipin","dianzi","yiliaozhipin","yimei","yiyao"]:
        return "no domain",400
    infer_args.domain = domain
    start_time = time.time()
    if domain == "dianzi":
        ret_json = dianzi_pipeline_infer.predict(doc)
    elif domain == "xuezhipin":
        ret_json = xuezhipin_pipeline_infer.predict(doc)
    elif domain == "yimei":
        ret_json = yimei_pipeline_infer.predict(doc)
    elif domain == "yiliaoqixie":
        ret_json = yiliaoqixie_pipeline_infer.predict(doc)
    elif domain == "yiyao":
        ret_json = yiyao_pipeline_infer.predict(doc)

    ret_json = json.loads(ret_json)
    ret_json["time"] = time.time() - start_time
    ret_json = json.dumps(ret_json, ensure_ascii=False, indent=2)
    # print("_____________")
    # for a in re_json.keys():
    #     print(a)
    # print("_____________")
    # print(ret_json)
    return ret_json,200



if __name__ == "__main__":
    # curl -i -k -H "Content-type: application/json" -X POST -d '{"domain":"yimei", "doc":"公司2020年研发费用、管理费用率以及销售费用率均呈上升态势：2020年，研发费用率由2019年的15.4%上升至17.7%，研发费用同比增长29.3%至2.3亿元；管理费用率由2019年的13.9%小幅上升至14.2%，管理费用同比增长14.8%至1.8亿元；销售费用率由2019年的40.8%上升至56.1%，销售费用同比增长54.4%至7.3亿元。"}' http://0.0.0.0:8080/infer
    # gpu_args = argparse.ArgumentParser("gpu")
    # gpu_args.add_argument("--gpu",help='select gpu to use',default = "0")

    # args = gpu_args.parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    app.run(host='0.0.0.0',port =6001)

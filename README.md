pipeline相关文件
- data/yanbao/
- span_ner.py
- span_ner_infer.py
- span_ner_train_yanbao.py
- yanbao_api_main_sample.py
- yb_data_converter.py
- data_reader.py
- datautils.py
- modules.py

依赖
torch = 1.8.0
transformers = 4.3.3

现有已训练好的电子(dianzi)和yiyao(医药)领域模型
剩下的医美、医疗器械、血制品可自己运行训练

step0: 运行前提
假设: 行业=医美 即domain=yimei(行业英文名自取)

step1: 准备数据
新建目录并将相应领域的所有研报json文件放入data/yanbao/yimei/raw_json_data目录下

step2: 处理数据
python yb_data_converter.py --domain yimei

step3: 训练数据
python span_ner_train_yanbao.py --domain yimei --model cls  # 命令1_1
python span_ner_train_yanbao.py --domain yimei --model pred_cls  # 命令1_2
python span_ner_train_yanbao.py --domain yimei --model span  # 命令1_3
python span_ner_train_yanbao.py --domain yimei --model multimrc  # 命令2_1

说明:
命令1_1 1_2 1_3 需要串行运行(运行完一个才能运行下一个)
命令2_1可并行运行 即在开始运行1_1时可另开窗口运行2_1
可通过 --gpu 0 指定GPU 使用CPU则是 --gpu -1

step4: 预测接口
参考yanbao_api_main_sample.py __main__

输入为一篇研报的docment str, 输出结果为json, 结构如下:
{
    text:{指标1 = value: [属性1 = value, 属性2 = value, ...], 指标2 = value: [属性1 = value, 属性2 = value, ...], ...,
    text:{指标1 = value: [属性1 = value, 属性2 = value, ...], 指标2 = value: [属性1 = value, 属性2 = value, ...], ...,
}

具体示例:
{
  "国内市场方面，BCI高频数据显示手机需求持续走低，4月末第17周（4.19-4.25）国内手机销量462.6万台，同比下降21.8%。": {
    "销量 = 462.6万台": [
      "产品 = 手机",
      "区域 = 国内",
      "时间 = 4月末第17周"
    ],
    "销量同比 = 下降21.8%": [
      "区域 = 国内",
      "时间 = 4月末第17周"
    ]
  },
  "（日经中文网）全球前五大智能手机华为Q1首度落榜根据市调机构IDC初步资料显示，第一季全球智能型手机出货量超过3.45亿支，比去年同期激增25.5％，中国与亚太地区是最大成长动能来源，而至第一季底实体清单禁令已施行半年，华为首度跌出全球前五大手机品牌之列，市占率遭到竞争对手vivo、OPPO与小米分食，三星则在新机效应下重回全球智能型手机出货量冠军。": {
    "出货量 = 3.45亿支": [
      "产品 = 智能型手机",
      "区域 = 全球",
      "时间 = 第一季",
      "来源 = 市调机构IDC",
      "行业 = 智能型手机"
    ],
    "出货量同比 = 25.5％": [
      "产品 = 智能型手机",
      "区域 = 全球",
      "时间 = 第一季",
      "来源 = 市调机构IDC"
    ]
  },
  "（中时新闻网）Counterpoint预测：2021年全球TWS市场同比增长33％据Counterpoint全球可穿戴设备（TWS）市场预测2021-2023，全球TWS（真无线立体声耳机）市场预计到2021年将同比增长33％，达到3.1亿部。": {
    "市场规模同比 = 33％": [
      "产品 = TWS",
      "区域 = 全球",
      "时间 = 2021年",
      "来源 = Counterpoint"
    ],
    "出货量 = 3.1亿部": [
      "区域 = 全球",
      "时间 = 2021年",
      "来源 = Counterpoint"
    ]
  },
  "根据Counterpoint全球可穿戴设备（TWS）2020年第四季度的市场追踪数据显示，尽管受到疫情引发的经济下滑影响，但2020年仍TWS同比增长了78％。": {
    "市场规模同比 = 78％": [
      "区域 = 全球",
      "时间 = 2020年",
      "来源 = Counterpoint全球可穿戴设备（TWS）"
    ]
  },
}

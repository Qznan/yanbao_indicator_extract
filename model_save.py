from transformers import AutoModel
import torch


model_predictor_cls = torch.jit.load("./model1/yiyao_predictor_cls.pth")
# model_predictor_multimrc = torch.jit.load("./model1/dianzi_predictor_multimrc.pth")
# model_predictor_span = torch.jit.load("./model1/dianzi_predictor_span.pth")




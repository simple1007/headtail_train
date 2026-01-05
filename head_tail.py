from enum import auto
import torch
import os
from unicode import join_jamos
from headtail import normalize,removesym, infer, HTInputTokenizer,get_xstags
# from headtail import infer_onnx as infer
from customutils import AC,original_pos
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["MKL_NUM_THREADS"] = "8"
# torch.set_num_threads(8)
root = __file__

root = root.split(os.sep)
root = os.sep.join(root[:-1])
print(root)
os.environ['HT'] = root + os.sep + "postagger_model"

import tokenizer_ht as tokenizer_ht
from tokenizer_model import TK_Model_Mini2 as TK_Model2

import sys
sys.setrecursionlimit(100000)
sys.path.append(root + os.sep + "postagger_model")
from postagger_model import PosLSTMMini2 as PosLSTM
tok_max_len = 230

import numpy as np
import transformers
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

# import tagging_lstm as tagging
import json
import re
import time
import numpy as np
import threading, requests, time
import threading
from multiprocessing import Pool, Process
import time
import torch
from torch.quantization import QuantStub, DeQuantStub, float_qparams_weight_only_qconfig, default_qconfig
import argparse 
import pickle
import torch.nn as nn
from transformers import BertModel, DistilBertModel#, AdamW
from pos_vocab import BiPosVocab
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
# import numpy as np
# from train_tkbigram_pos_tagger_pytorch_peft_bi import PosTaggerModelPeftTorch
# with open(root + os.sep + "postagger_model" + os.sep + "posvocab.pkl","rb") as f:
#     posvocab = pickle.load(f)postagger_modu_mini3
with open(root + os.sep + "postagger_model" + os.sep + "/ht_postagger_model2/subwordvocab.pkl","rb") as f:
    posvocab = pickle.load(f)

start_t = time.time()
# print(os.sep,root)
with open(root + os.sep + "tokenizer_model" + os.sep + 'lstm_vocab.pkl','rb') as f:
    lstm_vocab = pickle.load(f)
# with open(root + os.sep + "tokenizer_model" + os.sep + 'bigram_vocab.pkl','rb') as f:
#     bigram = pickle.load(f)

from transformer import Encoder, TransformerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

# with open(root + os.sep + "postagger_model" + os.sep + 'kcc150_all_tag_dict.pkl','rb') as f:
#     tag_dict = pickle.load(f)
#     tag_dict = { v:k for k,v in tag_dict.items() }
# with open(root + os.sep + "postagger_model" + os.sep + 'kcc150_all_tokenbi.pkl','rb') as f:
#     tkbigram = pickle.load(f)

tag_len = len(posvocab.index2pos)

parser = argparse.ArgumentParser(description="Postagger")

parser.add_argument("--MAX_LEN",type=int,help="MAX Sequnce Length",default=230)
parser.add_argument("--BATCH",type=int,help="BATCH Size",default=50)
parser.add_argument("--EPOCH",type=int,help="EPOCH Size",default=5)
parser.add_argument("--epoch_step",type=int,help="Train Data Epoch Step",default=4000)
parser.add_argument("--validation_step",type=int,help="Validation Data Epoch Step",default=240)
parser.add_argument("--hidden_state",type=int,help="BiLstm Hidden State",default=tag_len*2)
parser.add_argument("--GPU_NUM",type=str,help="Train GPU NUM",default="0")
parser.add_argument("--model_name",type=str,help="Tokenizer Model Name",default="tkbigram_one_first_alltag_bert_tagger.model")
parser.add_argument("--input",type=str,help="input_file",default="false")
parser.add_argument("--output",type=str,help="output_file",default="false")

args = parser.parse_args()
EPOCH = args.EPOCH
max_len = args.MAX_LEN

import torch.quantization

print('Model 로딩 중입니다. 모델 로딩까지 2분이상 소요 될 수 있습니다.')

tok_model = None
pos_model = None
# mod = None
tok_model2 = None
tok_model3 = None
pos_model2 = None
pos_model3 = None

def tokmodel():
    global tok_model, tok_model2, tok_model3
    tok_model = TK_Model2(max_len,lstm_vocab)
    tok_model.load_state_dict(torch.load(root+os.sep + "tokenizer_model"+'/ht_tokenizer_model2/model',weights_only=True,map_location=device))
    tok_model = tok_model.to(dtype=torch.bfloat16,device=device)
    tok_model.eval()
    
    # # for name,param in tok_model.named_parameters():
    # #     print(f"{name}: {param.dtype}")
    # # exit()
    # # # model.eval()

    # # # 4. 더미 입력 생성 (모델 입력 크기와 일치해야 함)
    # # dummy_input = torch.randn(1, 3, 224, 224)
    # tok_model = torch.jit.script(tok_model)
    # exit()
    # dummy_input = torch.randint(0,len(lstm_vocab),(100,120),requires_grad=True,dtype=torch.float32).to("cuda")#.to(device)
    # # torch_out = tok_model(x)

    # # 5. ONNX로 내보내기
    # # with torch.autocast(device_type="cuda",dtype=torch.float16):
    # torch.onnx.export(
    #     tok_model,               # 내보낼 모델
    #     dummy_input,         # 모델 입력 예시
    #     "tok_model.onnx",        # 저장할 파일 이름
    #     export_params=True,  # 모델 가중치 저장 여부
    #     opset_version=17,    # ONNX 버전
    #     do_constant_folding=True,  # 상수 폴딩 최적화 적용 여부
    #     input_names=['input'],     # 입력 레이어 이름
    #     output_names=['output'],   # 출력 레이어 이름
    #     dynamic_axes={
    #         'input': {0: 'batch_size'},   # 가변적인 배치 크기
    #         'output': {0: 'batch_size'}
    #     }
    # )

    # # print("모델이 성공적으로 ONNX 형식으로 변환되었습니다.")
    
    # # # print("모델이 성공적으로 ONNX 형식으로 변환되었습니다.")
    # import onnx

    # # # ONNX 모델 불러오기
    # tok_model = onnx.load("tok_model.onnx")

    # # # 모델 검증
    # onnx.checker.check_model(tok_model)


    # output = "tok_model.onnx" # onnx 모델 위치
    # converter = "tok-model-uint8.onnx" # 저장할 모델 위치

    # quantize_dynamic(output, converter, weight_type=QuantType.QUInt8)
    # # # # GPU 세션 생성
 
    # # # print("ONNX 모델이 성공적으로 검증되었습니다.")
    
    # # # exit()
    # from onnxconverter_common import float16, auto_mixed_precision
    # # # dummy_input = torch.randint(0,len(lstm_vocab),(100,230),requires_grad=True).numpy()#.to("cuda")#.to(device)
    
    # # tok_model = float16.convert_float_to_float16(tok_model)#auto_mixed_precision.auto_convert_mixed_precision(tok_model,{"input":dummy_input.detach().cpu().numpy()},rtol=0.01, atol=0.001, keep_io_types=True)#float16.convert_float_to_float16(tok_model)
    # # onnx.save(tok_model,"tok_model_fp16.onnx")
    # sess_opt = ort.SessionOptions()
    # sess_opt.intra_op_num_threads = 8
    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    # tok_model = ort.InferenceSession(converter, providers=providers,sess_options=sess_opt)

    # # exit()
    # def exportonnx():
    #     x = torch.randint(0,len(lstm_vocab),(1,230),requires_grad=True,dtype=torch.float).to("cuda")#.to(device)
    #     torch_out = tok_model(x)

    #     torch.onnx.export(tok_model,               # 실행될 모델
    #         x,                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
    #         "super_resolution.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
    #         export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
    #         opset_version=12,          # 모델을 변환할 때 사용할 ONNX 버전
    #         do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
    #         input_names = ['input'],   # 모델의 입력값을 가리키는 이름
    #         output_names = ['output'], # 모델의 출력값을 가리키는 이름
    #         dynamic_axes={'input' : {0 : 'batch_size'},    # 가변적인 길이를 가진 차원
    #                     'output' : {0 : 'batch_size'}})
    #     import onnx
        
    #     onnx.save(onnx.shape_inference.infer_shapes(onnx.load("super_resolution.onnx")), "super_resolution.onnx")
    #     onnx_model = onnx.load("super_resolution.onnx")
        
    #     onnx.checker.check_model(onnx_model)
    #     # print(onnx_model(x))
    #     # exit()
    #     import onnxruntime
    #     x = torch.randint(0,len(lstm_vocab)+1,(1,230),dtype=torch.float).to("cpu")#.to(device)
    #     # print()
    #     ort_session = onnxruntime.InferenceSession("super_resolution.onnx")
    #     # torch_out = torch_out.detach().numpy()
    #     def to_numpy(tensor):
    #         return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    #     x = x.detach().cpu().numpy()
    #     torch_out = torch_out[0].detach().cpu().numpy()
    #     # print(torch_out)
    #     # ONNX 런타임에서 계산된 결과값
    #     ort_inputs = {ort_session.get_inputs()[0].name: x}
    #     ort_outs = ort_session.run(None, ort_inputs)

    #     # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
    #     np.testing.assert_allclose(torch_out, ort_outs, rtol=1e-03, atol=1e-05)
    #     # print(len(ort_outs))
    #     print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    #     exit()
    # # tok_model = torch.jit.script(tok_model)
    # # torch.jit.save(script_model, "mobilenet2.pt")
    # # tok_model.eval()

def posmodel():
    global pos_model,pos_model2,pos_model3
    
    pos_model = PosLSTM(posvocab)
    pos_model.load_state_dict(torch.load(root + os.sep + "postagger_model" + os.sep + "ht_postagger_model2/model",weights_only=True,map_location=device)["model"])
    pos_model = pos_model.to(dtype=torch.bfloat16,device=device)
    pos_model.eval()
    
    # dummy_input = torch.randint(0,15000,(100,120),requires_grad=True,dtype=torch.float).to("cuda")#.to(device)
    # # # torch_out = tok_model(x)

    # # # 5. ONNX로 내보내기
    # with torch.autocast(device_type="cuda",dtype=torch.float16):
    # torch.onnx.export(
    #     pos_model,               # 내보낼 모델
    #     dummy_input,         # 모델 입력 예시
    #     "pos_model.onnx",        # 저장할 파일 이름
    #     export_params=True,  # 모델 가중치 저장 여부
    #     opset_version=17,    # ONNX 버전
    #     do_constant_folding=True,  # 상수 폴딩 최적화 적용 여부
    #     input_names=['input'],     # 입력 레이어 이름
    #     output_names=['output'],   # 출력 레이어 이름
    #     dynamic_axes={
    #         'input': {0: 'batch_size'},   # 가변적인 배치 크기
    #         'output': {0: 'batch_size'}
    #     }
    # )

    # print("모델이 성공적으로 ONNX 형식으로 변환되었습니다.")
    
    # # print("모델이 성공적으로 ONNX 형식으로 변환되었습니다.")
    # import onnx

    # # # ONNX 모델 불러오기
    # pos_model = onnx.load("pos_model.onnx")

    # # # 모델 검증
    # onnx.checker.check_model(pos_model)


    # output = "pos_model.onnx" # onnx 모델 위치
    # converter = "pos-model-uint8.onnx" # 저장할 모델 위치

    # quantize_dynamic(output, converter, weight_type=QuantType.QUInt8)
    # from onnxconverter_common import float16, auto_mixed_precision
    # # # dummy_input = torch.randint(0,len(lstm_vocab),(100,230),requires_grad=True).numpy()#.to("cuda")#.to(device)
    
    # # pos_model = float16.convert_float_to_float16(pos_model)#auto_mixed_precision.auto_convert_mixed_precision(tok_model,{"input":dummy_input.detach().cpu().numpy()},rtol=0.01, atol=0.001, keep_io_types=True)#float16.convert_float_to_float16(tok_model)
    # # onnx.save(pos_model,"pos_model_fp16.onnx")
    # sess_opt = ort.SessionOptions()
    # sess_opt.intra_op_num_threads = 8
    # providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    # pos_model = ort.InferenceSession(converter, providers=providers,sess_options=sess_opt)

    # # # GPU 세션 생성
    # providers = ['CUDAExecutionProvider']#, 'CPUExecutionProvider']
    # pos_model = ort.InferenceSession(output, providers=providers)

    # print("ONNX 모델이 성공적으로 검증되었습니다.")

tokmodel()
print("tokenizer model load compelte")
posmodel()
print("pos model load compelte")

end_t = time.time()
print("모델 로딩 {}".format((end_t-start_t)))

def get_tok(line,tok_model,device,verbose=0):
    # print(device)
    tok = tokenizer_ht.predict2(tok_model,line,device,verbose=verbose)
    # tok = tokenizer_ht.predict_onnx(tok_model,line,device,verbose=verbose)
    
    # result['1'] = tok
    return tok

def get_pos(tok,pos_model,verbose=0):
    # print(tok)
    pos = infer(posvocab,pos_model,tok,device)
    
    return pos
    postmp = []
    for pos_ in pos:
        sent = []
        for pos__ in pos_.split():
            headmp = get_xstags(pos__)
            sent.append(headmp)
        postmp.append(" ".join(sent))
    # result['pos'] = pos
    pos = postmp
    return pos

thread_num = 1
batch = 128
# models = [[tok_model,tok_model2,tok_model3],[pos_model,pos_model2,pos_model3]]
from tqdm import tqdm    
from cnoun import cnoun,one_syl,iscnoun
from konlpy.tag import Komoran
k = Komoran()
# n_tags = ["합","봤","본","보","본","냐","될","됄","한","했","하","해","되","됐","됬",'한','할','될','시키','시켜','시켰','시킬','시킨','된','스러',"스럽","스런","드리","드렸","돼","됄"]
n_tags = ["웠", "워","같","함","합","될","됄","한","했","하","해","되","됐","됬",'한','할','될','시키','시켜','시켰','시킬','시킨','된','스러',"스럽","스런","드리","드렸","돼","됄"]
# n_tags = []
class Node(object):
    def __init__(self, key, data=None):
        self.key = key
        self.data = data
        self.children = {}

class Trie:
    def __init__(self):
        self.head = Node(None)

    def insert(self, string):
        current_node = self.head

        for char in string:
            if char not in current_node.children:
                current_node.children[char] = Node(char)
            current_node = current_node.children[char]
        current_node.data = string

    def search(self, string):
        current_node = self.head

        for char in string:
            if char in current_node.children:
                current_node = current_node.children[char]
            else:
                return False

        if current_node.data:
            return True
        else:
            return False

    def starts_with(self, prefix):
        current_node = self.head
        words = []

        for p in prefix:
            if p in current_node.children:
                current_node = current_node.children[p]
            else:
                return None

        current_node = [current_node]
        next_node = []
        while True:
            for node in current_node:
                if node.data:
                    words.append(node.data)
                next_node.extend(list(node.children.values()))
            if len(next_node) != 0:
                current_node = next_node
                next_node = []
            else:
                break

        return words

import pandas as pd
from jamo import h2j,j2hcj
from collections import defaultdict

head_dict = pd.read_csv("head_ori_dict.csv",encoding="utf-8-sig")
tail_dict = pd.read_csv("tail_ori_dict.csv",encoding="utf-8-sig")

def to_dict(dic_,mode="head"):
    dict_ = defaultdict(set)
    dic = dic_.loc[:,["token","res"]].values.tolist()
    for t,r in dic:
        # print(t,r)
        if mode=="tail":
            t = t.split("@")
            # if len(t)
            if len(t) > 1:
                # print(j2hcj(h2j(t[0][0])),j2hcj(h2j(r[0])))
                # exit()
                t1 = t[1].split("_")
                if len(t1) > 1:
                    if "-" in t1[0]:
                        t1[0] = t1[0].split("-")[0]
                    if "-" in t1[-1]:
                        t1[-1] = t1[-1].split("-")[0]
                    t1 = t1[0]+"_"+t1[-1]
                    t = t[0] + "@" + t1
                else:    
                    t = "@".join(t)
            else:
                t = "@".join(t)
        # print(t,r)
        # exit()
        if h2j(j2hcj(t[0]))[0] == h2j(j2hcj(r[0]))[0]:
            # print(t,r)
            dict_[t].add(r)
    
    return dict_

head_dict = to_dict(head_dict)
tail_dict = to_dict(tail_dict,mode="tail")
        
import pickle
if True:
    with open("heads.dict","rb") as f:
        heads_list = pickle.load(f)
        tmp = Trie()
        for h in heads_list:
            h.strip()
            jamo = j2hcj(h2j(h))
            tmp.insert(jamo)
        heads_list = tmp

    with open("general.dict","wb") as f:
        pickle.dump(heads_list,f)
        

with open("general.dict","rb") as f:
    heads_list = pickle.load(f)

def line_normalie(x):
    if type(x) == str:
        xxx = [x]
    else:
        xxx = x    
    X = []
    for idx,xx in enumerate(xxx):
        dot_flag = False
        xx = xx.strip()
        xx = xx.replace("+","_")
        xx = xx.replace("/","_")
        x_ = normalize(xx)
        # if dot_flag and len(xx) >= tok_max_len:
        #     xx = xx[:tok_max_len-2]
        #     xx = normalize(xx)
            
        # else:
        xx = x_
        xx = removesym(xx)
        xx = re.sub(r"♥"," ",xx)#xx.replace("♥"," ")
        xx = re.sub(r" +"," ",xx).strip()
        if xx == "":
            xx = "NoChar"
        if xx == "END":
            xx = "end"
        if xx == "STAT":
            xx = "stat"
        if xx == "ST":
            xx = "st"
        xx = max_tok_remove(xx)
        # print(xx)
        if xx == "end":
            print(x_,xx)
        for xxx in xx:
            X.append(xxx)
    return X

def max_tok_remove(xx):
    # print(len(xx))
    if tok_max_len < len(xx):
        xx = xx.split()
        tmplength = 0
        tmpsent = ""
        sents = []
        # print(1)
        prev =""
        for idx,xxx in enumerate(xx):
            # length = len(xxx) + 1
            # tmplength += len(xxx)
            tmpsent += " "+xxx
            tmpsent = re.sub(r" +"," ",tmpsent).strip()
            tmpsent = tmpsent.strip()
            tmplength = len(tmpsent)
            if tmplength >= tok_max_len:
                if prev == "":
                    sents.append("ST▁"+tmpsent.strip().replace(' ','▁').strip('▁'))
                else:
                    prev_ = prev.split()[-4:]
                    prev_ = " ".join(prev_)
                    sents.append(prev_.strip().replace(' ','▁').strip('▁')+"▁ST▁"+tmpsent.strip().replace(' ','▁').strip('▁'))
                # print(len(sents[-1]))                
                prev = ""
                tmpsent = ""
                tmplength = 0
            prev = tmpsent
                # print(tmpsent)
            
            # else:
                # tmpsent += " "+xxx
                # tmplength = len(tmpsent)
        if tmpsent.strip() != "":
            if prev != "":
                prev_ = prev.split()[-4:]
                prev_ = " ".join(prev_)
                sents.append(prev_.strip().replace(' ','▁').strip('▁') + "▁ST▁"+tmpsent.strip().replace(' ','▁').strip('▁'))
            else:
                sents.append("ST▁"+tmpsent.strip().replace(' ','▁').strip('▁'))
        
        # print(sents)
        # if len(sents) > 0:
        return ["STAT"] + sents + ["END"]
        # else:
        #     return [""]
        # # else:
        #     print(3,xx)
        # xx = xx[:tok_max_len]
        # xx = xx.strip()
    return [xx.replace(' ','▁').strip('▁')]
        # X.append(xx.replace(' ','▁').strip('▁'))

def inference_tok(X):        
    result = [{}]
    tok = get_tok(X,tok_model,device)
    tok_ = []
    for tk in tok:
        tk = re.sub(" +"," ",tk)
        tok_.append(tk)
    
    return tok

def inference_pos(tok):
    pos = get_pos(tok,pos_model,device)
    # pos = result[0]['pos']
    return pos

def original_token(pos):
    # print(ll)
    # print(j2hcj(h2j("주")))
    # print(pos)
    # exit()
    pos_ori = []
    lcnt = 0
    # print(pos)
    sent = ""
    longsentflag = False
    postmp = []
    # print(pos)
    flag = False
    
    stflag = False
    # for p in pos:
        
    #     # print(p)
    #     p = p.strip()
    #     pps = p.split()
    #     # ptmp = ""
    #     # if len(pps) == 1:
    #     #     for ppss in pps[0].split("+"):
    #     #         ptmp += ppss.split("/")[0]
        
    #     tmpp = []
    #     # print(pps)
    #     line_tmp = []
    #     for pps_ in pps:
    #         tmp = []
    #         for pps__ in pps_.split("+"):
    #             tmp.append(pps__.split("/")[0])
    #         line_tmp.append("".join(tmp))
    #     line_tmp = " ".join(line_tmp)
    #     line_tmp = line_tmp.strip()
    #     if " ST " in line_tmp or line_tmp.strip().startswith("ST "):
    #         # print(line_tmp)
    #         for pps_ in pps:
    #             # print(1,line_tmp)
    #             # print(2,
    #             # pps_)
    #         # for pps_ in pps[0].split("+"):
    #             # ptmp += ppss.split("/")[0]
    #             # print(pps_)
    #             if pps_.startswith("ST/") or ("S/" in pps_ and "+T/" in pps_): #pps_.startswith("S/SL+T/SL"):
    #                 stflag = True
    #             if stflag:
    #                 tmpp.append(pps_)
    #         # print(1,tmpp)
    #         pps = tmpp
    #         # print(3,pps)
    #         p = " ".join(pps)
    #         stflag = False
    #         # print(4,p)
    #         # print(p)
    #     postmp.append(p)
    #     # if ptmp != "STAT" and ptmp != "END" and not p.startswith("ST/") and (p.strip() != "" and "S/" not in p.split()[0] and "+T/" not in p.split()[0]) :
    #     #     postmp.append(p)
    #     #     flag = False
    #         # import sys
    #         # sys.stdout.write(str(1)+": "+line_tmp+"\n")
    #         # continue
        
    #         # if ptmp != "STAT" and ptmp != "END":
    #             # print(ptmp) 
    #     # flag = 
    #     # if ptmp == "STAT":
    #     #     # continue
    #     #     flag = True
    #     # elif flag and (p.startswith("ST/") or (p.strip() != "" and "S/" in p.split()[0] and "+T/" in p.split()[0])):
    #     #     senttmp = []
    #     #     longsentflag = True
    #     #     # for pp in pos:
    #     #     pp = " ".join(p.split()[1:])
    #     #     # senttmp.append(pp)
    #     #     # print(1,pp)
    #     #     sent = sent + " " + pp#.join(senttmp).strip()
    #     #     # print(sent)
    #     # elif flag and ptmp == "END":#.startswith("END/"):
    #     #     postmp.append(sent)
    #     #     sent = ""
    #     #     flag = False
            
    #     # elif not flag:
    #     #     postmp.append(" ".join(pps))
    # pos = postmp
    # print(len(pos))
    
    # print(pos)
    for idx1,p in enumerate(pos):
        # print(p)
        # exit()
        line = []
        p = re.sub(r" +"," ",p).strip()
        # p = p.replace(" + ","+").replace(" +","+").replace("+ ","+")
        p = p.strip().strip("+").strip()
        maxl = ""
        sent = p

            # maxl += " "+p#p.split()[1:]
            # lcnt += 1
            # if p.startswith("ST/") and lcnt != len(pos):
            #     continue
            # elif p.startswith("ST/") and lcnt == len(pos):
            #     sent = maxl
            #     maxl = ""
        sent = re.sub(r" +"," ",sent).strip()
        for l in sent.split():
            # sent = l
            
            tk = []
            for idx,ll in enumerate(l.split("+")):
                # print(ll)
                if idx == 0 and "/V" in ll:# or (idx ==1 and ("/VV" in ll or "/VA" in ll)):
                    ll_ = ll.split("/")
                    # print(ll_,ll)
                    try:    
                        headlast_char = ll_[0][-1]
                    except:
                        exit()
                    headtag = ll_[1]
                    
                    if "+" in l and idx == 0:
                        try:
                            tail_t = l.split("+")[1].split("/")[1]
                            res = head_dict[f"{headlast_char}/{headtag}@{tail_t}"]#.query(f"token==\"{ll_[0][-1]}/{ll_[1]}\"")
                        except:
                            exit()
                    else:
                        try:
                            res = head_dict[f"{headlast_char}/{headtag}"]
                        except Exception as ex:
                            print(ex)
                            exit()
                        
                    heads = ll
                    if len(res) > 0:
                        prev = 0
                        heads_last = ""
                        orijamo = j2hcj(h2j(headlast_char))
                        for r in res:
                            tktmp = r[0]
                            # print(tktmp)
                            # lheadtmp = j2hcj(h2j(ll[0][-1]))
                            # lht_flag = False
                            # if len(lheadtmp) == 3:
                                # print(lheadtmp)
                                # lheadtmp = lheadtmp[-1]
                                # lht_flag = True
                                # print(lheadtmp)
                            
                            head_tmp = ll_[0][:-1]+tktmp#+(lheadtmp if lht_flag else "")
                            jamo = j2hcj(h2j(head_tmp.split("/")[0]))
                                
                            if heads_list.search(jamo):
                                heads = head_tmp#+head_tmp.split("/")[1]
                                
                                cnt = 0
                                for ori,cmp in zip(orijamo,jamo):
                                    if ori == cmp:
                                        cnt += 1
                                if cnt >= prev:
                                    heads_last = heads        
                                
                                prev = cnt
                        
                        if heads_last != "":
                            if "+" in l:
                                l_ = l.split("+")[1].split("/")[1]
                                if len(heads_last) >= 2 and "EP_" in l_:
                                    cjj = j2hcj(h2j(heads_last[-2]))
                                    heads_last = list(heads_last)
                                    if len(cjj) == 3:
                                        heads_last[-2] = join_jamos(cjj[:2])
                                    heads_last = "".join(heads_last)
                            # print(ll_)
                            heads = heads_last +"/" + ll_[1]
                    
                        tk.append(heads)
                    else:
                        tk.append(heads)
                elif idx == 0:
                    tk.append(ll)
                elif idx == 1 and "/V" in ll:
                    # print(ll)
                    if "_" not in ll.split("/")[1]:# or len(j2hcj(h2j(ll.split("/")[0][0]))) < 3:
                        # print(ll)
                        tk.append(ll)
                        continue
                    head_t = l.split("+")[0].split("/")[1]
                    ll = ll.replace("/",f"/{head_t}@")
                    
                    res = tail_dict[f"{ll}"]
                    # print(res)
                    tagsep = "@@@"
                    # print(1,res)
                    # print(t,r)
                    if len(res) > 0:
                        lasteum = j2hcj(h2j(ll.split("/")[0][0]))
                        maxsim = -1
                        reseum = ""
                        for r in list(res):
                            r = r.split()[0].split(tagsep)
                            rr = j2hcj(h2j(r[0]))
                            score = 0
                            for candieum,leum in zip(rr,lasteum):
                                if leum == candieum:
                                    score += 1
                            
                            if score > maxsim:
                                # print(r)
                                maxsim = score
                                reseum = r[0] + "/" + r[1]
                                # print(3,reseum)
                            # for rr in r.split():
                                
                            
                        #     # r = j2hcj()
                        # res = list(res)[-1]
                        # res_mp = [tmp.split(tagsep)[0] for tmp in res.split()]
                        # res_tag = [tmp.split(tagsep)[1] for tmp in res.split()]
                        if reseum != "":
                            tk.append(reseum)
                        # print(tk)
                    # else:
                    #     ll = ll.split("@")
                    #     ll[0] = ll[0].split("/")[0]
                    #     ll = "/".join(ll)
                    #     tk.append(ll)
            line.append("+".join(tk))
        pos_ori.append(" ".join(line))
        # if longsentflag:
        #     print(pos_ori)
        #     break
    pos = pos_ori
    return pos

def analysis(x,verbose=0):
    # print(x)
    X = line_normalie(x)
    # print(X)
    # exit()
    # print(len(X[0]))
    tok = inference_tok(X)
    # print(tok)
    
    pos = inference_pos(tok)
    # print(pos)
    # pos = original_pos(pos)
    # print("------------------------------------")
    x_ = []
    # p_ = []
    prev = ""
    indexs = []
    tmp = []
    s = 0
    e = 0
    splitflag = False
    resX = []
    import copy
    xtmp = copy.deepcopy(x)
    from collections import deque
    # x = deque(x)
    for idx,xx in enumerate(X):
        xx = xx.replace("▁"," ").strip()
        if xx == "STAT":
            indexs.append(pos[idx])
            # tmp.append(idx)
            splitflag =True
            # x.popleft()
            s += 1
            # print(0,idx)
        elif xx == "END":
            # tmp.append(idx)
            indexs.append(pos[idx])
            tmp = []
            resX.append(" ".join(indexs[1:-1]))
            # x.popleft()
            # x_.append(x.popleft())
            # print("END"," ".join(indexs[1:-1]))
            indexs = []
            splitflag = False
            e += 1
        elif splitflag:
            restok = pos[idx].strip().split()
            xstchk = xx.split()#removesym(normalize(xx.strip())).split()

            stchk = -1
            for stidx, xxx in enumerate(xstchk):
                if xxx.strip() == "ST":
                    stchk = stidx + 1
            if stchk > -1:
                # print("1$"+restok[stchk-1]+"$2","3$"+restok[stchk]+"$4")
                restok = restok[stchk:]
                
            restok = " ".join(restok)
            # print(restok)
            indexs.append(restok)
        elif not splitflag:
            resX.append(pos[idx])
            # print("NOT",pos[idx])
            splitflag = False
            # x_.append(x.popleft())
            # print(1,idx)
    # if type(x) == str:
    # print(x)
    # exit()
    for xx in x:
        if xx != "STAT" and xx != "END":
            x_.append(xx)
            # prev = xx
    # print(len(resX))
    # exit()
    # print(e,s)
    # print(pos)
    # pos_ = []
    # start = 0
    # for pp in indexs:
        
    #     if pp[0] != start:
    #         # print(start,pp)    
    #         pos_ = pos_ + pos[start:pp[0]+1][:-1]
    #         # print(pos[start:pp[0]+1][:-1])
    #         # print("a",start)
    #         # print("a",start,len(pos[start:pp[0]+1]))
    #         # start = start + len(pos[start:pp[0]+1])
            
        
    #     pos_ = pos_ + [" ".join(pos[pp[0]:pp[1]+1][1:-1])]
    #     print(len(pos[pp[0]:pp[1]+1]),pp)
    #     # print(pos[pp[0]:pp[1]+1][:-1])
    #     # start = start + len(pos[pp[0]:pp[1]+1]) + 1
    #     start = pp[1] + 1
    #     # print(pos[start])
        # print("b",start,len(pos[pp[0]:pp[1]+1]))
    # print(start)
    # if start < len(pos):# and len(indexs) > 0:
    #     # print(start,len(pos_),len(pos[start:]),pos[start])
    #     pos_ = pos_ + pos[start:]
        # print(pos[start:])
        # len(pos_)
        
    
    # exit()
    # print(len(x_))
    # print(len(resX))
    # exit()
    # if len(x_) != len(resX):
        # print(e,s)
        # print(x_,resX)
        # print(len(pos),len(resX),len(X),len(xtmp),len(x_))
        # # exit()
        # for xx,pp in zip(x_,resX):
            # print(1,xx)
            # print(2,pp)
        # print(pos)
        # for xx,rr in zip(x_,resX):
        #     print(1,xx)
        #     print(2,rr)
        # for xx in X:
            # print(1,x[:5])
            
            # print(xx.replace("▁"," "))
        # for xx in x:
        #     xx = removesym(normalize(xx)).strip()
        #     # if len(xx) > tok_max_len:
        #         # print(xx)
        #     xx = xx.replace("♥"," ")
        #     xx = re.sub(r" +"," ",xx).strip()
        #         # print(xx)
        #     for xxx in max_tok_remove(xx):
        #         print(xxx.replace("▁"," "))
            # x = x.replace("▁"," ")
            # print(1,x)
            # print(2,p)
    # print(pos)
        # exit()
    pos = resX
    return pos
# torch.
def analysis_file(filename,result_f):
    # result_f = input('output file name:')
    inputf = open(filename,encoding='utf-8')
    output = open(result_f,'w',encoding='utf-8')
    # output_tk = open('tk_'+result_f,'w',encoding='utf-8')
    count = 0
    for i in inputf:
        count += 1
    inputf.seek(0)
    X = []
            # thread_tok()
    t = 0
    with tqdm(total=count) as pbar:
        for _ in range(count):#input:
            
            l = inputf.readline()
            l = l.strip()
            if l == "":
                continue
            l = l[:tok_max_len]

            X.append(l)
            if len(X) == batch:#2000:
                line = analysis(X)
                X = []
                        
                output.write('\n'.join(line)+'\n')
                pbar.update(batch)
                        
                        
        if len(X) > 0:
            line = analysis(X)
            pbar.update(batch)
            output.write('\n'.join(line)+'\n')
    output.close()

def file_analysis(filename,result_f):
    inputf = open(filename,encoding='utf-8')
    output = open(result_f+"tmp",'w',encoding='utf-8')
    cmpfile = open(filename.replace(".txt","_cmp.txt"),'w',encoding='utf-8')
    count = 0
    for i in inputf:
        count += 1
    inputf.seek(0)
    X = []
    t = 0
    docend = []
    with tqdm(total=count) as pbar:
        for docidx in range(count):#input:
            pbar.update(1)
            l = inputf.readline()
            l = l.strip()
            # if l == "":
                # pbar.update(1)
                # continue
            if l == "<<EOD>>":
                docend.append(docidx)
            # if len(l) > tok_max_len:
            #     X.append("STAT")
            #     X.append(l)
            #     X.append("END")
            # else:
            
            # ltt = removesym(normalize(l)).replace("♥","").strip()
            # if ltt == "":
            #     continue
            # else:
            #     cmpfile.write(l+"\n")
            l = l.replace("END","end")
            l = l.replace("STAT","stat")
            X.append(l)
            if len(X) >= batch:# or len(l) > tok_max_len:#2000:
                # X = X[:-1]
                # if len(X) > 0:
                line = analysis(X)
                for line_ in line:
                    # if line_.strip() == "":
                    #     continue
                    output.write(line_.strip()+'\n')
                    output.flush()
                    # a=True
                    # pbar.update(len(X))
                X = []
                    
            # print(len())
            
                
        if len(X) > 0:
            line = analysis(X)
            # print(line)
            # pbar.update(len(X))
            for line_ in line:
                # if line_.strip() == "":
                #     continue
                output.write(line_.strip()+'\n')
                output.flush()
    inputf.close()
    output.close()
    
    
    outputtmp = open(result_f+"tmp",encoding='utf-8')
    output = open(result_f,'w',encoding='utf-8')
    resulttmp = []
    for idx, l in enumerate(outputtmp):
        if idx in docend:
            output.write("\n".join(resulttmp)+"\n<<EOD>>\n")
            resulttmp = []
        else:
            l = l.strip()
            resulttmp.append(l)
    
    if len(resulttmp) > 0:
        output.write("\n".join(resulttmp)+"\n")
    
    outputtmp.close()
    output.close()
if __name__ == '__main__':

    while True:

        if args.input != "false" and args.output != "false":
            file_analysis(args.input,args.output)
            exit()
        
        mode = input('mode file!!"filename",or text!!"text or cnoun!!"headtail text file" or exit:')
        args_ = mode.split('!!')
        mode = args_[0]
        if len(args_) > 1:
            filename = args_[1]
        if mode == 'file':
            result_f = input('output file name:')
            file_analysis(filename,result_f)
        elif mode == 'text':
            line = filename
            line = line.strip()
            # print(line)
            # print(len(line))
            # if len(line) > tok_max_len:
            #     line = [line]
            pos = analysis(line)
            print(pos[0])

        # elif mode == "cnoun":
        #     file = open(filename.replace(".txt","_cnoun.txt"),"w",encoding="utf-8")
        #     with open(filename,encoding="utf-8") as f:
        #         cnt = len(f.readlines())
                
        #         pbar = tqdm(total=cnt)
        #         f.seek(0)
        #         for l in f:
        #             pbar.update(1)
        #             l = l.strip()
        #             l = get_cnoun([l])[0]
        #             file.write(l+"\n")
        #         pbar.close()   
        #     file.close()
            
        elif mode == 'exit':
            break

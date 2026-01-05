from enum import auto
import torch
import os
# from unicode import join_jamos
from headtail_train import normalize,removesym, infer, HTInputTokenizer,get_xstags, TokIOProcess, HeadTail, PickableInferenceSession
# from headtail import infer_onnx as infer
from customutils import AC,original_pos
# from headtail import convert_qint4

root = __file__

root = root.split(os.sep)
root = os.sep.join(root[:-1])
# print(root)
os.environ['HT'] = root + os.sep + "postagger_model"

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["OMP_NUM_THREADS"] = "8"
# os.environ["MKL_NUM_THREADS"] = "8"
# torch.set_num_threads(8)


import tokenizer_ht as tokenizer_ht
from tokenizer_model import TK_Model_Mini2 as TK_Model2

import sys
sys.setrecursionlimit(100000)
sys.path.append(root + os.sep + "postagger_model")
from postagger_model import PosLSTMMini2 as PosLSTM
tok_max_len = 230
parrel_size = 16
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

from transformer import Encoder, TransformerConfig
import torch.quantization

tok_model = None
pos_model = None
# mod = None
tok_model2 = None
tok_model3 = None
pos_model2 = None
pos_model3 = None
def tokmodel(posvocab,device):
    print("tok")
    max_len = 260
    # tag_len = len(posvocab.index2pos)
    global tok_model, tok_model2, tok_model3
    with open(root + os.sep + "tokenizer_model" + os.sep + 'lstm_vocab.pkl','rb') as f:
        lstm_vocab = pickle.load(f)
    tok_model = TK_Model2(max_len,lstm_vocab)
    tok_model.load_state_dict(torch.load(root+os.sep + "tokenizer_model"+'/ht_tokenizer_model3/model_9',weights_only=True,map_location=device))
    tok_model = tok_model.to(device=device)
    tok_model.eval()
    x = torch.tensor(torch.randint(0,len(lstm_vocab),(1,260),requires_grad=True,dtype=torch.float).detach().numpy().tolist(),dtype=torch.float)#.to(device)#.to(device)
    unfold = x.unfold(1,3,1)
    print(x.shape)
    print(unfold.shape)
    # exit()
    torch.onnx.export(
        tok_model,               # 실행될 모델
        (x,unfold),                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
        "super_resolution_tok.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
        export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
        opset_version=17,          # 모델을 변환할 때 사용할 ONNX 버전
        do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
        input_names = ['input',"unfold"],   # 모델의 입력값을 가리키는 이름
        output_names = ['output'], # 모델의 출력값을 가리키는 이름
        dynamic_axes={
            'input' : {0 : 'batch_size',1:"sentence_length"},    # 가변적인 길이를 가진 차원
            'unfold' : {0 : 'batch_size',1:"sentence_length"},
            'output' : {0 : 'batch_size',1:"sentence_length"}
        }
    )
    # exit()
    def exportonnx():
        x = torch.tensor(torch.randint(0,len(lstm_vocab),(1,260),requires_grad=True,dtype=torch.float).detach().numpy().tolist()).to(device)#.to(device)
        unfold = x.unfold(1,3,1)
        
        print(x.shape)
        torch_out = tok_model(x,unfold)
        print(torch_out[0].shape)
        # exit()
        torch.onnx.export(
            tok_model,               # 실행될 모델
            (x,unfold),                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
            "super_resolution_tok.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
            export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
            opset_version=12,          # 모델을 변환할 때 사용할 ONNX 버전
            do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
            input_names = ['input',"unfold"],   # 모델의 입력값을 가리키는 이름
            output_names = ['output'], # 모델의 출력값을 가리키는 이름
            dynamic_axes={
                'input' : {0 : 'batch_size',1:"sentence_length"},    # 가변적인 길이를 가진 차원
                'unfold' : {0 : 'batch_size',1:"sentence_length"},
                'output' : {0 : 'batch_size',1:"sentence_length"}
            }
        )
        
        # import onnx
        
        # onnx.save(onnx.shape_inference.infer_shapes(onnx.load("super_resolution_tok.onnx")), "super_resolution_tok.onnx")
        # onnx_model = onnx.load("super_resolution_tok.onnx")
        
        # onnx.checker.check_model(onnx_model)
        # # print(onnx_model(x))
        # # exit()
        # import onnxruntime
        # x = torch.randint(0,len(lstm_vocab)+1,(1,260,1),dtype=torch.float32).to("cpu")#.to(device)
        # unfold = x.unfold(1,3,1)
        # # print()
        # ort_session = onnxruntime.InferenceSession("super_resolution_tok.onnx")
        # torch_out = torch_out.detach().numpy()
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        x = x.detach().cpu().numpy()

        output = "super_resolution_tok.onnx" # onnx 모델 위치
        converter = "tok-model-uint8.onnx" # 저장할 모델 위치
        quantize_dynamic(output, converter, weight_type=QuantType.QInt8)

        # converterint4 = "tok-model-uint4.onnx"
        # convert_qint4(output,converterint4)
        
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        # exit()
    # exportonnx()

def posmodel(device):
    print("pos")
    global pos_model,pos_model2,pos_model3
    with open(root + os.sep + "postagger_model" + os.sep + "/ht_postagger_model3/subwordvocab.pkl","rb") as f:
        posvocab = pickle.load(f)
    pos_model = PosLSTM(posvocab)
    pos_model.load_state_dict(torch.load(root + os.sep + "postagger_model" + os.sep + "ht_postagger_model3/model",weights_only=True,map_location=device)["model"])
    pos_model = pos_model.to(device=device)
    pos_model.eval()
    
    x = torch.randint(0,15001,(1,260),requires_grad=True,dtype=torch.float).to(device)#.to(device)
    unfold = x.unfold(1,3,1)
    torch.onnx.export(pos_model,               # 실행될 모델
            (x,unfold),                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
            "super_resolution_pos.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
            export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
            opset_version=17,          # 모델을 변환할 때 사용할 ONNX 버전
            do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
            input_names = ['input',"unfold"],   # 모델의 입력값을 가리키는 이름
            output_names = ['output'], # 모델의 출력값을 가리키는 이름
            dynamic_axes={'input' : {0 : 'batch_size',1:"sentence_length"},    # 가변적인 길이를 가진 차원
                        'unfold' : {0 : 'batch_size',1:"sentence_length"},
                        'output' : {0 : 'batch_size',1:"sentence_length"}})
    # exit()
    def exportonnx():
        x = torch.randint(0,15001,(1,260),requires_grad=True,dtype=torch.float).to(device)#.to(device)
        unfold = x.unfold(1,3,1)
        print(x.dtype)
        torch_out = pos_model(x,unfold)

        torch.onnx.export(pos_model,               # 실행될 모델
            (x,unfold),                         # 모델 입력값 (튜플 또는 여러 입력값들도 가능)
            "super_resolution_pos.onnx",   # 모델 저장 경로 (파일 또는 파일과 유사한 객체 모두 가능)
            export_params=True,        # 모델 파일 안에 학습된 모델 가중치를 저장할지의 여부
            opset_version=12,          # 모델을 변환할 때 사용할 ONNX 버전
            do_constant_folding=True,  # 최적화시 상수폴딩을 사용할지의 여부
            input_names = ['input',"unfold"],   # 모델의 입력값을 가리키는 이름
            output_names = ['output'], # 모델의 출력값을 가리키는 이름
            dynamic_axes={'input' : {0 : 'batch_size',1:"sentence_length"},    # 가변적인 길이를 가진 차원
                        'unfold' : {0 : 'batch_size',1:"sentence_length"},
                        'output' : {0 : 'batch_size',1:"sentence_length"}})
        import onnx
        
        onnx.save(onnx.shape_inference.infer_shapes(onnx.load("super_resolution_pos.onnx")), "super_resolution_pos.onnx")
        onnx_model = onnx.load("super_resolution_pos.onnx")
        
        onnx.checker.check_model(onnx_model)
        # print(onnx_model(x))
        # exit()
        import onnxruntime
        x = torch.randint(0,15001,(1,260),dtype=torch.float).to("cpu")#.to(device)
        # print()
        ort_session = onnxruntime.InferenceSession("super_resolution_pos.onnx")
        # torch_out = torch_out.detach().numpy()
        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
        x = x.detach().cpu().numpy()

        print("Exported model has been tested with ONNXRuntime, and the result looks good!")
        output = "super_resolution_pos.onnx" # onnx 모델 위치
        converter = "pos-model-uint8.onnx" # 저장할 모델 위치

        quantize_dynamic(output, converter, weight_type=QuantType.QInt8)

        
        # converterint4 = "pos-model-uint4.onnx"
        # convert_qint4(output,converterint4)
    # exportonnx()
    
def get_tok(line,tok_model,device,verbose=0):
    tok = tokenizer_ht.predict2(tok_model,line,device,verbose=verbose)

    return tok

def get_pos(tok,pos_model,verbose=0):
    # print(tok)
    pos = infer(posvocab,pos_model,tok,device)


thread_num = 16
batch = 256
# models = [[tok_model,tok_model2,tok_model3],[pos_model,pos_model2,pos_model3]]
from tqdm import tqdm    

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

        
import pickle


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
        # x_ = xx
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

        return ["STAT"] + sents + ["END"]
    return [xx.replace(' ','▁').strip('▁')]

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
    return pos

import multiprocessing as mp
from datetime import datetime
import os
import math

def cpu_analysis(args,X,mps): 
    start = datetime.now()
    starttotal = start

    xtmp = []
    XXX = []
    XX = []
    
    for xx in X:
        xtmp.append(xx)
        if len(xtmp) == 8:
            XX.append(xtmp)
            xtmp = []
    
    if len(xtmp) > 0:
        XX.append(xtmp)
    batch_size = math.ceil(len(XX) / args.workers)

    XXXtmp = []
    for xxx in XX:
        XXXtmp.append(xxx)
        if len(XXXtmp) == batch_size:
            XXX.append(XXXtmp)
            XXXtmp = []
    if len(XXXtmp) > 0:
        XXX.append(XXXtmp)
    
    for i in range(len(XXX)):
        mps[i][1].put(XXX[i])

    toks = []
    tmp = []

    count = 0
    for i in range(len(XXX)):
        toks.append(mps[i][2].get())
        count += 1

    toks = sorted(toks,key=lambda x: x[0])
    result = []
    for tok in toks:
        result = result + tok[1]
    # print(result[0])
    # exit()
    return result

def make_mp(args,origin=True,tok_model="tok-model-uint8.onnx",pos_model="pos-model-uint8.onnx",tkmode=True,posmode=True):
    mps = []
    
    outq = mp.Queue()
    for i in range(args.workers):
        inq = mp.Queue()
        io_process = TokIOProcess(i,inq,tok_model,pos_model,outq,origin,tkmode,posmode)
        mps.append([io_process,inq,outq])
    
    return mps

def cpu_analysis_textmode(X,origin=True):
    start = datetime.now()
    starttotal = start
    toks = []
    i = 0

    q = [] #mp.Queue()
    io_process = HeadTail(i,[X],os.path.join(os.environ["KJM"],"headtail","tok-model-uint8.onnx"),os.path.join(os.environ["KJM"],"headtail","pos-model-uint8.onnx"),q,origin)
        # io_process.start()
    y = io_process.run(X)  
    # if os.path.exists(f"{i}_pr.txt"):
    #     with open(f"{i}_pr.txt",encoding="utf-8") as f:
    #         toks = toks + f.readlines()
    #     os.remove(f"{i}_pr.txt")
    # toks = list(map(lambda x: x.strip(),toks))

    
    return y#toks#result

def analysis(args,x,verbose=0,cpuflag=True,istext=False,mps=[],origin=True):
    X = line_normalie(x)
    
    if cpuflag and istext:
        pos = cpu_analysis_textmode(X,origin=origin)
    elif cpuflag and not istext:
        pos = cpu_analysis(args,X,mps=mps)
    else:
        tok = inference_tok(X)
        pos = inference_pos(tok)
    
    x_ = []

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
    for idx,xx in enumerate(X):
        xx = xx.replace("▁"," ").strip()
        if xx == "STAT":
            # print(len(pos),idx)
            
            indexs.append(pos[idx])
            splitflag =True
            s += 1
        elif xx == "END":
            # print(len(pos),idx)
            indexs.append(pos[idx])
            tmp = []
            resX.append(" ".join(indexs[1:-1]))
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
                restok = restok[stchk:]
                
            restok = " ".join(restok)
            # print(restok)
            indexs.append(restok)
        elif not splitflag:
            # print(len(pos))
            # print(idx)
            # if idx <
            # if 0 <= idx < len(pos):
            # print(len(X),len(pos))
            resX.append(pos[idx])
            splitflag = False

    for xx in x:
        if xx != "STAT" and xx != "END":
            x_.append(xx)
            
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

def file_analysis(args,filename,result_f,mps):
    
    for mp_ in mps:
        if not mp_[0].is_alive():
            mp_[0].start()
            # print("start")
    linecount = 0
    inputf = open(filename,encoding='utf-8')
    output = open(result_f,'w',encoding='utf-8')
    # cmpfile = open(filename.replace(".txt","_cmp.txt"),'w',encoding='utf-8')
    count = 0
    for i in inputf:
        count += 1
    inputf.seek(0)
    X = []
    t = 0
    docend = []
    linecnt = 0
    with tqdm(total=count) as pbar:
        for docidx in range(count):#input:
            l = inputf.readline()
            l = l.strip()
            # l = l.replace("▁"," ")
            if l == "<<EOD>>":
                docend.append(docidx)
            else:
                linecnt += 1
            l = l.replace("END","end")
            l = l.replace("STAT","stat")
            X.append(l)
            

            if len(X) >= batch:
                line = analysis(args,X,mps=mps)
                for line_ in line:
                    if line_ == "EOD/SL":
                        output.write('<<EOD>>\n')
                    else:    
                        output.write(line_.strip()+'\n')
                    output.flush()

                X = []
            pbar.update(1)
            # if linecnt == 360000:
            #     break
            
                
        if len(X) > 0:
            line = analysis(args,X,mps=mps)

            for line_ in line:
                output.write(line_.strip()+'\n')
                output.flush()
    inputf.close()
    output.close()
    
    for mp_ in mps:
        mp_[1].put("[END]")
    
    for mp_ in mps:
        mp_[0].join()
if __name__ == '__main__':
    
    mp.set_start_method('spawn') # This is important and MUST be inside the name==main block.    
    
    print('Model 로딩 중입니다. 모델 로딩까지 2분이상 소요 될 수 있습니다.')
    with open(root + os.sep + "postagger_model" + os.sep + "/ht_postagger_model3/subwordvocab.pkl","rb") as f:
        posvocab = pickle.load(f)

    start_t = time.time()
    with open(root + os.sep + "tokenizer_model" + os.sep + 'lstm_vocab.pkl','rb') as f:
        lstm_vocab = pickle.load(f)
        device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu") 
    posmodel(device)
    tokmodel(posvocab,device)
    exit()


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
    parser.add_argument("-o",help="형태소 원형으로 변환",action="store_true")
    parser.add_argument("--workers",help="multi processing 개수",type=int,default=16)
    # parser.add_argument("--folder",help="폴더 경로",type=str,default="")
    args = parser.parse_args()
    EPOCH = args.EPOCH
    max_len = args.MAX_LEN
    
    # tokmodel()
    print("tokenizer model load compelte")
    # posmodel()
    print("pos model load compelte")
    # exit()
    end_t = time.time()
    print("모델 로딩 {}".format((end_t-start_t)))

    while True:

        if args.input != "false" and args.output != "false":
            mps = make_mp(args,origin=args.o)
            file_analysis(args,args.input,args.output,mps)
            
            exit()
        
        mode = input('mode file!!"filename",or text!!"text or cnoun!!"headtail text file" or exit:')
        args_ = mode.split('!!')
        mode = args_[0]
        if len(args_) > 1:
            filename = args_[1]
        if mode == 'file':
            mps = make_mp(args,origin=args.o)


            result_f = input('output file name:')
            file_analysis(args,filename,result_f,mps)

        elif mode == 'text':
            line = filename
            line = line.strip()

            pos = analysis(args,line,istext=True,origin=args.o)
            print(pos[0])
        
        elif mode == "folder":
            dirs = os.listdir(filename)
            
            tmpinfdname = filename.split(os.sep)[-1]
            outpath = os.path.join(os.environ["DATASET"],tmpinfdname+"_headtail")
            mpas = make_mp(args,origin=args.o)
            for fn in dirs:
                file_analysis(args,os.path.join(),os.path.join(outpath,fn),mps)
            
        elif mode == 'exit':
            break

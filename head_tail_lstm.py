from enum import auto
import torch
import os
from unicode import join_jamos
from headtail import normalize,removesym, infer, HTInputTokenizer
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
from tokenizer_model import TK_Model3 as TK_Model2

import sys
sys.setrecursionlimit(100000)
# sys.path.append("C:/Users/ty341/OneDrive/Desktop/server/headtail/postagger/autospace")
sys.path.append(root + os.sep + "postagger_model")
from postagger_model import PosLSTM3 as PosLSTM
# from autospace import AutoSpace, SentenceSplit
# autospace = AutoSpace(temp=0.52)
# stk = SentenceSplit()
# print(root)
tok_max_len = 300
# print('ht',root)
# from urllib import request
# from flask import Flask,request, render_template
# import tensorflow as tf
# from keras import backend as K
# from keras.models import load_model
# import threading as t
import numpy as np
import transformers
transformers.utils.logging.set_verbosity(transformers.logging.ERROR)

import tagging_lstm as tagging
import json
import re
import time
import numpy as np
# app = Flask(__name__)
# import tensorflow as tf
import threading, requests, time
import threading
from multiprocessing import Pool, Process
import time
import torch
from torch.quantization import QuantStub, DeQuantStub, float_qparams_weight_only_qconfig, default_qconfig
import argparse 
import pickle
import torch.nn as nn
from transformers import BertModel, DistilBertModel, AdamW
from pos_vocab import BiPosVocab

# from train_tkbigram_pos_tagger_pytorch_peft_bi import PosTaggerModelPeftTorch
with open(root + os.sep + "postagger_model" + os.sep + "posvocab.pkl","rb") as f:
    posvocab = pickle.load(f)
with open(root + os.sep + "postagger_model" + os.sep + "modu2/subwordvocab2.pkl","rb") as f:
    posvocab = pickle.load(f)
start_t = time.time()
# class PosTaggerModelPeftTorch(nn.Module):
#     def __init__(self,model,maxlen,hidden_state,tag_len):
#         super(PosTaggerModelPeftTorch,self).__init__()
#         self.emb = nn.Embedding(len(tkbigram.keys()),80)#bilm.emb#nn.Embedding(len(bigram.keys()),80)
#         self.emb_lstm = nn.LSTM(80,64)
#         self.bert = model

#         self.sub_layer = nn.Linear(768,64)
#         self.bilstm = nn.LSTM(64,64*2)
#         self.output_tag = nn.Linear(64 * 2,tag_len)

#         self.softmax = nn.LogSoftmax(dim=-1)
#     def forward(self,bi,data,masks,segments):
#         emb = self.emb(bi)
#         emb_lstm,_ = self.emb_lstm(emb)
        
#         bert_output = self.bert(input_ids=data,attention_mask = masks)[0]#,token_type_ids = segments)[0]
#         bert_output = self.sub_layer(bert_output)
#         output = bert_output + emb_lstm#torch.concat([emb_lstm,bert_output[:,1:,:]],dim=-1)
#         output,_ = self.bilstm(output)#(output)
#         output = self.output_tag(output)

#         output = self.softmax(output)
#         return output

print(os.sep,root)
with open(root + os.sep + "tokenizer_model" + os.sep + 'lstm_vocab.pkl','rb') as f:
    lstm_vocab = pickle.load(f)
with open(root + os.sep + "tokenizer_model" + os.sep + 'bigram_vocab.pkl','rb') as f:
    bigram = pickle.load(f)

from transformer import Encoder, TransformerConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

with open(root + os.sep + "postagger_model" + os.sep + 'kcc150_all_tag_dict.pkl','rb') as f:
    tag_dict = pickle.load(f)
    tag_dict = { v:k for k,v in tag_dict.items() }
with open(root + os.sep + "postagger_model" + os.sep + 'kcc150_all_tokenbi.pkl','rb') as f:
    tkbigram = pickle.load(f)

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

with open(root + os.sep + "tokenizer_model" +'/lstm_vocab.pkl','rb') as f:
    vocab = pickle.load(f)
with open(root + os.sep + "tokenizer_model" +'/bigram_vocab.pkl','rb') as f:
    bigram = pickle.load(f)



tok_model = None
pos_model = None
# mod = None
tok_model2 = None
tok_model3 = None
pos_model2 = None
pos_model3 = None
def thread_tok():
    global tok_model, tok_model2, tok_model3
    tok_model = TK_Model2(max_len,lstm_vocab)
    tok_model.load_state_dict(torch.load(root+os.sep + "tokenizer_model"+'/tokenizer_model_modu/model',weights_only=True,map_location=device))
    tok_model = tok_model.to(dtype=torch.bfloat16,device=device)
    tok_model.eval()

def thread_pos():
    global pos_model,pos_model2,pos_model3
    
    pos_model = PosLSTM(posvocab)
    pos_model.load_state_dict(torch.load(root + os.sep + "postagger_model" + os.sep + "/modu3/model",weights_only=True,map_location=device)["model"])
    pos_model = pos_model.to(dtype=torch.bfloat16,device=device)
    pos_model.eval()

thread_tok()
print("tokenizer model load compelte")
thread_pos()
print("pos model load compelte")

end_t = time.time()
print("모델 로딩 {}".format((end_t-start_t)))

def get_tok(line,result,tok_model,device,verbose=0):
    tok = tokenizer_ht.predict2(tok_model,line,device,verbose=verbose)
    result['1'] = tok

def get_pos(tok,tok_line,result,pos_model,verbose=0):
    # print(tok)
    pos = infer(posvocab,pos_model,tok,device)
                        
    # pos = tagging.predictbi_pt(pos_model,tok,tok_line,device)
    result['pos'] = pos
# def get_tok2(line,result,verbose=0):
#     tok = tokenizer.predict(tok_model,line,verbose=verbose)
#     result['2'] = tok
# def get_pos2(tok,tok_line,result,verbose=0):
#     pos = tagging.predictbi(pos_model,tok,tok_line,lite=False,verbose=verbose)
#     result['pos2'] = pos

thread_num = 1
batch = 256
models = [[tok_model,tok_model2,tok_model3],[pos_model,pos_model2,pos_model3]]
from tqdm import tqdm
def preprocess(l):
    
    l = re.sub(r"[\.]+"," . ",l).strip()
    l = re.sub(r"[\!]+"," ! ",l).strip()
    l = re.sub(r"[\?]+"," ? ",l).strip()
    l = re.sub(r"[^a-z|A-Z|ㄱ-ㅎ|가-힣|\d|\.|\!|\?]"," ",l)
    l = re.sub(r"([ㄱ-ㅎ]+)",r" \1 ",l)
    l = re.sub(r"(\d+)",r" \1 ",l)
    # l = re.sub("[0-9]+"," ",l)
    l = re.sub(r" +",r" ",l)
    
    l = re.sub(r"([a-z|A-Z]+)",r" \1 ",l)
    l = re.sub(r" +",r" ",l)
    
    l = re.sub(r"(\d+) \. (\d+)",r" \1.\2 ",l)
    l = re.sub(r" +",r" ",l)
    # l = re.sub(r"([0-9]+)",r" \1 ",l)
    # l = l.replace("."," .")
    # if not l.endswith(".") and not l.endswith("!") and not l.endswith("?"):
    #     l = l.strip()+" ."
    # elif l.endswith("?") or l.endswith("!"):
    #     l = re.sub(r"[\!]+"," !",l)
    #     l = re.sub(r"[\?]+"," ?",l)
    
    l = re.sub(r" +",r" ",l).strip()
    return l, False
    
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
            # tmp[h] = True
            # print(jamo,len(jamo))
            tmp.insert(jamo)
            # print(1,h)
        heads_list = tmp

    with open("general.dict","wb") as f:
        pickle.dump(heads_list,f)
        

with open("general.dict","rb") as f:
    heads_list = pickle.load(f)
    
def analysis(x,verbose=0):
    if type(x) == str:
        xxx = [x]
    else:
        xxx = x    
    X = []
    for idx,xx in enumerate(xxx):
        dot_flag = False
        # if idx+1==17:
        #     print(xx)
        #     print(normalize(xx))
        #     print(removesym(normalize(xx)))
        xx = xx.strip()
        xx = xx.replace("+","_")
        xx = xx.replace("/","_")
        x_ = normalize(xx)
        # x_ = removesym(x_)
        if dot_flag and len(xx) >= tok_max_len:
            xx = xx[:tok_max_len-2]
            xx = normalize(xx)
            # xx = removesym(xx)
            
        else:
            xx = x_
        xx = removesym(xx)
        # xx = x_     
        if tok_max_len > len(xx):
            xx = xx[:tok_max_len]
        xx = xx.strip()
        X.append(xx.replace(' ','▁').strip('▁'))
    result = [{}]
    get_tok(X,result[0],tok_model,device)
    tok_ = []
    tok = result[0]['1']
    # print(tok)
    for tk in tok:
        tk = re.sub(" +"," ",tk)
        tok_.append(tk)
    get_pos(tok_,tok,result[0],pos_model,device)
    pos = result[0]['pos']
    pos_ori = []
    
    # for p in pos:
    #     p = p.split()
    #     ptmp = []
    #     for idx,pp in enumerate(p):
    #         # print(pp)
    #         pp = pp.split("+")[-1].split("/")[1]
    #         if idx + 1 != len(p):
    #             ppnext = p[idx+1].split("+")[-1].split("/")[1]
    #         else:
    #             ppnext = ""
    #         if pp.endswith("EF"):
    #             if ppnext.endswith("SF"):
    #                 ptmp.append(p[idx])
    #             else:
    #                 ptmp.append(p[idx]+"|")
    #         elif pp.endswith("SF"):
    #             if ppnext.endswith("SF"):
    #                 ptmp.append(p[idx])
    #             else:
    #                 ptmp.append(p[idx]+"|")
    #         else:
    #             ptmp.append(p[idx])
            
    #     pos_ori.append(" ".join(ptmp))
    # pos = pos_ori
    # print(pos)
    if False:
        for idx1,p in enumerate(pos):
            line = []
            for l in p.split():
                tk = []
                for idx,ll in enumerate(l.split("+")):
                    if idx == 0 or (idx ==1 and ("/VV" in ll or "/VA" in ll)):
                        ll_ = ll.split("/")
                        headlast_char = ll_[0][-1]
                        headtag = ll_[1]
                        if "+" in l and idx == 0:
                            tail_t = l.split("+")[1].split("/")[1]
                            res = head_dict[f"{headlast_char}/{headtag}@{tail_t}"]#.query(f"token==\"{ll_[0][-1]}/{ll_[1]}\"")
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
                                head_tmp = ll_[0][:-1]+tktmp
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
                    elif idx == 1:
                        if "_" not in ll.split("/")[1]:
                            tk.append(ll)
                            continue
                        head_t = l.split("+")[0].split("/")[1]
                        ll = ll.replace("/",f"/{head_t}@")
                        
                        res = tail_dict[f"{ll}"]
                        
                        if len(res) > 0:
                            res = list(res)[-1]
                            res_mp = [tmp.split("/")[0] for tmp in res.split()]
                            res_tag = [tmp.split("/")[1] for tmp in res.split()]
                            tk.append("_".join(res_mp)+"/"+"_".join(res_tag))
                        else:
                            ll = ll.split("@")
                            ll[0] = ll[0].split("/")[0]
                            ll = "/".join(ll)
                            tk.append(ll)
                line.append("+".join(tk))
            pos_ori.append(" ".join(line))
        pos = pos_ori
    # exit()
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
                    
                    #if tok_max_len > len(l):
            l = l[:tok_max_len]

            X.append(l)
                    #print(X[-1])
                    # X.append(l)
            if len(X) == batch:#2000:
                line = analysis(X)
                X = []
                        
                output.write('\n'.join(line)+'\n')
                pbar.update(batch)
                        
                        
        if len(X) > 0:
            line = analysis(X)
            pbar.update(batch)
                    # print(line)
            output.write('\n'.join(line)+'\n')
    output.close()

if __name__ == '__main__':

    while True:

        if args.input != "false" and args.output != "false":
            inputf = open(args.input,encoding="utf-8")
            outputf = open(args.output,"w",encoding="utf-8")
            
            count = 0
            for i in inputf:
                count += 1
            inputf.seek(0)
            X = []
            t = 0
            with tqdm(total=count) as pbar:
                for _ in range(count):#input:
                    l = inputf.readline()
                    l = l.strip()
                    l = l[:tok_max_len]

                    X.append(l)
                    if len(X) == batch:#2000:
                        line = analysis(X)
                        X = []
                        for result in line:
                            for res in result.split("|"):
                                if res.strip() != "":
                                    outputf.write(res.strip()+'\n')
                        pbar.update(batch)
                        # exit()
                        
                if len(X) > 0:
                    line = analysis(X)
                    # print(line)
                    for result in line:
                        for res in result.split("|"):
                            if res.strip() != "":
                                outputf.write(res.strip()+'\n')
            inputf.close()
            outputf.close()
            exit()
        
        mode = input('mode file!!"filename",or text!!"text or cnoun!!"headtail text file" or exit:')
        dot_flag = []
        args_ = mode.split('!!')
        mode = args_[0]
        if len(args_) > 1:
            filename = args_[1]
        if mode == 'file':
            # filename = input('file name:')
            result_f = input('output file name:')
            inputf = open(filename,encoding='utf-8')
            output = open(result_f,'w',encoding='utf-8')
            output_tk = open(result_f.replace(".txt","_tk.txt"),'w',encoding='utf-8')
            count = 0
            for i in inputf:
                count += 1
            inputf.seek(0)
            X = []
            t = 0
            with tqdm(total=count) as pbar:
                for _ in range(count):#input:
                    l = inputf.readline()
                    l = l.strip()
                    l = l[:tok_max_len]

                    X.append(l)
                    if len(X) == batch:#2000:
                        line = analysis(X)
                        X = []
                        
                        output.write('\n'.join(line)+'\n')
                        pbar.update(batch)
                        
                        
                if len(X) > 0:
                    line = analysis(X)
                    # print(line)
                    output.write('\n'.join(line)+'\n')
            inputf.close()
            output.close()
                    
        elif mode == 'text':
            line = filename
            line = line.strip()
            line = line[:tok_max_len]
            pos = analysis(line)
            print(pos[0])
            # # dot_flag = False
            # # if line[-1] != '.':
            # #     dot_flag = True
            # line_, dot_flag = preprocess(line)
            # if dot_flag and len(line) >= tok_max_len:
            #     line = line[:tok_max_len-2]
            #     line,_ = preprocess(line)
            # else:
            #     line = line_
            
            # # line,_ = preprocess(line)
            # line = [line.replace(' ','▁')]
            # tok = get_tok(line)
            # tok[0] = tok[0].replace('▁',' ')
            # tok_ = []
            # for tk in tok:
            #     tok_.append(tk.replace('+',' '))
            # pos = get_pos(tok_,tok)
            # # print(pos)
            # if dot_flag:
            #     pos = [' '.join(pos[0].split()[:-1])]
            # print(pos[0])
        elif mode == "cnoun":
            file = open(filename.replace(".txt","_cnoun.txt"),"w",encoding="utf-8")
            with open(filename,encoding="utf-8") as f:
                cnt = len(f.readlines())
                
                pbar = tqdm(total=cnt)
                f.seek(0)
                for l in f:
                    pbar.update(1)
                    l = l.strip()
                    l = get_cnoun([l])[0]
                    file.write(l+"\n")
                pbar.close()   
            file.close()
            
        elif mode == 'exit':
            break

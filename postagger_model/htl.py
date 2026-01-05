from enum import auto
import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

root = __file__

root = root.split(os.sep)
root = os.sep.join(root[:-1])
os.environ['HT'] = root
import sys
sys.path.append("/data1/postagger/autospace")
from autospace import AutoSpace, SentenceSplit
autospace = AutoSpace(temp=0.6)
stk = SentenceSplit()
# print(root)
tok_max_len = 230
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
import tokenizer_ht as tokenizer_ht
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
# from train_tkbigram_pos_tagger_pytorch_peft_bi import PosTaggerModelPeftTorch
with open(root + os.sep + "posvocab.pkl","rb") as f:
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


with open(root + os.sep + 'lstm_vocab.pkl','rb') as f:
    lstm_vocab = pickle.load(f)
with open(root + os.sep + 'bigram_vocab.pkl','rb') as f:
    bigram = pickle.load(f)

# class PosTaggerModelPeftTorchLSTM(nn.Module):
#     def __init__(self,model,maxlen,hidden_state,tag_len):
#         super(PosTaggerModelPeftTorchLSTM,self).__init__()
 
#         self.emb = nn.Embedding(len(posvocab.word2index),128)#bilm.emb#nn.Embedding(len(bigram.keys()),80)
 
#         self.emb_lstm = nn.LSTM(128,256,bidirectional=True,batch_first=True)

#         self.sub_layer = nn.Linear(256*2,len(posvocab.pos2index))

#         self.softmax = nn.LogSoftmax(dim=-1)

#     def forward(self,bi,data,masks,segments):

#         emb = self.emb(data)

#         emb_lstm,_ = self.emb_lstm(emb)

#         output = self.sub_layer(emb_lstm)
#         output = self.softmax(output)

#         return output
    
# class PosTaggerModelPeftTorchLSTM2(nn.Module):
#     def __init__(self,model,maxlen,hidden_state,tag_len):
#         super(PosTaggerModelPeftTorchLSTM2,self).__init__()
#         # self.quant1 = torch.ao.quantization.QuantStub()
#         # self.quant2 = torch.ao.quantization.QuantStub()
#         # self.quant3 = torch.ao.quantization.QuantStub()
#         # self.quant4 = torch.ao.quantization.QuantStub()
#         self.emb = nn.Embedding(len(posvocab.word2index),64)#bilm.emb#nn.Embedding(len(bigram.keys()),80)
#         # print(w2v.wv.vectors[0])

#         self.emb_lstm = nn.LSTM(64,128,bidirectional=True,batch_first=True)

#         self.sub_layer = nn.Linear(128*2,len(posvocab.pos2index))
#         # self.bilstm = nn.LSTM(64,64*2)
#         # self.output_tag = nn.Linear(12,tag_len)

#         self.softmax = nn.LogSoftmax(dim=-1)

#     def forward(self,bi,data,masks,segments):

#         emb = self.emb(data)

#         emb_lstm,_ = self.emb_lstm(emb)
        
#         output = self.sub_layer(emb_lstm)
#         output = self.softmax(output)

#         return output

class TK_Model(nn.Module):
    def __init__(self):
        super(TK_Model,self).__init__()
        self.emb = nn.Embedding(len(lstm_vocab.keys()),128)
        
        self.emb_bi = nn.Embedding(len(bigram.keys()),128)

        self.bilstm = nn.LSTM(128,256,bidirectional=True,batch_first=True)
        self.bilstm_bi = nn.LSTM(128,256,bidirectional=True,batch_first=True)

        self.output_tag = nn.Linear(256 * 2,4)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self,x,bi):
        emb = self.emb(x)
        emb_bi = self.emb_bi(bi)

        bilstm,_ = self.bilstm(emb)
        bilstm_bi,_ = self.bilstm_bi(emb_bi)

        concat = bilstm + bilstm_bi#torch.concat([bilstm,bilstm_bi],dim=-1)
        output = self.output_tag(concat)
        output = self.softmax(output)

        return output

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
# torch.set_num_threads(50)
# device = torch.device("cpu")
# with open('kcc150_all_tag_dict.pkl','rb') as f:
#     tag_dict = pickle.load(f)
with open(root + os.sep + 'kcc150_all_tag_dict.pkl','rb') as f:
    tag_dict = pickle.load(f)
    tag_dict = { v:k for k,v in tag_dict.items() }
with open(root + os.sep + 'kcc150_all_tokenbi.pkl','rb') as f:
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

args = parser.parse_args()
EPOCH = args.EPOCH
max_len = args.MAX_LEN


# device = torch.device("cuda" if torch.cuda.is_available() else "
# ") # PyTorch v0.4.0


# tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
print('Model 로딩 중입니다. 모델 로딩까지 2분이상 소요 될 수 있습니다.')


root = os.environ['HT']
# root = '.'
# max_len = 300
with open(root+'/lstm_vocab.pkl','rb') as f:
    vocab = pickle.load(f)
with open(root+'/bigram_vocab.pkl','rb') as f:
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
    tok_model = TK_Model()
    tok_model.load_state_dict(torch.load(root+'/tokenizer_model/model', map_location="cpu"))
    tok_model = tok_model.to(device)
    tok_model.eval()

    # tok_model = torch.quantization.quantize_dynamic(
    #   	tok_model,{torch.nn.Linear,torch.nn.LSTM}, dtype=torch.qint8
    # )
    # tf.keras.models.load_model(root + os.sep + 'lstm_bigram_tokenizer.model',compile=False)
    # tok_model2 = tf.keras.models.load_model(root + os.sep + 'lstm_bigram_tokenizer.model',compile=False)
    # tok_model3 = tf.keras.models.load_model(root + os.sep + 'lstm_bigram_tokenizer.model',compile=False)
# from tensorflow import Graph, Session
# thread_graph = Graph()
# with thread_graph.as_default():
#     thread_session = Session()
#     with thread_session.as_default():
class PosLSTM(nn.Module):
    def __init__(self,model,maxlen,hidden_state,tag_len):
        super(PosLSTM,self).__init__()
        # print(tag_len)
        self.emb = nn.Embedding(len(posvocab.index2word),128)#bilm.emb#nn.Embedding(len(bigram.keys()),80)
 
        self.emb_lstm = nn.LSTM(128,256,bidirectional=True,batch_first=True)

        self.embbi = nn.Embedding(len(posvocab.index2uni),128)#bilm.emb#nn.Embedding(len(bigram.keys()),80)
 
        self.embbi_lstm = nn.LSTM(128,256,bidirectional=True,batch_first=True)


        self.sub_layer = nn.Linear(256*2,tag_len)
        

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,bi,data,masks,segments):
        # print(data.shape)

        emb = self.emb(data)
        # print(emb.shape)
        emb_lstm,_ = self.emb_lstm(emb)

        embbi = self.embbi(bi)
        embbi_lstm, _ = self.embbi_lstm(embbi)
        # print(emb_lstm.shape)
        # exit()

        output = self.sub_layer(emb_lstm+embbi_lstm)
        output = self.softmax(output)

        return output
    
def thread_pos():
    global pos_model,pos_model2,pos_model3
    # ptmodel = DistilBertModel.from_pretrained('monologg/distilkobert')
    # pos_model_ = PosTaggerModelPeftTorch(ptmodel,max_len,args.hidden_state,tag_len)
    # pos_model_.load_state_dict(torch.load(root + os.sep + "pos_model_bi/model"))
    # pos_model_ = pos_model_.to(device)
    # pos_model_.eval()

    # # start_t = time.time()
    # # model.eval()
    # pos_model = torch.quantization.quantize_dynamic(
    #  	pos_model_,{torch.nn.Linear,torch.nn.LSTM}, dtype=torch.qint8
    # )
    # with open(root + os.sep + "posvocab.pkl","rb") as f:
    #     posvocab = pickle.load(f)
    pos_model = PosLSTM(None,None,None,tag_len)
    pos_model.load_state_dict(torch.load(root + os.sep + "lstm/model", map_location="cpu"))
    pos_model = pos_model.to(device)
    pos_model.eval()
    # pos_model = torch.quantization.quantize_dynamic(
    #   	pos_model,{torch.nn.Linear,torch.nn.LSTM}, dtype=torch.qint8
    # )
    # tf.keras.models.load_model(root + os.sep + 'tkbigram_one_first_alltag_bert_tagger_distil.model',compile=False)
    # pos_model2 = tf.keras.models.load_model('tkbigram_one_first_alltag_bert_tagger_electra.model',compile=False)
    # pos_model3 = tf.keras.models.load_model('tkbigram_one_first_alltag_bert_tagger_electra.model',compile=False)
    # pos_model3 = tf.keras.models.load_model('tkbigram_one_first_alltag_bert_tagger_electra.model',compile=False)
    # pos_model = tf.keras.models.load_model(root + os.sep + 'tkbigram_one_first_alltag_bert_tagger_distil.model',compile=False)
        # graph = tf.get_default_graph()
        # sess = tf.get_default_session()



thread_tok()
thread_pos()
# self.cnn_model = load_model(model_path)
# self.cnn_model.predict(np.array([[0,0]])) # warmup
# K.clear_session()
# session = K.get_session()
# graph = tf.get_default_graph()
# graph.finalize() # finalize
end_t = time.time()
print("모델 로딩 {}".format((end_t-start_t)))
# end = time.time()
# print(end-start)
# exit()
def get_tok(line,result,tok_model,device,verbose=0):
    # line = line.replace(' ','▁')
    # if model == None:
    # with session.as_default():
    #     with graph.as_default():
    tok = tokenizer_ht.predict(tok_model,line,device,verbose=verbose)
    #print(tok)
    # token = tok[0].split(' ')
    # result = []
    result['1'] = tok
    # for t in token:
    #     t = t.split('+')
    #     result.append(t)

    # return tok

def get_pos(tok,tok_line,result,pos_model,verbose=0):
    # with session.as_default():
    #     with graph.as_default():
    pos = tagging.predictbi_pt(pos_model,tok,tok_line,device)
    result['pos'] = pos
    # x1 = tok[:100]
    # x2 = tok[100:]
    # tkline1 = tok_line[:100]6
    # tkline2 = tok_line[100:]
    # pos = tagging.start(x1,x2,tkline1,tkline2)
    # postag = pos[0].split(' ')
    # result = []
    # for p in postag:
    #     p = p.split('+')
    #     result.append(p)
    
    # return pos#result,postag
def get_tok2(line,result,verbose=0):
    # line = line.replace(' ','▁')
    # if model == None:
    # with session.as_default():
    #     with graph.as_default():
    tok = tokenizer_ht.predict(tok_model,line,verbose=verbose)
    result['2'] = tok
    # token = tok[0].split(' ')
    # result = []
    # for t in token:
    #     t = t.split('+')
    #     result.append(t)

    # return tok
def get_pos2(tok,tok_line,result,verbose=0):
    # with session.as_default():
    #     with graph.as_default():    
    pos = tagging.predictbi(pos_model,tok,tok_line,lite=False,verbose=verbose)
    result['pos2'] = pos
    # x1 = tok[:100]
    # x2 = tok[100:]
    # tkline1 = tok_line[:100]6
    # tkline2 = tok_line[100:]
    # pos = tagging.start(x1,x2,tkline1,tkline2)
    # postag = pos[0].split(' ')
    # result = []
    # for p in postag:
    #     p = p.split('+')
    #     result.append(p)
    
    # return pos#result,postag
# def get_pos(tok,tok_line,verbose=0):
    
#     pos = tagging.predictbi(pos_model,tok,tok_line,lite=False,verbose=verbose)
thread_num = 1
batch = 50#200
models = [[tok_model,tok_model2,tok_model3],[pos_model,pos_model2,pos_model3]]
from tqdm import tqdm
def preprocess(l):
    # l = l.split()
    # templ = []
    # for ll in l:
    #     if len(ll) >= 5:
    #         ll = autospace.autospace(ll)
    #     templ.append(ll)
    # print(l)
    if len(l) > 0 and l[-1] != ".":
        l = l +"."
    l = " ".join(stk.split(l))
    l = autospace.autospace(l)
    # l = ' '.join(templ)    
    # print(l)
    l = re.sub("[^a-z|A-Z|ㄱ-ㅎ|가-힣|\.|0-9+]"," ",l)
    #l = re.sub("[0-9]+"," ",l)
    l = re.sub(r"([a-z|A-Z]+)",r" \1 ",l)
    l = re.sub(r"([0-9]+)",r" \1 ",l)
    l = re.sub(" +"," ",l)
    l = l.replace("."," .")
    return l, False
    temp_l = []
    l_temp = 'ㄱ'
    dot_flag = False
    # l = re.sub('[0-9]+\.[0-9]+\.[0-9]+','datetime',l)
    # l = re.sub('[0-9]+\.[0-9]+','float',l)
    # l = re.sub('[0-9]+','number',l)
    for index,ll in enumerate(l):
        ll = ll.lower()
        if '가' <= ll <= '힣' \
            or 'ㄱ' <= ll <= 'ㅎ' \
            or 'a' <= ll <= 'z' \
            or ll == '.' \
            or ll == ' ':
                if ll == '.' and not ('0' <= l_temp <='9'):
                    temp_l.append(' ')
                temp_l.append(ll)
        elif '0' <= ll <= '9':
            temp_l.append(ll)
            if len(l)-1 != index:
                if not ('0' <= l[index+1] <= '9'):
                    temp_l.append(' ')
        l_temp = ll
                #x_te[index_temp] = ' '#' '+x_te[index_temp] + ' '
    #if len(l) > 0  and l[-1] != '.':
    #    temp_l.append(' .')
    #    dot_flag = True
    l = ''.join(temp_l)
    l = re.sub(' +',' ',l)
    # print('pre',l)
    return l, dot_flag
from korean_compound_noun_decomposer.cnoun import cnoun,one_syl,iscnoun
from konlpy.tag import Komoran
k = Komoran()
def get_cnoun(text):
    results = []
    # print(text)
    # tags = ["NNP","NNG"]
    
    for txt_ in text:
        tmp_res = []
        # print(txt_)
        for txt in txt_.split():
            words = txt.split('+')
            # print(txt)
            ht= words[0].split("/")
            # print(ht,iscnoun(ht[0]))
            # print(ht)
            if ht[0] == "":
                continue
            if (ht[1].startswith("NNP") or ht[1].startswith("NNG") or ht[1].startswith("N")) and len(ht[0]) > 1: #and iscnoun(ht[0]):
                # for cn in cnoun(ht[0]):
                #     print(cn)
                # print(ht[0])
                if False:#len(ht[0]) <=3:
                    cnouns = [cn+"/"+ht[1] for cn in one_syl(ht[0],caches=True)]
                else:    
                    cnouns = [cn+"/"+ht[1] for cn in cnoun(ht[0])]
                #cnouns = [cn + "/" +ht[1] for cn in k.morphs(ht[0])]
                tmp_res = tmp_res + ["+".join(cnouns)]
                # print(words)
                if len(words) >= 2:
                    tmp_res = tmp_res + ['+'+'+'.join(words[1:])]
            else:
                # print(1,txt,words)
                tmp_res.append(txt)
        results.append(' '.join(tmp_res).replace(" +","+"))
        # print("fff",results)
        # exit()
    # print(text,results)
    # print(results)
    # exit()
    return results

# n_tags = ["합","봤","본","보","본","냐","될","됄","한","했","하","해","되","됐","됬",'한','할','될','시키','시켜','시켰','시킬','시킨','된','스러',"스럽","스런","드리","드렸","돼","됄"]
n_tags = ["합","될","됄","한","했","하","해","되","됐","됬",'한','할','될','시키','시켜','시켰','시킬','시킨','된','스러',"스럽","스런","드리","드렸","돼","됄"]
# n_tags = []

def head_split(X):
    result =[]
    for x in X:
        x = x.strip().split()[:-1]
        temp =[]
        # print(x)
        for xx in x:
            # xx = xx.split("+")
            morphs = xx
            if "+" in xx:
                h = xx.split("+")[0]
                t = xx.split("+")[1]
                tm = t.split("/")[0]
                tp = t.split("/")[1]
                hm = h.split("/")[0]
                if len(hm) == 0:
                    continue
                ht = h.split("/")[1]

                # if ht.startswith("V") and hm[-1] in n_tags and len(hm) >= 3:
                #if hm[-1] in n_tags and len(hm) >= 3:
                    # print(hm)
                #    ht = "V"
                #    hm = hm[:-1] +"/NNP" #+ hm[-1]+"/"+ht# + "+" + t
                    # hm = hm[:-1] +"/NNP+" + hm[-1]+tm+"/"+ht+"_"+tp
                #    morphs = hm
                    # print(morphs)
                    
                    # exit()
                # elif ht.startswith("V") and hm[-2:] in n_tags and len(hm) >= 3:
                #elif hm[-2:] in n_tags and len(hm) >= 3:
                #    ht = "V"
                #    hm = hm[:-2] +"/NNP" #+ hm[-2:]+"/"+ht# + "+" + t
                    # hm = hm[:-2] +"/NNP+" + hm[-2:]+tm+"/"+ht+"_"+tp
                #    morphs = hm
                    # print(morphs)
                    # exit()
                if ht.startswith("N") and hm[-1] == "들" and len(hm) >= 3:
                    hm = hm[:-1] +"/NNP+" + hm[-1] +"/"+"NNP+"+t#"/"+"XSN" + "+" + t
                    # hm = hm[:-1] +"/NNP+" + hm[-1]+tm+"/"+"XSN_"+ht+"_"+tp
                    morphs = hm
                else:
                    morphs = hm+"/"+ht
                # print(morphs)
            else:
                m = xx.split("/")[0]
                t = xx.split("/")[1]
                if len(m) == 0:
                    continue
                 
                # if t.startswith("V") and m[-1] in n_tags and len(m) >= 3:
                #if m[-1] in n_tags and len(m) >= 3:
                #    t = "V"
                #    m = m[:-1] + "/NNP" #+ m[-1] +"/" + t
                #    morphs = m
                # elif t.startswith("V") and m[-2:] in n_tags and len(m) >=3:
                #elif m[-2:] in n_tags and len(m) >=3:
                #    t = "V"
                #    m = m[:-2] + "/NNP" #+ m[-2:] +"/" + t
                #    morphs = m
                if t.startswith("N") and m[-1] == "들" and len(m) >= 3:
                    m = m[:-1] +"/NNP+" + m[-1]+"/"+"NNP" #+ "+" + t
                    morphs = m
                else:
                    morphs = m + "/" + t
                # print(morphs)
            # print(morphs)
            temp.append(morphs)
        
        result.append(" ".join(temp))

    return result

          

def analysis(x,verbose=0):
    if type(x) == str:
        xxx = [x]
    else:
        xxx = x    
    X = []
    for xx in xxx:
        x_,dot_flag = preprocess(xx)
        if dot_flag and len(xx) >= tok_max_len:
            xx = xx[:tok_max_len-2]
            xx,_ = preprocess(xx)
        else:
            xx = x_
                
        if tok_max_len > len(xx):
            xx = xx[:tok_max_len]
        xx = xx.strip()
        X.append(xx.replace(' ','▁').strip('▁'))
    result = [{}]
    get_tok(X,result[0],tok_model,device)
    tok_ = []
    tok = result[0]['1']
    # return tok 
    # print(tok)
    
    # return tok_
    for tk in tok:
        # if tk.startswith("A 양+"):
        # print(tk)
    	#    exit()
        tok_.append(tk.replace("+"," "))
    get_pos(tok_,tok,result[0],pos_model,device)
    # X = []
    pos = result[0]['pos']
    # print(pos)
    #pos = head_split(pos)
    # print(pos)
    pos = get_cnoun(pos)
    # print(pos)
    # print(pos)
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
                    # print(line)
            output.write('\n'.join(line)+'\n')
    output.close()

if __name__ == '__main__':

    while True:
        mode = input('mode file!!"filename",or text!!"text" or exit:')
        dot_flag = []
        args = mode.split('!!')
        mode = args[0]
        if len(args) > 1:
            filename = args[1]
        
        
        if mode == 'file':
            # filename = input('file name:')
            result_f = input('output file name:')
            inputf = open(filename,encoding='utf-8')
            output = open(result_f,'w',encoding='utf-8')
            output_tk = open('tk_'+result_f,'w',encoding='utf-8')
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
                    print(line)
                    output.write('\n'.join(line)+'\n')
                    
     
        elif mode == 'text':
            # line = input('text:')
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

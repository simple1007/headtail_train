import onnxruntime as ort
import numpy as np
import multiprocessing as mp
# from ht_utils_new import HTInputTokenizer
from korean_cnoun import cnoun
from collections import deque,defaultdict
from headtail_train import get_xstags,HTInputTokenizer
import os
import pickle
import onnx
import re
import torch

# os.environ["HT"] = "D:\\kjm\\headtail\\postagger_model"
def init_session(model_path):
    EP_list = ['CPUExecutionProvider']
    sess_opt = ort.SessionOptions()
    # print(sess_opt.__dict__.keys())
    # print(type(sess_opt))
    # exit()
    sess_opt.intra_op_num_threads = 2
    sess = ort.InferenceSession(model_path,sess_options=sess_opt, providers=EP_list)
    return sess

class PickableInferenceSession: # This is a wrapper to make the current InferenceSession class pickable.
    def __init__(self, model_path):
        self.model_path = model_path
        self.sess = init_session(self.model_path)

    def run(self, *args):
        return self.sess.run(*args)

    def __getstate__(self):
        return {'model_path': self.model_path}

    def __setstate__(self, values):
        self.model_path = values['model_path']
        self.sess = init_session(self.model_path)

class PosIOProcess (mp.Process):
    def __init__(self,x,model_path,queue):
        super(PosIOProcess, self).__init__()
        self.sess = PickableInferenceSession(model_path)
        self.x = x
        self.queue = queue
    def run(self):
        with open("postagger_model/ht_postagger_model/subwordvocab.pkl","rb") as f:
            subwordvocab = pickle.load(f)
        xs,inputs = make_input(self.x,subwordvocab)

        input_name = self.sess.sess.get_inputs()[0].name
        output_name = self.sess.sess.get_outputs()[0].name
        
        y = self.sess.run(None, {input_name: np.array(inputs["input_ids"],dtype=np.float32)})
        y = pos_predict_onnx(subwordvocab,xs,y,inputs)

        self.queue.put(y)

import copy
def pos_predict_onnx(subwordvocab,xs,result,X,origin=False):
    result = np.exp(np.array(result[0]))
    resulttmp = result.copy()
    # print(result.shape)
    # print(sum(result[0][0]))
    result = np.argmax(result,axis=-1)
    # print(result.shape)
    Y = []
    for sent,xx,mapping,yy in zip(xs,X["input_ids"],X["offset_mapping"],result):
        xxtmp = copy.deepcopy(xx)
        yytmp = copy.deepcopy(yy)
        res = []
        xy = xx[xx != subwordvocab.pad]
        xx = xx[:xy.shape[0]]#.cpu().numpy().tolist()
        yy = yy[:xy.shape[0]]#.cpu().numpy().tolist()
        xx = deque(xx)
        yy = deque(yy)
        tmp = []
        ptmp = []
        c = 0
        for offset in mapping:
            if offset[0] == 0 and c!= 0:
                if tmp[-1] == subwordvocab.space and len(tmp) == 1:
                    res.append([subwordvocab.to_mp([tmp])[0],subwordvocab.to_tag([ptmp])[0]])
                    tmp = []
                    ptmp = []
                else:
                    res.append([subwordvocab.to_mp([tmp])[0],subwordvocab.to_tag([ptmp])[0]])
                    tmp = []
                    ptmp = []
            if len(xx) == 0:
                break
            c += 1
            tk = xx.popleft()
            p = yy.popleft()
            tmp.append(tk)
            ptmp.append(p)
            if tk == subwordvocab.pad:
                break
        restmp = []
        for r in res[1:]:
            word = subwordvocab.tokenizer.convert_tokens_to_string(r[0])
            word = re.sub(r" +","",word)

            if word == "[SEP]":
                break
            if len(r[0]) == len(r[1]):
                biocheck = defaultdict(int)

                if r[0][0] == "▁":
                    continue
                elif r[0][0] == "+":
                    restmp.append(["+","+"])
                    continue
                
                failflag = False
                last_pos = ""
                for idx,(t,p) in enumerate(zip(r[0],r[1])):
                    if p.startswith("B_") and idx == 0:
                        if p != "+" and p != "O":
                            biocheck[p[2:]] += 1
                        else:
                            biocheck["Fail"] += 1
                    elif idx == 0 and not p.startswith("B_"):
                        failflag = True
                        if p != "+" and p != "O":
                            biocheck[p[2:]] += 1
                        else:
                            biocheck["Fail"] += 1
                        
                    elif idx > 0 and not p.startswith("B_"):
                        if p != "+" and p != "O":
                            biocheck[p[2:]] += 1
                        else:
                            biocheck["Fail"] += 1
                        failflag = True
                    elif idx > 0 and p.startswith("B_"):
                        if p != "+" and p != "O":
                            biocheck[p[2:]] += 1
                        else:
                            biocheck["Fail"] += 1
                    last_pos = p[2:]
            # if True:   
                if len(biocheck) == 1:
                    for k,v in biocheck.items():
                        if v == len(r[0]):
                            restmp.append([word,k])
                else:
                    # if "NOUN" in biocheck and ("VV" == last_pos or "NVV" == last_pos  or "VA" == last_pos  or "NVA" == last_pos ):
                    #     if "VV" == last_pos or "NVV" == last_pos :
                    #         restmp.append([word,"NVV"])
                    #     elif "VA" == last_pos  or "NVA" == last_pos:
                    #         restmp.append([word,"NVA"])
                    # elif "NOUN" in biocheck and ("XSN"  == last_pos  or "XSV"  == last_pos or "XSA" == last_pos ):
                    #     if "XSN" == last_pos :
                    #         restmp.append([word,"XSN"])
                    #     elif "XSV" == last_pos:
                    #         restmp.append([word,"XSV"])
                    #     elif "XSA" == last_pos :
                    #         restmp.append([word,"XSA"])
                    # else:
                    
                        biocheck = sorted(biocheck.items(),key=lambda x: x[1],reverse=True)
                        if biocheck[0][0] != "Fail":
                            restmp.append([word,biocheck[0][0]])
                        elif len(biocheck) >= 2:
                            restmp.append([word,biocheck[1][0]])
                        else:
                            restmp.append([word,"Fail"])
            else:
                # biocheck = max(biocheck.items(),key=lambda x: x[1])
                # restmp.append([word,biocheck[0]])
                
                biocheck = sorted(biocheck.items(),key=lambda x: x[1],reverse=True)
                if biocheck[0][0] != "Fail":
                    restmp.append([word,biocheck[0][0]])
                elif len(biocheck) >= 2:
                    restmp.append([word,biocheck[1][0]])
                else:
                    restmp.append([word,"Fail"])
                # if "NOUN" in biocheck and ("VV" == last_pos or "NVV" == last_pos  or "VA" == last_pos  or "NVA" == last_pos ):
                #     if "VV" == last_pos or "NVV" == last_pos :
                #         restmp.append([word,"NVV"])
                #     elif "VA" == last_pos  or "NVA" == last_pos:
                #         restmp.append([word,"NVA"])
                # elif "NOUN" in biocheck and ("XSN"  == last_pos  or "XSV"  == last_pos or "XSA" == last_pos ):
                #     if "XSN" == last_pos :
                #         restmp.append([word,"XSN"])
                #     elif "XSV" == last_pos:
                #         restmp.append([word,"XSV"])
                #     elif "XSA" == last_pos :
                #         restmp.append([word,"XSA"])
                # else:
                #     biocheck = max(biocheck.items(),key=lambda x: x[1])
                #     restmp.append([word,biocheck[0]])
                restmp.append([word,"Fail"])
        # print(restmp)
        htjoin = []
        for idx,rr in enumerate(restmp):
            if rr[0] == "+":
                htjoin.append("+")
            else:
                if "[UNK]" in rr[0]:
                    sent = sent.strip()

                    sent_ = sent.split()
                    for i in range(len(sent_)):
                        sent_[i] = sent_[i].strip("+")
                    sent_ = (" ".join(sent_)).replace("+"," + ").split()
                    
                    try:
                        rr[0] = sent_[idx]
                    except Exception as ex:
                        import traceback
                        traceback.print_exception()
                        exit()
                
                if re.search(r"[a-z|A-Z]",rr[0]) and not re.search(r"[^a-z|A-Z]",rr[0]):
                    # print(rr)
                    rr[1] = "SL"
                elif re.search(r"\d+",rr[0]):
                    rr[1] = "SN"
                elif re.search(r"[^a-zA-Z가-힣\d]",rr[0]):
                    rr[1] = "SF"
                ht = "/".join(rr)
                # if origin:
                httmp = copy.deepcopy(ht)
                ht = get_xstags(ht,orimode=origin)
                # print(1,ht)
                # ht = "/".join(ht)
                # print(ht)
                try:
                    if ht.split("+")[0].endswith("/NOUN"):
                        ht = ht.split("+")
                        ht[0] = ht[0].split("/")
                        if len(ht[0][0]) > 2 and len(ht[0][0]) <= 15:
                            ht[0][0] = cnoun(ht[0][0]).replace(" ","_")
                        ht[0] = "/".join(ht[0])
                        ht = "+".join(ht)
                except Exception as ex:
                    import traceback
                    traceback.print_exc()
                    print(ex)
                    print(1,ht,httmp)
                    exit()
                htjoin.append(ht)

        restmp = " ".join(htjoin)
        restmp = re.sub(r" +"," ",restmp)
        restmp = restmp.replace(" + ","+")

        Y.append(restmp)
    return Y
# import copy
# def pos_predict_onnx(subwordvocab,xs,result,X,origin=False):
#     # result = np.array(result[0])
#     # print(result.shape)
#     result = np.array(result[0])
#     resulttmp = result.copy()
#     result = np.argmax(result,axis=-1)
#     Y = []
#     for sent,xx,mapping,yy in zip(xs,X["input_ids"],X["offset_mapping"],result):
#         xxtmp = copy.deepcopy(xx)
#         yytmp = copy.deepcopy(yy)
#         res = []
#         xy = xx[xx != subwordvocab.pad]
#         # print(xy.shape)
#         # print(yy.shape)
#         xx = xx[:xy.shape[0]]#.cpu().numpy().tolist()
#         yy = yy[:xy.shape[0]]#.cpu().numpy().tolist()
#         xx = deque(xx)
#         yy = deque(yy)
#         tmp = []
#         ptmp = []
#         c = 0
#         for offset in mapping:
#             if offset[0] == 0 and c!= 0:
#                 if tmp[-1] == subwordvocab.space and len(tmp) == 1:
#                     res.append([subwordvocab.to_mp([tmp])[0],subwordvocab.to_tag([ptmp])[0]])
#                     tmp = []
#                     ptmp = []
#                 else:
#                     res.append([subwordvocab.to_mp([tmp])[0],subwordvocab.to_tag([ptmp])[0]])
#                     tmp = []
#                     ptmp = []
#             if len(xx) == 0:
#                 break
#             c += 1
#             tk = xx.popleft()
#             p = yy.popleft()
#             tmp.append(tk)
#             ptmp.append(p)
#             if tk == subwordvocab.pad:
#                 break
#         restmp = []
#         for r in res[1:]:
#             word = subwordvocab.tokenizer.convert_tokens_to_string(r[0])
#             word = re.sub(r" +","",word)

#             if word == "[SEP]":
#                 break
#             if len(r[0]) == len(r[1]):
#                 biocheck = defaultdict(int)

#                 if r[0][0] == "▁":
#                     continue
#                 elif r[0][0] == "+":
#                     restmp.append(["+","+"])
#                     continue
                
#                 failflag = False
#                 last_pos = ""
#                 for idx,(t,p) in enumerate(zip(r[0],r[1])):
#                     if p.startswith("B_") and idx == 0:
#                         if p != "+" and p != "O":
#                             biocheck[p[2:]] += 1
#                         else:
#                             biocheck["Fail"] += 1
#                     elif idx == 0 and not p.startswith("B_"):
#                         failflag = True
#                         if p != "+" and p != "O":
#                             biocheck[p[2:]] += 1
#                         else:
#                             biocheck["Fail"] += 1
                        
#                     elif idx > 0 and not p.startswith("I_"):
#                         if p != "+" and p != "O":
#                             biocheck[p[2:]] += 1
#                         else:
#                             biocheck["Fail"] += 1
#                         failflag = True
#                     elif idx > 0 and p.startswith("I_"):
#                         if p != "+" and p != "O":
#                             biocheck[p[2:]] += 1
#                         else:
#                             biocheck["Fail"] += 1
#                     last_pos = p[2:]
#             # if True:   
#                 if len(biocheck) == 1:
#                     for k,v in biocheck.items():
#                         if v == len(r[0]):
#                             restmp.append([word,k])
#                 else:
#                     # if "NOUN" in biocheck and ("VV" == last_pos or "NVV" == last_pos  or "VA" == last_pos  or "NVA" == last_pos ):
#                     #     if "VV" == last_pos or "NVV" == last_pos :
#                     #         restmp.append([word,"NVV"])
#                     #     elif "VA" == last_pos  or "NVA" == last_pos:
#                     #         restmp.append([word,"NVA"])
#                     # elif "NOUN" in biocheck and ("XSN"  == last_pos  or "XSV"  == last_pos or "XSA" == last_pos ):
#                     #     if "XSN" == last_pos :
#                     #         restmp.append([word,"XSN"])
#                     #     elif "XSV" == last_pos:
#                     #         restmp.append([word,"XSV"])
#                     #     elif "XSA" == last_pos :
#                     #         restmp.append([word,"XSA"])
#                     # else:
                    
#                         biocheck = sorted(biocheck.items(),key=lambda x: x[1],reverse=True)
#                         if biocheck[0][0] != "Fail":
#                             restmp.append([word,biocheck[0][0]])
#                         elif len(biocheck) >= 2 and biocheck[1][0]:
#                             restmp.append([word,biocheck[1][0]])
#             else:
#                 # biocheck = max(biocheck.items(),key=lambda x: x[1])
#                 # restmp.append([word,biocheck[0]])
                
#                 biocheck = sorted(biocheck.items(),key=lambda x: x[1],reverse=True)
#                 if biocheck[0][0] != "Fail":
#                     restmp.append([word,biocheck[0][0]])
#                 elif len(biocheck) >= 2 and biocheck[1][0]:
#                     restmp.append([word,biocheck[1][0]])
#                 # if "NOUN" in biocheck and ("VV" == last_pos or "NVV" == last_pos  or "VA" == last_pos  or "NVA" == last_pos ):
#                 #     if "VV" == last_pos or "NVV" == last_pos :
#                 #         restmp.append([word,"NVV"])
#                 #     elif "VA" == last_pos  or "NVA" == last_pos:
#                 #         restmp.append([word,"NVA"])
#                 # elif "NOUN" in biocheck and ("XSN"  == last_pos  or "XSV"  == last_pos or "XSA" == last_pos ):
#                 #     if "XSN" == last_pos :
#                 #         restmp.append([word,"XSN"])
#                 #     elif "XSV" == last_pos:
#                 #         restmp.append([word,"XSV"])
#                 #     elif "XSA" == last_pos :
#                 #         restmp.append([word,"XSA"])
#                 # else:
#                 #     biocheck = max(biocheck.items(),key=lambda x: x[1])
#                 #     restmp.append([word,biocheck[0]])
#                 # restmp.append([word,"Fail"])
#         # print(restmp)
#         htjoin = []
#         for idx,rr in enumerate(restmp):
#             if rr[0] == "+":
#                 htjoin.append("+")
#             else:
#                 if "[UNK]" in rr[0]:
#                     sent = sent.strip()

#                     sent_ = sent.split()
#                     for i in range(len(sent_)):
#                         sent_[i] = sent_[i].strip("+")
#                     sent_ = (" ".join(sent_)).replace("+"," + ").split()
                    
#                     try:
#                         rr[0] = sent_[idx]
#                     except Exception as ex:
#                         import traceback
#                         traceback.print_exception()
#                         exit()
                
#                 if re.search(r"[a-z|A-Z]",rr[0]) and not re.search(r"[^a-z|A-Z]",rr[0]):
#                     # print(rr)
#                     rr[1] = "SL"
#                 elif re.search(r"\d+",rr[0]):
#                     rr[1] = "SN"
#                 elif re.search(r"[^a-zA-Z가-힣\d]",rr[0]):
#                     rr[1] = "SF"
#                 ht = "/".join(rr)
#                 if origin:
#                     ht = get_xstags(ht)
#                 # if ht.endswith("/NOUN"):
#                     # ht = ht.split("/")
#                     # if len(ht[0]) > 2 and len(ht[0]) <= 15:
#                     #     ht[0] = cnoun(ht[0]).replace(" ","_")
#                     # ht = "/".join(ht)
#                 htjoin.append(ht)

#         restmp = " ".join(htjoin)
#         restmp = re.sub(r" +"," ",restmp)
#         restmp = restmp.replace(" + ","+")

#         Y.append(restmp)
#     return Y


def make_input(x,subwordvocab,maxlength=600,type_="np"):
    xs = []
    for idx,xx in enumerate(x):
        xx = xx.strip().strip("+").strip()
        xs.append(xx)
    X = subwordvocab.make_intoken(xs,max_length=maxlength,type_=type_)
    return xs,X

class TokIOProcess (mp.Process):
    def __init__(self,index,x,model_path,model_path2,outqueue,origin,tkmode,posmode):
        super(TokIOProcess, self).__init__(daemon=True)
        self.sess = PickableInferenceSession(model_path)
        self.sess2 = PickableInferenceSession(model_path2)
        # self.sess.intra_op_num_threads = 0
        # self.sess2.sess_options.intra_op_num_threads = 0
        # self.res = []
        self.result = []
        self.x = x
        # self.inqueue = inqueue
        self.outqueue = outqueue
        self.index = index
        self.origin = origin
        self.tkmode = tkmode
        self.posmode = posmode
        # self.origin = origin
        
    def run(self):
        with open("postagger_model/ht_postagger_model3/subwordvocab.pkl","rb") as f:
            subwordvocab = pickle.load(f)
        # with open(f"{self.index}_pr.txt","w",encoding="utf-8") as tmpf:
        if True:
            while True:
                resultht = []
                x = self.x.get()
                xtmp = x
                if x == "[END]":
                    break
                for xx in x:
                    BI = []
                    x = xx
                    x_temp = xx
                    x_len = len(x)
                    x = w2i(x)
                    x = np.array(x,dtype=np.float32)
                    
                    if self.tkmode:
                        BI = np.array(BI)
                        input_name = self.sess.sess.get_inputs()[0].name
                        input_name2 = self.sess.sess.get_inputs()[1].name
                        output_name = self.sess.sess.get_outputs()[0].name
                        xt = x.tolist()
                        unfold = torch.tensor(xt)
                        unfold = unfold.unfold(1,3,1).detach().cpu().tolist()
                        unfold = np.array(unfold,dtype=np.float32)
                        y = self.sess.run(None, {input_name: x,input_name2:unfold})
                        # for yy in y:
                        # print(y)
                            # print(yy)
                        result = tok_predict_onnx(y,x_temp)
                        if not self.posmode:
                            resultht = result
                    # self.queue.put(result)
                    if self.posmode:
                        if not self.tkmode:
                            result = []
                            for xx in xtmp:
                                result.append(xx[0].replace("▁"," "))
                        # print(result[0])
                        # print(result)
                        xs,inputs = make_input(result,subwordvocab)
                        unfold = torch.tensor(inputs["input_ids"],dtype=torch.long)
                        unfold = unfold.unfold(1,3,1).detach().cpu().numpy().tolist()
                        input_name = self.sess2.sess.get_inputs()[0].name
                        input_name2 = self.sess2.sess.get_inputs()[1].name
                        output_name = self.sess2.sess.get_outputs()[0].name
                        
                        y = self.sess2.run(None, {input_name: np.array(inputs["input_ids"],dtype=np.float32),input_name2:np.array(unfold,dtype=np.float32)})
                        y = pos_predict_onnx(subwordvocab,xs,y,inputs,origin=self.origin)
                        for yy in y:
                            resultht.append(yy)
                        # tmpf.write(yy.strip()+"\n")
                # with open(f"{self.index}_ends.txt","w",encoding="utf-8") as tmpf:
                #     tmpf.write(f"{self.index}")
                self.outqueue.put((self.index,resultht))
        #     # print(res)
        #     self.result.append(res)
class HeadTail:
    def __init__(self,index,x,model_path,model_path2,queue,origin):
        self.sess = PickableInferenceSession(model_path)
        self.sess2 = PickableInferenceSession(model_path2)
        # self.res = []
        self.result = []
        self.x = x
        self.queue = queue
        self.index = index
        self.origin = origin
    def run(self,text):
        with open(os.path.join(os.environ["KJM"],"headtail","postagger_model/ht_postagger_model3/subwordvocab.pkl"),"rb") as f:
            subwordvocab = pickle.load(f)

        xx = text
        BI = []
        x = xx
        x_temp = xx
        x_len = len(x)
        x = w2i(x)
        x = np.array(x,dtype=np.float32)
        
        BI = np.array(BI)
        input_name = self.sess.sess.get_inputs()[0].name
        input_name2 = self.sess.sess.get_inputs()[1].name
        output_name = self.sess.sess.get_outputs()[0].name
        xt = x.tolist()
        unfold = torch.tensor(xt)
        unfold = unfold.unfold(1,3,1).detach().cpu().tolist()
        unfold = np.array(unfold,dtype=np.float32)
        y = self.sess.run(None, {input_name: x,input_name2:unfold})
        result = tok_predict_onnx(y,x_temp)
        # print(result)
        xs,inputs = make_input(result,subwordvocab)
        unfold = torch.tensor(inputs["input_ids"])
        unfold = unfold.unfold(1,3,1).detach().cpu().numpy().tolist()
        input_name = self.sess2.sess.get_inputs()[0].name
        input_name2 = self.sess2.sess.get_inputs()[1].name
        output_name = self.sess2.sess.get_outputs()[0].name
        
        y = self.sess2.run(None, {input_name: np.array(inputs["input_ids"],dtype=np.float32),input_name2:np.array(unfold,dtype=np.float32)})
        y = pos_predict_onnx(subwordvocab,xs,y,inputs,origin=self.origin)
        # print(y)
        return y                
root = __file__
root = root.split(os.sep)
root = os.sep.join(root[:-1])
# print(root)
os.environ['HT'] = root + os.sep + "postagger_model"
root = os.environ['HT']
# root = '.'
max_len = 360
with open(root+'/../tokenizer_model/lstm_vocab.pkl','rb') as f:
    vocab = pickle.load(f)
with open(root+'/../tokenizer_model/bigram_vocab.pkl','rb') as f:
    bigram = pickle.load(f)

def w2i(x_t):
    res = []
    if type(x_t) is str:
        x_t = [x_t]
    max_in_length = 0
    for i in x_t:
        x = i.replace('  ',' ').replace(' ','▁')
        max_in_length = max(max_in_length,len(x))
    for i in x_t:
        x = i.replace('  ',' ').replace(' ','▁')
        x = list(x)
        x = [vocab[e] if e in vocab else vocab['[UNK]'] for e in x ] 

        x = x
        if max_in_length < 4:
            max_in_length = 4
        x = x + [0] * (max_in_length-len(x))
        x = [vocab["[SOS]"]] + x + [vocab["[EOS]"]]
        # x = x[:max_len]
        res.append(x)

    x = np.array(res)

    return x

def tok_predict_onnx(result,x_temp):
    # print(result)
    # exit()
    result = np.array(result[0])
    # print(result.shape)
    # exit()
    resulttmp = result.copy()
    # result = np.exp(result)
    # print(result[0][1:,:])
    result[result >= 0.5] = 1
    result[result < 0.5] = 0
    result = result[:,1:]
    result = result.squeeze(axis=-1)
    # print(result.shape)
    # exit()
    # result = np.argmax(result,axis=-1)[:,1:]
    # print(result[:,1:])
    tagging = []
    for res,txt in zip(result,x_temp):

        txt = list(txt)
        for idx,res_ in enumerate(res):

            if (idx+1) > len(txt):

                break
            # print(res)
            if res_ == 1:
                txt[idx] = "+" + txt[idx]# + "+"

        txt = "".join(txt).replace("▁"," ")
        txttmp = []
        for txt_ in txt.split():
            if txt_.count("+") >= 2:
                txt_ = txt_.split("+")
                head = "".join(txt_[:-1])
                tail = txt_[-1]
                txt_ = head + "+" + tail
            txttmp.append(txt_)
        
        txt = " ".join(txttmp) 
        tagging.append(txt.replace("▁"," "))

    return tagging

if __name__ == '__main__':
    # with open("postagger_model/ht_postagger_model/subwordvocab.pkl","rb") as f:
    #     subwordvocab = pickle.load(f)
    x = ["나는 밥을 먹고 학교에 갔다 ." for _ in range(100)] 
    # make_input(x,subwordvocab)
    # exit()
    mp.set_start_method('spawn') # This is important and MUST be inside the name==main block.
    
    from datetime import datetime
    start = datetime.now()
    starttotal = start
    mps = []
    for i in range(10):
        q = mp.Queue()
        io_process = TokIOProcess(i,x,"tok-model-uint8.onnx","pos-model-uint8.onnx",q)
        io_process.start()
        mps.append([io_process,q])
    toks = []
    for io_process in mps:
        # print("start")
        starttmp = datetime.now()
        toks.append(io_process[1].get())
        print(datetime.now() - starttmp)

    for io_process in mps:
        io_process[0].join()
    
    # print(toks)
    print("tok",datetime.now()-start)
    print("------------------------------------")
    # start = datetime.now()
    # mps = []
    # poss = []
    # for tk in toks:
    #     q = mp.Queue()
    #     io_process = PosIOProcess(tk,"pos-model-uint8.onnx",q)
    #     io_process.start()
    #     mps.append([io_process,q])

    # for io_process in mps:
    #     # print("start")
    #     poss.append(io_process[1].get())
    # for io_process in mps:
    #     io_process[0].join()
    # print(poss[-1])
    # print("pos",datetime.now()-start)
    
    # print("------------------------------------")
    print("total",datetime.now()-starttotal)


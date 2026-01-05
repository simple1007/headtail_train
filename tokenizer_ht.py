import numpy as np
# import tensorflow as tf
import pickle
import re
import os
import math
import torch
# from train_bilstm_bigram_model_pytorch import TK_Model
from customutils import AC
userdic = ["어항","구피", "여과기","열대어","디스커스","베타","프론","먹이","코리","코리도라스","피라냐","입고"]
josa = ["은","는","이","가","을","를","의","에","에게","에서도","와","으","로","도","라","과","들"]
useraho = AC()

for userw in userdic:
    useraho.maketrie(userw)
useraho.constructfail()

import pickle
if os.path.exists("aho.pkl"):
    with open("aho.pkl","rb") as f:
        eomiaho = pickle.load(f)
        eomiaho.maketrie("중는")
        eomiaho.constructfail()
# user
import copy
def chkeomi(sent):
    return sent
    # print(sent)
    # for i in range(0,-len(sent))
    for i in range(len(sent)):
        # print(sent)
        if useraho.find(sent[i:len(sent)]):
            return sent
    if len(sent) < 3:
        return sent
    tmp = copy.deepcopy(sent)
    # print(tmp)
    sent = list(sent)
    sent = list(reversed(sent))
    sent_ = copy.deepcopy(sent)
    sent = "".join(sent)
    eomiidx = -1
    previdx = -1
    for i in range(len(sent)-1):
        eomi_ = sent[:i+1]
        # print(eomi_)
        # res = eomiaho.search(eomi_,dfs=True)
        # # print(res)
        # minv =[]
        # if len(res) == 0:
            # continue
        # minv = min(res,key=lambda x: len(x))
        # # print(minv)
        # if len(minv) > 2 or (len(minv) == 2 and len("".join(minv)) > 2):# len(eomi_):
        #     continue
        res = eomiaho.find(eomi_)
        # print(minv,eomi_)
        if res:#len(res) > 0:
            head = list(reversed(sent[i:]))
            
            head = "".join(head)
            ures = useraho.search(head,dfs=False)
            # print(ures,head)
            if len(ures) > 0:
                # print(ures)
                # print(head)
                # print(ures)
                mv = max(ures.keys())
                mv = max(ures[mv],key=lambda x: x[2])
                eomiidx = len(sent) - mv[2]#max(previdx,i)
                # print(mv,i,len(sent))
                # if mv[2] == len(sent) - i:
                    # print(i,len(sent)-i)
                break
                # if eomiidx == i:
                #     previdx = i
                # break
            # else:
            #     # print(res)
            #     eomiidx = i+1#max(previdx,i)
                # if len(sent) == 3 and i > 0:
                    # break
                # print(eomi_,i)
                # if eomiidx == i:
                    # previdx = i
                # previdx = i
            
    # sent = list(sent)
    if eomiidx > -1:
        # print(eomiidx)
        # if eomiidx == 0:
            # eomiidx = 1
        # print(eomiidx,sent_)
        sent_[eomiidx] = sent_[eomiidx] + "+"
    # print(sent_)
    # print(sent_)
    # print(eomiidx)
    sent_ = list(reversed(sent_))
    reseoj = "".join(sent_)
    # print(reseoj)
    return reseoj
# print(chkeomi("사진찍는"))
# print(chkeomi("신경써주시는"))
# print(chkeomi("피라냐는"))
# print(chkeomi("봤었지만"))
# print(chkeomi("햄치즈는"))
# print(chkeomi("할때는"))
# print(chkeomi("똥싸는"))
# print(chkeomi("배송오는"))
# print(chkeomi("밥주는중"))
# exit()
def userdic_chk(txt):
    return chkeomi(txt)
    # print(txt)
    txt = txt.strip()
    
    if txt == "": #and txt[-1] not in josa:
        return txt
    res = useraho.search(txt,dfs=False)
    if len(res) == 0:
        return txt
    maxkey = max(res.keys())
    # print(res[maxkey])
    maxv = max(res[maxkey],key=lambda x: x[2])
    # print(maxv)
    txt_ = txt.replace(maxv[0],maxv[0]+"+")
    txt_ = txt_.strip("+").strip()
    if txt_.split("+")[0] in josa:
        txt = txt_
    elif txt_.split("+")[0] not in josa:
        txt = txt_.replace("+"," ").strip()
    # elif txt_[-1]
        # print(txt_)
    # txt = txt_
    return txt    

def only_head_chk(key,sho):
    
    # print(key)
    maxkey = max(sho.keys())
    # print(res[maxkey])ㅔㄱ
    maxv = max(sho[maxkey],key=lambda x: x[2])
    # print(maxv)
    return key.endswith(maxv[0])
    # txt_ = txt.replace(maxv[0],maxv[0]+"+")
    # if txt_.split("+")[-1] in josa: 
    #     txt = txt_


# while True:
#     txt = input("입력: ")
#     print(userdic_chk(txt))
# model = TK_Model()
# model.load_state_dict(torch.load('tokenizer_model/model'))
# model = model.to(device)
root = os.environ['HT']
# root = '.'
max_len = 120
with open(root+'/../tokenizer_model/lstm_vocab.pkl','rb') as f:
    vocab = pickle.load(f)
with open(root+'/../tokenizer_model/bigram_vocab.pkl','rb') as f:
    bigram = pickle.load(f)

def w2i(x_t):
    res = []
    max_in_length = 0
    for i in x_t:
        x = i.replace('  ',' ').replace(' ','▁')
        max_in_length = max(max_in_length,len(x))
    if max_in_length > 400:
        max_in_length = 400
    for i in x_t:
        x = i.replace('  ',' ').replace(' ','▁')
        x = list(x)[:max_in_length]
        x = [vocab[e] if e in vocab else vocab['[UNK]'] for e in x ] 

        x = x

        x = x + [vocab["[PAD]"]] * (max_in_length-len(x))
        # x = x[:max_len]
        res.append(x)

    x = np.array(res)

    return x

def predict_onnx(onnx,x,device,verbose=0):
    BI = []
    x_temp = x
    x_len = len(x)
    x = w2i(x)
    x = np.array(x,dtype=np.float32)
    
    BI = np.array(BI)
    result = None
    
    input_name = onnx.get_inputs()[0].name
    output_name = onnx.get_outputs()[0].name
    
    result = onnx.run(None, {input_name: x})
    
    result = np.array(result[0])
    # print(result.shape)
    result = np.argmax(result,axis=-1)
    # print(result.shape)
    
    tagging = []
    for res,txt in zip(result,x_temp):
        txt = list(txt)
        for idx,res_ in enumerate(res):
            if (idx+1) > len(txt):
                break
            if res_ == 2:
                txt[idx] = "+" + txt[idx]
        txt = "".join(txt)
        line = []
        for txt_ in txt.split("▁"):
            if "+" not in txt_:
                txt_ = userdic_chk(txt_)
            
            txt_ = txt_.strip("+")
            if txt_.count("+") >= 2:
                tmp = txt_.split("+")
                head = "".join(tmp[:-1])
                tail=tmp[-1]
                headtail = head+"+"+tail
                line.append(headtail)
            else:
                line.append(txt_)
        
        tagging.append(" ".join(line).replace("▁"," "))
    # print(tagging)
    # exit()
    return tagging
        

def predict2(model,x,device,verbose=0):
    with torch.no_grad():
        BI = []
        x_temp = x
        x_len = len(x)
        # print(len(x_temp))
        # for l in x:
        #     bi_npy = []
        #     for i in range(len(l)-1):
        #         bi = l[i:i+2]
        #         if bi in bigram:
        #             bi_npy.append(bigram[bi])
        #         else:
        #             bi_npy.append(bigram['[UNK]'])
        #     # if len(l) == '':
        #     #     bi_npy.append(bigram['_'])
        #     bi_npy = bi_npy + [0] * (max_len - len(bi_npy))
        #     bi_npy = np.array(bi_npy[:max_len])
        #     BI.append(bi_npy)
        x = w2i(x)
        x = np.array(x)
        
        BI = np.array(BI)
        # print(type(BI),BI.shape)
        result = None
        
        # try:
        x = torch.tensor(x).to(dtype=torch.long,device=device)
        # BI = torch.tensor(BI).to(device)
        # result = model(x,BI)
        unfold = x.unfold(1,3,1)
        # with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        result = model(x,unfold)[0]#,BI)
        # result = torch.exp(result)
        resulttmp = result.clone().cpu().numpy()
        # result = torch.argmax(result,dim=-1)
        result[result >= 0.5] = 1
        result[result < 0.5] = 0
        # result_= result_.cpu().numpy()
        result = result.cpu().numpy()
        # print(result.shape,resulttmp.shape)
        # autospace = torch.exp(autospace)
        # autospace = torch.argmax(autospace,dim=-1)
        
        tagging = []
        for res,txt,prob in zip(result,x_temp,resulttmp):
            # print(len(txt),res.shape,prob.shape)
            # exit()
            txt = list(txt)
            for idx,res_ in enumerate(res):
                # if res_ == 3:
                    # break
                if (idx+1) > len(txt):
                    # print(idx)
                    break
                # print(idx,txt[idx],res_,auto_)
                # print(res_)
                if res_ == 1:
                    txt[idx] = "+" + txt[idx]# + "+"
                # if auto_ == 1:
                #     txt[idx] = txt[idx] + " "
            txt = "".join(txt)
            # if "구피는" in txt:
            #     print(txt)
            #     exit()
            line = []            # print(txt)
            # print(txt)
            # exit()
            for txt_ in txt.split("▁"):
                # print(txt_)
                # if "+" not in txt_ and "구피" in txt_ and txt_[-1] == "는":
                #     txt_ = txt_.replace("구피","구피+는").strip("+")
                # print(txt_)
                if "+" not in txt_:
                    
                    if txt_ == "구피는":
                        # txt_
                        # print(txt)
                        # exit()
                        tmpguppy = ""
                        tmpprob = []
                        print(len(txt),prob.shape)
                        # exit()
                        for t,prob_ in zip(txt.replace("▁"," "),prob):
                            
                            tmpguppy += t
                            tmpprob.append(prob_)
                            # print(tmpguppy)
                            # print(tmpguppy[-3:])
                            if tmpguppy[-3:] == "구피는":
                                print(tmpguppy[-3:],tmpprob[-3:])
                                exit()
                        # print(txt)
                        exit()
                    txt_ = userdic_chk(txt_)
                    # if txt__ != txt_: 
                    #     # print(txt_,txt__)
                    #     txt_ = txt__
                # elif txt_.split("+")[-1][0] not in josa:
                #     txt__ = txt_.replace("+","")
                #     sho = useraho.search(txt__,dfs=False)
                #     if len(sho) >= 1 and not only_head_chk(txt__,sho): 
                #         print(txt_,txt__)   
                #         txt_ = txt__
                txt_ = txt_.strip("+")
                if txt_.count("+") >= 2:
                    tmp = txt_.split("+")
                    head = "".join(tmp[:-1])
                    tail=tmp[-1]
                    headtail = head+"+"+tail
                    line.append(headtail)
                else:
                    line.append(txt_)
            # print()
            tagging.append(" ".join(line).replace("▁"," "))
        # print(tagging)
        return tagging
    
def predict3(model,x,device,verbose=0):
    with torch.no_grad():
        BI = []
        x_temp = x
        x_len = len(x)
        
        x = w2i(x)
        x = np.array(x)
        
        BI = np.array(BI)
        result = None

        x = torch.tensor(x).to(dtype=torch.long,device=device)

        unfold = x.unfold(1,3,1)
        # with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        result = model(x)[0]#,BI)

        result = result.squeeze()
        # print(result.shape)
        # exit()
        resulttmp = result.clone().cpu().numpy()
        result[result >= 0.5] = 1
        result[result < 0.5] = 0
        result = result.cpu().numpy()
        # print(result)
        tagging = []
        for res,txt in zip(result,x_temp):
            txt = list(txt)
            for idx,res_ in enumerate(res):
                if (idx+1) > len(txt):
                    # print(idx)
                    break
                if res_ == 1:
                    txt[idx] = "+" + txt[idx]# + "+"
            txt = "".join(txt)
            line = []            # print(txt)

            for txt_ in txt.split("▁"):
                if "+" not in txt_:
                    txt_ = userdic_chk(txt_)
                txt_ = txt_.strip("+")
                if txt_.count("+") >= 2:
                    tmp = txt_.split("+")
                    head = "".join(tmp[:-1])
                    tail=tmp[-1]
                    headtail = head+"+"+tail
                    line.append(headtail)
                else:
                    line.append(txt_)
            
            tagging.append(" ".join(line).replace("▁"," "))
        # print(tagging)
        return tagging
    

def predict4(model,x,device,verbose=0):
    with torch.no_grad():
        BI = []
        x_temp = x
        x_len = len(x)
        
        x = w2i(x)
        x = np.array(x)
        
        BI = np.array(BI)
        result = None

        x = torch.tensor(x).to(dtype=torch.long,device=device)

        unfold = x.unfold(1,3,1)
        # with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
        # result 
        pred = model(x,unfold)#,BI)
        result = pred[0]
        result = result.squeeze(-1)
        # print(result)
        resulttmp = result.clone().cpu().numpy()
        result[result >= 0.5] = 1
        result[result < 0.5] = 0
        result = result.cpu().numpy()

        resultauto = pred[1]
        # print(resultauto.shape)
        resultauto = resultauto.squeeze(-1)
        resultauto[resultauto >= 0.5] = 1
        resultauto[resultauto < 0.5] = 0
        resultauto = resultauto.cpu().numpy()

        # print(result)
        tagging = []
        for res,resauto,txt in zip(result,resultauto,x_temp):
            txt = list(txt)
            # print(res)
            for idx,(res_,resauto_) in enumerate(zip(res,resauto)):
                if (idx+1) > len(txt):
                    # print(idx)
                    break
                if res_ == 1:
                    txt[idx] = "+" + txt[idx]# + "+"
                if resauto_ == 1:
                    if idx > 0 and idx < len(res) - 1:
                        if txt[idx-1][-1] != " ":
                            txt[idx] = "▁" + txt[idx]
            txt = "".join(txt)
            line = []            # print(txt)

            for txt_ in txt.split("▁"):
                if "+" not in txt_:
                    txt_ = userdic_chk(txt_)
                txt_ = txt_.strip("+")
                if txt_.count("+") >= 2:
                    tmp = txt_.split("+")
                    head = "".join(tmp[:-1])
                    tail=tmp[-1]
                    headtail = head+"+"+tail
                    line.append(headtail)
                else:
                    line.append(txt_)
            
            tagging.append(" ".join(line).replace("▁"," "))
        # print(tagging)
        return tagging


def predict(model,x,device,verbose=0):
    with torch.no_grad():
        BI = []
        x_temp = x
        x_len = len(x)
        for l in x:
            bi_npy = []
            for i in range(len(l)-1):
                bi = l[i:i+2]
                if bi in bigram:
                    bi_npy.append(bigram[bi])
                else:
                    bi_npy.append(bigram['[UNK]'])
            # if len(l) == '':
            #     bi_npy.append(bigram['_'])
            bi_npy = bi_npy + [0] * (max_len - len(bi_npy))
            bi_npy = np.array(bi_npy[:max_len])
            BI.append(bi_npy)
        x = w2i(x)
        x = np.array(x)
        
        BI = np.array(BI)
        # print(type(BI),BI.shape)
        result = None
        # try:
        x = torch.tensor(x).to(device)
        BI = torch.tensor(BI).to(device)
        result = model(x,BI)
        result = torch.exp(result)
        _,result_ = torch.topk(result,1,dim=2)
        result_= result_.cpu().numpy()
        result = result.cpu().numpy()
        
        tagging = []
        
        # print(result_)
        for index,tag_ in enumerate(result_):
            x_te = list(x_temp[index])
            # print(tag_)
            # print(x_te)
            tag_prob = []
            # print(tag_)
            for index_temp,te in enumerate(tag_):
                if result[index][index_temp][te].any() <= 0.6 and te == 2:
                    continue
                    # te = "팅팅"
                # print(te)
                if te == 2 and index_temp <= len(x_te)-1:
                    x_te[index_temp] =  x_te[index_temp] + '&&' + str(result[index][index_temp][2]) + '&&' + '+'
                    # tag_prob.append([index_temp,result[index][index_temp]])
                elif te == 3 and index_temp < len(tag_)-1 and tag_[index_temp+1] == 3:
                    break
            x_te = ''.join(x_te).replace('▁',' ')
            # print(x_te)
            x_te = x_te.split(' ')
            temp_tok = []
            for ti,xttt in enumerate(x_te):
                if xttt.count('+') > 1:
                    # print((xttt))
                    # xttt = xttt.split("+")
                    # xttt = "".join(xttt[:-1]) + "+" + xttt[-1]
                    # continue
                    xttt = xttt.split('&&')
                    mem = []
                    mem_prob = []
                    max = 0
                    maxi = -1
                    # print(xttt)
                    for memi, xttt_ in enumerate(xttt):
                        if memi % 2 == 0:
                            mem.append(xttt_.strip('+'))
                            continue
                        else:
                            1 == 1
                        
                        xttt_ = xttt_.split('&&')
                        
                        if max < float(xttt_[0]):
                            max = float(xttt_[0])
                            maxi = memi - len(mem) + 1
                    
                    h = ''.join(mem[:maxi])#[0]
                    # t = ''.join(xttt[1:])
                    t = ''.join(mem[maxi:])
                    xttt = h+'+'+t
                # xttt = xttt.strip("+")
                temp_tok.append(xttt)

            x_te = ' '.join(temp_tok)
            x_te = re.sub(' +',' ',x_te)
            # print(x_te)
            x_te = re.sub('&&[0-9].[0-9]+&&','',x_te)
            # print(x_te)
            x_te = x_te.strip()
            tagging.append(x_te)
        # print(tagging)
    return tagging

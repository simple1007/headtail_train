import numpy as np
# import tensorflow as tf
import pickle
import re
import os
import math
import torch
# from train_bilstm_bigram_model_pytorch import TK_Model

# model = TK_Model()
# model.load_state_dict(torch.load('tokenizer_model/model'))
# model = model.to(device)
root = os.environ['HT']
# root = '.'
max_len = 230
with open(root+'/../tokenizer/lstm_vocab.pkl','rb') as f:
    vocab = pickle.load(f)
with open(root+'/../tokenizer/bigram_vocab.pkl','rb') as f:
    bigram = pickle.load(f)

def w2i(x_t):
    res = []
    for i in x_t:
        x = i.replace('  ',' ').replace(' ','▁')
        x = list(x)
        x = [vocab[e] if e in vocab else vocab['[UNK]'] for e in x ] 

        x = x

        x = x + [0] * (max_len-len(x))
        x = x[:max_len]
        res.append(x)

    x = np.array(res)

    return x

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

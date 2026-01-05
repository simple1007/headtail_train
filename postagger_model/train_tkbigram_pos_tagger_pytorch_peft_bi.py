# from transformers import TFBertForTokenClassification
# from tensorflow.keras.layers import LSTM,Input,Dropout, Bidirectional,Embedding,TimeDistributed,Dense
# from tensorflow.keras.models import Model
# from transformers import create_optimizer
# from transformers import TFBertModel
from einops._torch_specific import allow_ops_in_compiled_graph
allow_ops_in_compiled_graph()

import os
path = os.path.dirname(os.path.abspath("."))
import sys
import re
# sys.stdout.reconfigure(encoding='utf-8')
# print(sys.getdefaultencoding())

# print("한글")
# exit()
# print(path)
# exit()
import multiprocessing
from multiprocessing import Process
import pickle
import numpy as np
import datetime
import pickle
import os
import argparse
from collections import defaultdict
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import random_split
from torch.quantization import QuantStub, DeQuantStub, float_qparams_weight_only_qconfig, default_qconfig
import sys
from threading import Thread
sys.path.append(path)
from ht_utils import tagged_reduce, HTInputTokenizer,infer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import torch
import torch.nn as nn
# print(torch.cuda.get_device_properties(0).multi_processor_count)
# exit()
# torch.set_float32_matmul_precision('high')
# with open('kcc150_all_tag_dict.pkl','rb') as f:
#     tag_dict = pickle.load(f)
# #vocab_size = len(map_dict)
# tag_len = len(tag_dict.keys())
import copy

import pickle
from collections import deque,defaultdict
from tqdm import tqdm
from pos_vocab import PosVocab,BiPosVocab,TokenPosVocab#,tagged_reduce

def htdataset(args=None,worker=False):
    # for _ in range(1):
    X = []
    Y = []
    BI = []
    BIO = []
    batchcount = 0
    mps = []
    resX = []
    resY = []
    # print("worker",worker,mfn)
    # exit()
    # exit()
    countstag = defaultdict(int)
    print("workers start !!")
    inpath = open("mps.txt","w",encoding="utf-8")
    targetpath = open("tags.txt","w",encoding="utf-8")
    pbar = tqdm(total=(count_data+validation_step+1+20+1) * args.BATCH)
    with open(mfn,encoding="utf-8") as m, open(tfn,encoding="utf-8") as p:
        for i in range(count*args.BATCH):
        # for i in range((count_data+validation_step+1+20+1) * args.BATCH):
            pbar.update(1)
            mtk, mp = m.readline(),p.readline()
            if mtk.strip() == "" or mp.strip() == "":
                continue
            mp_ = mp.strip().strip("+").strip()
            mtk_ = mtk.strip().strip("+").strip()
            flag = True
            for mpp,mtkk in zip(mp_.replace(f"{htsep}"," _").split(),mtk_.replace(f"{htsep}"," ").split()):
                tag__ = tagged_reduce(mpp)
                # print(mpp)
                # print(tag__)
                # countstag[tag__] += 1
                # if "_" in tag__:
                #     print(tag__)
                if "B_"+tag__ not in subwordvocab.pos2index:# or tag__.startswith("UN"):
                    flag = False
                    # countstag[tag__] += 1
                    # print(subwordvocab.pos2index)
                    # print(tag__)
                    # print(mtk_)
                    # exit()
                    break
                else:
                    countstag[tag__] += 1
            if flag:
                # for mpp,mtkk in zip(mp_.replace(f"{htsep}"," ").split(),mtk_.replace(f"{htsep}"," _").split()):
                    # print(mp_.replace(htsep,"+"))
                    # print(mtk_.replace(htsep,"+"))
                    # print(mtkk,mpp,end=" ")
                # print()
                # print()
                inpath.write(mtk.strip()+"\n")
                targetpath.write(mp.strip()+"\n")
                X.append(mtk.strip().strip("+").strip())
                Y.append(mp.strip().strip("+").strip())
            # print(subword)
            
            if len(X) != 0 and len(X) % args.BATCH == 0:
                if not subword:
                    U,X,Y = posvocab.make_batch(X,Y)
                    # if not worker:
                    resX = X
                    resY = Y
                    resU = U
                    X=[]
                    Y=[]
                    U=[]
                    # yield torch.tensor(resU),torch.tensor(resX),torch.tensor([[0]]),torch.tensor([[0]]),torch.tensor(resY)
                else:
                    if not worker:
                        
                        X,Y = subwordvocab.maketoken(X,Y)
                        resX = X
                        resY = Y
                        X=[]
                        Y=[]
                        
                        # yield torch.tensor([[0]]),torch.tensor(resX),torch.tensor([[0]]),torch.tensor([[0]]),torch.tensor(resY)
                        X=[]
                        Y=[]
                    else:
                        # print(subwordvocab.pos2index)
                        # print(Y)
                        # exit()
                        resX.append(X)
                        resY.append(Y)
                        X = []
                        Y = []
                    
                    # else:
                        # subwordvocab_ = copy.deepcopy(subwordvocab)
                        # mp_ = Process(target=multiproc,args=(X,Y,subwordvocab_,batchcount,args))
                        # mps.append(mp_)
                        # batchcount += 1
                        # pbar.update(1)
                        # mp_.start()
                        # if len(mps) == 16:
                        # #     # print(mps)
                        #     for mps_ in mps:
                        #         mps_.start()
                        #     for mps_ in mps:
                        #         mps_.join()
                        #     mps = []
                        # X = []
                        # Y = []
                    
                
            # if len(X) != 0 and len(X) % args.BATCH == 0:
            #     X = []
            #     Y = []
            #     BIO = []
    # if len(mps) > 0:
    #     for mp in mps:
    #         # mp.join()
    #         mp.start()
    #     for mp in mps:
    #     # for mps_ in mps:
    #     #     mps_.start()
    #         mp.join()
    #     mps = []
    inpath.close()
    targetpath.close()
    print(countstag)
    print(subwordvocab.pos2index)
    # print(sum(countstag.values()) / len(countstag))
    # exit()
    if worker:
        '''for x,y in zip(tqdm(resX),resY):
            subwordvocab.maketoken(x,y,addindex=True)'''
        """
        step = len(resX) // 16
        compressnpz = {}
        
        for i in tqdm(range(0,len(resX),step)):
            multiproc(resX[i:i+step],resY[i:i+step],subwordvocab,i,args,compressnpz)
        np.savez_compressed(os.path.join(args.datapath,"postagger","data.npz"),**compressnpz)
        """
        
        argsmps = []
        step = len(resX) // 16#args.BATCH
        # print("\n",step,len(X),len(Y))
        print(len(resX),step)
        # exit()
        args_mp = []
        with multiprocessing.Pool(processes=16) as pool:
            for i in tqdm(range(0,len(resX),step)):
                subwordvocab_ = copy.deepcopy(subwordvocab)
                x = resX[i:i+step]
                y = resY[i:i+step]
                args_mp.append((x,y,subwordvocab_,i,args,{},))
                # print(i,step)
                # continue
                # mp = Process(target=multiproc,args=(x,y,subwordvocab_,i,args),daemon=True)
            result = pool.starmap(multiproc,args_mp)
        compressnpz = {}
        for res in result:
            for k,v in res.items():
                compressnpz[k] = v
        np.savez_compressed(os.path.join(args.datapath,"postagger","data.npz"),**compressnpz)
            # print(res)
                # argsmps.append(mp)
                # mp.start()
            # if len(argsmps) == 16:
        # for mp in argsmps:
        #     mp.join()
        # argsmps = []
        # exit()     
        # if len(argsmps) > 0:
        #     # for mp in argsmps:
        #     #     mp.start()   
        #     for mp in argsmps:
        #         mp.join()
        #     argsmps = []
    # pbar.close()

'''def htevaldataset():
    for _ in range(EPOCH):
        X = []
        Y = []
        BIO = []
        # pbar = tqdm(total=count_data+validation_step+1)
        with open(mfn,encoding="utf-8") as m, open(tfn,encoding="utf-8") as p:
            for i in range((count_data+validation_step+1) * args.BATCH):
                
                mtk, mp = m.readline(),p.readline()
                
            for i in range(10 * args.BATCH):
                mtk, mp = m.readline(),p.readline()
                
                if mtk.strip() == "" or mp.strip() == "":
                    continue
                X.append(mtk)
                Y.append(mp)
                
                if len(X) != 0 and len(X) % args.BATCH == 0:
                    if not subword:
                        U,X,Y = posvocab.make_batch(X,Y)  
                        yield torch.tensor(U),torch.tensor(X),torch.tensor([[0]]),torch.tensor([[0]]),torch.tensor(Y)
                    else:
                        # Process()
                        X,Y = subwordvocab.maketoken(X,Y)
                        yield torch.tensor([[0]]),torch.tensor(X),torch.tensor([[0]]),torch.tensor([[0]]),torch.tensor(Y)

                if len(X) != 0 and len(X) % args.BATCH == 0:
                    X = []
                    Y = []
                    BIO = []'''
def multiproc_wrap(X):
    return multiproc(X[0],X[1],X[2],X[3],X[4],X[5])
def multiproc(X,Y,subwordvocab,i,args,compressnpz):
    # print(args.datapath)
    name = i
    # print(name)
    # print(X,Y)
    for index,x in enumerate(tqdm(X)):
        # X,Y = subwordvocab.maketoken(X,Y)
        Xtmp = []
        # print(x)
        for xxx in x:
            xtmp = []
            # print(xxx)
            # print(x)
            # exit()
            xxx = xxx.replace("+","_").replace("/","")
            for xx in xxx.split():
                # if not xx.strip().startswith("♥"):
                
                xtmp.append(xx.strip())
            
            Xtmp.append(" ".join(xtmp))
        x = Xtmp
        y = Y[index]
        # print(x,y)
        # exit()
        x, y, bigram = subwordvocab.maketoken(x,y)
        # print(datapath)
        # torch.tensor(X),torch.tensor([[0]]),torch.tensor([[0]]),torch.tensor(Y)
        data = np.array(x)
        bi = np.array(bigram)
        y = np.array(y)
        # exit()
        # print(data)
        # np.save()
        # np.savez_compressed(os.path.join(args.datapath,"postagger",f"{name}_x.npy"),data)
        # np.savez_compressed(os.path.join(args.datapath,"postagger",f"{name}_y.npy"),y)
        # np.savez_compressed(os.path.join(args.datapath,"postagger",f"{name}_bi.npy"),bi)
        # np.save(os.path.join(args.datapath,"postagger",f"{name}_x.npy"),data)
        # np.save(os.path.join(args.datapath,"postagger",f"{name}_y.npy"),y)
        # np.save(os.path.join(args.datapath,"postagger",f"{name}_bi.npy"),bi)
        compressnpz[f"{name}_x.npy"] = data
        compressnpz[f"{name}_y.npy"] = y
        compressnpz[f"{name}_bi.npy"] = bi
        name += 1
        # print(bigram)
    return compressnpz
        

# compressdata = np.load(os.path.join(args.datapath,"postagger","data.npz"))
compressdata = None
def datasetload(datapath,index):
    global compressdata
    if not compressdata:
        compressdata = np.load(os.path.join(datapath,"data.npz"))
    return compressdata[f"{index}_x.npy"],compressdata[f"{index}_y.npy"],compressdata[f"{index}_bi.npy"]
def dataset():
    for _ in range(EPOCH):
        for i in range(count_data):
            masks = []
            segments = []
            
            data = np.load('kcc150_data/%05d_x.npy' % i)
            y = np.load('kcc150_data/%05d_y.npy' % i)
            # bi = np.load('kcc150_data/%05d_tk_bigram.npy' % i)
            lmx = []
            for d in data:
                d_ = [t for t in d if t != 1]
                mask = [1]*len(d_) + [0] * (max_len-len(d_))
                masks.append(mask)
                segment = [0] * max_len
                segments.append(segment)
            masks = np.array(masks)
            segments = np.array(segments)
            bi = torch.tensor([[0]])#torch.tensor(bi)
            data = torch.tensor(data)
            masks = torch.tensor(masks)
            segments = torch.tensor(segments)
            
            y = torch.tensor(y)
            
            yield bi,data,masks,segments,y

def validation():
    for _ in range(EPOCH):
        for i in range(count_data,count_data+validation_step):
            masks = []
            segments = []
            lmx = []
            data = np.load('kcc150_data/%05d_x.npy' % i)
            y = np.load('kcc150_data/%05d_y.npy' % i)
            # bi = np.load('kcc150_data/%05d_tk_bigram.npy' % i)
            
            for d in data:
                d_ = [t for t in d if t != 1]
                mask = [1]*len(d_) + [0] * (max_len-len(d_))
                masks.append(mask)
                segment = [0] * max_len
                segments.append(segment)
            masks = np.array(masks)
            segments = np.array(segments)

            bi = torch.tensor([[0]])#torch.tensor(bi)
            data = torch.tensor(data)
            masks = torch.tensor(masks)
            segments = torch.tensor(segments)
            y = torch.tensor(y)
            yield bi,data,masks,segments,y

clipcount = 0
def train(config,optimizer,loss,model,bi,x,y,device,subwordvocab,scaler,lr_scheduler):
    global clipcount
    model.train()
    
    bi = bi.to(device)
    x[0] = x[0].to(device)
    x[1] = x[1].to(device)
    x[2] = x[2].to(device)
    
    
    # bioy = y[1].to(device)
    y = y[0].to(device)
    # unfold = x[0].unfold(1,3,1)
    y_pred = model(x[0],bi)
    y_pred = y_pred.reshape(-1, tag_len)
    y = y.view(-1)
    y = y.type(torch.LongTensor).to(device)
    
    # ylstm = ylstm.view(-1,tag_len)
    loss_ = loss(y_pred,y)
    
    # loss_lstm = loss(ylstm,y)
    # y_pred_bio = y_pred_bio.view(-1,5)
    # bioy = bioy.view(-1)
    # loss_ += loss(y_pred_bio,bioy)
    
    loss_ = loss_ #- loss_lstm# / 2
    # loss_ = torch.abs(loss_)
    optimizer.zero_grad()
    # scaler.scale(loss_).backward()
    loss_.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(),config["max_grad_norm"])
    clipcount += 1
    # if clipcount % 500 == 0:
    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.5)
    optimizer.step()
    if lr_scheduler:
        lr_scheduler.step()
    
    return loss_.item()

def train_crf(config,optimizer,loss,model,bi,x,y,device):
    model.train()
    
    bi = bi.to(device)
    x[0] = x[0].to(device)
    x[1] = x[1].to(device)
    x[2] = x[2].to(device)
    
    
    bioy = y[1].to(device)
    y = y[0].to(device)
    
    y_pred,y_pred_bio,logloss,loglossbio = model(bi,x[0],x[1],x[2],y=y,ybio=bioy)
    # y_pred = y_pred.view(-1, tag_len)
    # y = y.view(-1)
    # y = y.type(torch.LongTensor).to(device)
    
    # loss_ = loss(y_pred,y)
    
    # y_pred_bio = y_pred_bio.view(-1,5)
    # bioy = bioy.view(-1)
    # loss_ += loss(y_pred_bio,bioy)
    loss_ = logloss + loglossbio
    loss_ = loss_ / 2
    
    optimizer.zero_grad()
    loss_.backward()
    # torch.nn.utils.clip_grad_norm_(model.parameters(),config["max_grad_norm"])

    optimizer.step()
    
    return loss_.item()

def eval_crf(val_dataloader,device,model):
    with torch.no_grad():
        count = 0
        result = 0
        resultbio = 0
        print("what")
        # for x,y iin zip()
        for index in range(validation_step):
            bi_,data,masks,segments,y_ ,bio = next(val_dataloader)
            x_ = [data,masks,segments]
            # count+=1
            bi_ = bi_.to(device)
            
            x_[0] = x_[0].to(device)
            x_[1] = x_[1].to(device)
            x_[2] = x_[2].to(device)

            ybio_ = bio.to(device)            
            y_ = y_.to(device)
            y_ = y_.type(torch.LongTensor).to(device)
            
            _,_,yy,yy_bio = model(bi_,x_[0],x_[1],x_[2],crfenc=False)

            yy = yy.to(device)
            
            # _,yy = torch.topk(yy,k=1,dim=-1)
            # yy = yy.view(yy.shape[0],yy.shape[1])
            index_sep = (y_ == 2).nonzero()
            
            print(f"\r{index+1}/{validation_step}",end="")
            for r, c in index_sep:
                y__ = y_[r][:c]
                yy_ = yy[r][:c]
                comp = torch.eq(yy_,y__)
                count += c#y__.shape[0]
                comp = comp[comp==True].view(-1)
                result += (comp.shape[0]/c)
            
            # res = torch.argmax(yy_bio,dim=-1)#.view(-1)
            res = torch.tensor(yy_bio).to(device)
            res = res.view(ybio_.shape[0],ybio_.shape[1])
            res = (res == ybio_) & (res != 4)
            comp = ybio_ != 4
            res = torch.tensor(res[res == True],dtype=torch.long)
            resultbio += (torch.sum(res) / comp[comp==True].shape[0])
        print(result)
        print("avg",result/(validation_step*args.BATCH))
        print("bigavg",resultbio/validation_step)
from customutils import get_mp_tags
from ht_utils import tagged_reduce
# @torch.compile
def eval_bio(subwordvocab,postagger,X,Y,device,f1_tagindex=None):
    # print(X)
    # exit()
    poss = infer(subwordvocab,postagger,X,device)
    eojavg = 0
    tkavg = 0
    xtmp = []
    ytmp = []
    X_,Y_ = [],[]
    f1_pred = []
    f1_true = []
    for x,y in zip(poss,Y):
        # print(x,y)
        x = x.replace("+","+")
        y = y.replace("+","+")
        # x = re.sub(" +"," ",x).strip()
        # y = re.sub(" +"," ",y).strip()
        # print(X[0],x,y)
        
        # exit()
        x = x.strip().strip("+").strip()
        y = y.strip().strip("+").strip()
        # print(x,y)
        # exit()
        # x = x.replace("hththththt","+")
        # x = x.replace("@@@","/")
        # y = y.replace("hththththt","+")
        eojcnt = 0
        tkcnt = 0
        eojlen = 0
        tklen = 0
        # if x != "":
        xtmp.append(x)
        ytmp.append(y)
        # if False:
            # if subwordvocab.pos2index:
            
        for xx,yy in zip(x.split(),y.split()):
            # print(xx)
            try:
                head, tail = get_mp_tags(xx)
            except:
                print(xx)
                print(x)
            if len(tail) > 0:
                # print(yy)
                yy = yy.split("+")
                yy[0] = tagged_reduce(yy[0]).strip().strip("_").strip()
                if f1_tagindex:
                    if head[1] in f1_tagindex:
                        f1_pred.append(f1_tagindex[head[1]])
                    else:
                        f1_pred.append(f1_tagindex["Fail"])
                    
                    if yy[0] in f1_tagindex:
                        f1_true.append(f1_tagindex[yy[0]])
                    else:
                        f1_true.append(f1_tagindex["Fail"])
                try:
                    yy[1] = tagged_reduce("_"+yy[1]).strip().strip("_").strip()
                    if f1_tagindex:
                        if tail[1] in f1_tagindex:
                            f1_pred.append(f1_tagindex[tail[1]])
                        else:
                            f1_pred.append(f1_tagindex["Fail"])
                        if yy[1] in f1_tagindex:
                            f1_true.append(f1_tagindex[yy[1]])
                        else:
                            f1_true.append(f1_tagindex["Fail"])
                except:
                    import traceback
                    traceback.print_exc()
                    for pt in x.split():
                        head, tail = get_mp_tags(pt)
                        # print("")
                        if len(tail) > 0:
                            print(head[0]+" "+head[1]+"+"+tail[0]+" "+tail[1],end=" ")
                        else:
                            print(head[0]+" "+head[1],end=" ")
                    print()
                    # print(x)
                    print(y)
                    print(yy,tail)
                    print(yy)
                    print(len(y.split()))
                    print(xx)
                    print(len(x.split()))
                    # exit()
                if False:
                    if "B_" +yy[0] not in subwordvocab.pos2index:
                        print(yy[0])
                    if "B_" +yy[1] not in subwordvocab.pos2index:
                        print(yy[1])
                yy = "+".join(yy)
                eojlen += 1
                tklen += 2
                if head[1] + "+" + tail[1] == yy:
                    eojcnt += 1
                if head[1] == yy.split("+")[0]:
                    tkcnt += 1
                if tail[1] == yy.split("+")[1]:
                    tkcnt+=1
            else:
                eojlen += 1
                tklen += 1
                yy = tagged_reduce(yy).strip().strip("_").strip()
                if f1_tagindex:
                    if head[1] in f1_tagindex:
                        f1_pred.append(f1_tagindex[head[1]])
                    else:
                        f1_pred.append(f1_tagindex["Fail"])
                    if yy in f1_tagindex:
                        f1_true.append(f1_tagindex[yy])
                    else:
                        f1_true.append(f1_tagindex["Fail"])
                if head[1] == yy:
                    eojcnt += 1
                    tkcnt += 1
                if False and "B_"+yy not in subwordvocab.pos2index:
                    print(yy)
        # print(X,Y)
        # print(tkcnt,tklen)
        # print(eojcnt,eojlen)
        # print(X,Y)
        # exit()
        tkavg += (tkcnt / (tklen+0))
        eojavg += (eojcnt / (eojlen+0))
    # exit()
    if f1_tagindex:
        return tkavg , eojavg,f1_pred,f1_true
    else:
        return tkavg, eojavg
def eval(val_dataloader,datapath,start,device,model,subwordvocab,bi,writer=None,loss=None,logstep=None,logname=None,batch=False):
    with torch.inference_mode():
        count = 0
        result = 0
        resultbio = 0
        loss__ = 0.0
        # print("what")
        # batch = False
        fail = []
        step = 10 if batch else validation_step
        # print()
        # print(start,step,validation_step)
        # print()
        for evalstart,index in enumerate(range(start,start+step)):
            # data = np.load(os.path.join(datapath,f"{index}_x.npy"),mmap_mode="r") 
            # y_ = np.load(os.path.join(datapath,f"{index}_y.npy"),mmap_mode="r")
            data,y_,bi_ = datasetload(datapath,index)
            # bi_,data,masks,segments,y_  = next(val_dataloader)
            # bi_ = np.load(os.path.join(datapath,f"{index}_bi.npy"))#torch.tensor([[0]])
            bi_ = torch.tensor(bi_)
            data = torch.tensor(data)
            y_ = torch.tensor(y_)
            x_ = [data,torch.tensor([[0]]),torch.tensor([[0]])]
            # count+=1
            bi_ = bi_.to(device)
            
            x_[0] = x_[0].to(device)
            x_[1] = x_[1].to(device)
            x_[2] = x_[2].to(device)

            # ybio_ = bio.to(device)            
            y_ = y_.to(device)
            y_ = y_.type(torch.LongTensor).to(device)
            # unfold = x_[0].unfold(1,3,1)
            yy = model(x_[0],bi_)

            yy = yy.to(device)
            if loss:
                loss_ = loss(yy.reshape(-1,tag_len),y_.view(-1))
                loss_.requires_grad = False
                loss_ = loss_.item()
                loss__ += loss_
            _,yy = torch.topk(yy,k=1,dim=-1)
            yy = yy.reshape(yy.shape[0],yy.shape[1])
            # index_sep = (y_ == 2).nonzero()
            
            # print(f"\r{evalstart+1}/{step}",end="")
            sptoks = (y_ != subwordvocab.pos2index["[PAD]"]) #& (y_ != subwordvocab.pos2index["O"]) & (y_ != subwordvocab.pos2index["+"]) & (y_ != subwordvocab.pos2index["[SOS]"]) & (y_ != subwordvocab.pos2index["[EOS]"])
            correct = y_[sptoks] == yy[sptoks]
            incorrect = y_[sptoks] != yy[sptoks]
            # print(incorrect.shape,correct.shape)       
            # exit()
            # print(incorrect.shape,y_[sptoks][incorrect].shape)
            # exit()
            # print(incorrect.shape)
            # print(incorrect.shape)
            # yfail = y_.view(-1)
            # print(yfail.shape)
            for fail_ in subwordvocab.to_tag([y_[sptoks][incorrect].cpu().numpy().tolist()]):
                fail = fail + fail_
            result += torch.sum(correct) / y_[sptoks].reshape(-1).shape[0]
            # for r, c in index_sep:
            #     y__ = y_[r][:c]
            #     yy_ = yy[r][:c]
            #     sptoks = ( y__ != 3) & (y__ != 4) & (y__ != 0)
            #     # print(yy_[sptoks].shape,yy_.shape)
            #     # exit()
            #     comp = torch.eq(yy_[sptoks],y__[sptoks])
            #     count += c#y__.shape[0]
            #     comp = comp[comp==True].view(-1)
            #     result += (comp.shape[0]/yy_[sptoks].shape[0])
            
            # res = torch.argmax(yy_bio,dim=-1)#.view(-1)
            # res = res.view(ybio_.shape[0],ybio_.shape[1])
            # comp = (ybio_ != 4) & (ybio_ != 2)
            # res = res[comp] == ybio_[comp] #& ((res != 4) & (res != 2))
            
            # res = res
            # resultbio += torch.sum(res) / ybio_[comp].view(-1).shape[0]
            # res = res[res == True].clone().detach().requires_grad_(False).type(torch.LongTensor).to(device)
            # resultbio += (torch.sum(res) / comp[comp==True].shape[0])
        # print(result)
        from collections import Counter
        c = Counter(fail)
        c = sorted(c.items(),key=lambda x: x[1],reverse=True)
        # print(c[:10])
        # print("\navg",(result/step).item())
        
        acc = (result/step).item()
        loss__ = loss__/step
        # print(acc,loss__)
        if writer:
            writer.add_scalar(f"Loss/{logname}",loss__,logstep)
            writer.add_scalar(f"Accuracy/{logname}",acc,logstep)
            # writer.flush()
        else:
            print()
            print("avg:",acc,"loss:",loss__)
        # print("bigavg",resultbio/step)
if __name__ == "__main__":
    base = "torch_base_modu_kiwi_cnoun"
    usemodel2 = False
    from tqdm import tqdm
    if usemodel2:
        from model2 import PosLSTMMini2 as PosLSTM,remove_prefix_from_state_dict
    else:
        from model import PosLSTMMini2 as PosLSTM,remove_prefix_from_state_dict
    parser = argparse.ArgumentParser(description="Postagger")

    parser.add_argument("--MAX_LEN",type=int,help="MAX Sequnce Length",default=450)
    parser.add_argument("--BATCH",type=int,help="BATCH Size",default=256)
    parser.add_argument("--EPOCH",type=int,help="EPOCH Size",default=5)
    parser.add_argument("--epoch_step",type=int,help="Train Data Epoch Step",default=4000)
    parser.add_argument("--validation_step",type=int,help="Validation Data Epoch Step",default=240)
    # parser.add_argument("--hidden_state",type=int,help="BiLstm Hidden State",default=tag_len*2)
    parser.add_argument("--GPU_NUM",type=str,help="Train GPU NUM",default="0")
    parser.add_argument("--model_name",type=str,help="Tokenizer Model Name",default="tkbigram_one_first_alltag_bert_tagger.model")
    # parser.add_argument("--infer",help="Postagger Inference Mode",action="store_true")
    parser.add_argument("--prepro_flag",help="Postagger Inference Mode",action="store_true")
    parser.add_argument("--subword",help="Postagger Inference Mode",action="store_true")
    parser.add_argument("--infer",help="Postagger Inference Mode",action="store_true") 
    parser.add_argument("--eval_step",type=int,help="Evaluation Step of batch train",default=500) 
    parser.add_argument("--model_path",type=str,help="model save path",default=base)
    parser.add_argument("--lr",type=float,help="Learning rate",default=1e-4)
    parser.add_argument("--sepoch",type=int,help="Learning rate",default=0)
    parser.add_argument("--mincount",type=int,help="postag minimum count",default=0)
    args = parser.parse_args()
    
    base = args.model_path

    EPOCH = args.EPOCH
    count_data = args.epoch_step#4000
    validation_step = args.validation_step
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    max_len = args.MAX_LEN

    tags_tmp = defaultdict(int)
    count = 0
    tag_dict_ = defaultdict(int)
    doc_num = 350000
    datapath = os.path.join(os.environ["DATASET"],"ht_dataset","modu","result")
    if args.prepro_flag:
        mfn = os.path.join(datapath,"delkiwimorphs.txt")
        tfn = os.path.join(datapath,"delkiwitags.txt")
    else:
        mfn = "mps.txt"
        tfn = "tags.txt"
    args.datapath = datapath
    eval_step = args.eval_step
    prepro_flag = args.prepro_flag#True
    subword = args.subword#False
    mincount = args.mincount
    posvocab = BiPosVocab(400)#PosVocab(max_len)
    subwordvocab = HTInputTokenizer()
    # length = []
    total = count
    inpath = open(mfn,encoding="utf-8")
    count = len(inpath.readlines())
    inpath.close()
    total = count
    print(f"doclength: {total}")
    # print(count)
    # print(111,count)
    # mfn.seek(0)
    # inpath.close()
    # count = 600000#93694#3000000
    count = count // args.BATCH
    count_data = int(count * 0.9)
    print(count_data)
    
    validation_step = int(count * 0.05)
    print(validation_step)
    xf = open(mfn,encoding="utf-8")
    yf = open(tfn,encoding="utf-8")
    biox = []
    bioy = []
    htsep = "hththththt"
    bioX = []
    bioY = []
    # print(count_data)
    # print(validation_step*args.BATCH)
    # print(validation_step)
    # print(validation_step*args.BATCH)
    # print(total)
    # exit()
    for _ in range((count_data+validation_step) * args.BATCH):
        xf.readline()
        yf.readline()
    for i in range((count_data+validation_step) * args.BATCH,(count_data+validation_step+200) * args.BATCH):
        x = xf.readline().strip().replace(htsep,"+")
        y = yf.readline().strip().replace(htsep,"+")
        # print(y)
        if x.strip() != "":
            biox.append(x)
            bioy.append(y)
        else:
            print(x)
            print(i)
        # print()
        # if x.strip() != "":
        #     # break
        #     print(x)
        #     print(y)
        #     print(i)
        # # else:
        #     # print()
        #     exit()
        if len(biox) == args.BATCH:
            bioX.append(biox)
            bioY.append(bioy)
            
            biox = []
            bioy = []
    print(len(bioX))
    # exit()
    worker_dataset = args.prepro_flag
    xf.close()
    yf.close()
    with open(tfn,encoding="utf-8") as f,open(mfn,encoding="utf-8") as mf, tqdm(total=8875652) as pbar:
        # res = open(mfn,encoding="utf-8")
        cnt = 0
        for l in f:
            m = mf.readline().strip()
            cnt += 1
            pbar.update(1)
            # m = res.readline()
            if prepro_flag:
                # m_ = m.split()
                ltmp = []
                mtmp = []
                # l = l.repalce
                for ll,mm in zip(l.split(),m.split()):
                    ll = ll.strip(htsep)
                    ltmp.append(ll)
                    
                    mm = mm.strip(htsep)
                    mtmp.append(mm)
                l = " ".join(ltmp)
                l = l.replace(htsep,"+")
                
                m = " ".join(mtmp)
                m = m.replace("+","").replace("/","")
                m = m.replace(htsep,"+")
                for ll,mm in zip(l.split(),m.split()):
                    # ll 
                    ll = ll.replace("+","+_")
                    tag_ = ll.split("+")
                    
                    # mm = mm.replace("+","+_")
                    mps = mm.split("+")
                    # print(ll,tag_)
                    # exit()
                    
                    # mm_ = m_.pop(0)
                    c = 0
                    for idx,(t,m_) in enumerate(zip(tag_,mps)):
                        if c >= 1:
                            # t = tagged_reduce("_"+t.strip("_"),word=m_)
                            tags_tmp[t] += 1
                            
                            # print(t)
                        elif c == 0:
                            t = t.strip()
                            # t = tagged_reduce(t.strip("_"),word=m_)
                            tags_tmp[t] += 1
                            c += 1
                        
        # exit()
                
        if args.prepro_flag:
            length = []
            # print(len(tags_tmp))
            for k,v in tags_tmp.items():
                length.append(v)
                # if k.startswith("_"):
                #     print(k,v)
                if v > mincount:# and not k.startswith("UN"):
                    
                    k = tagged_reduce(k).strip("_")
                    tag_dict_[k] += v
                    
                # else:
                #     print(1111,k,v)
            print(tag_dict_)
            print(sum(length)/len(length))
            # exit()
        if args.prepro_flag:
            with open("tags_count.pkl","wb") as f:
                pickle.dump(tag_dict_,f)
            
            del tags_tmp
        else:
            with open("tags_count.pkl","rb") as f:
                tag_dict_ = pickle.load(f)
# exit()

    # exit()
    if args.prepro_flag:
        with open(mfn,encoding="utf-8") as m, open(tfn,encoding="utf-8") as p:
            # p.seek(0)
            stopcnt = 0
            from tqdm import tqdm
            for mtk,mp in zip(m,p):#tqdm(range((count_data+validation_step) * args.BATCH)):
                # print(m.readline(),p.readline())
                # continue
                # mtk, mp = m.readline(),p.readline()
                # length.append(len(mtk.strip()))
                stopcnt += 1
                if not args.subword:
                    posvocab.make_dict(mtk,mp,tags=tag_dict_)
                else:
                    if mincount == 0:
                        tag_dict_ = None
                    mptmp = []
                    for mpp in mp.split():
                        mpp = mpp.strip(htsep)
                        mptmp.append(mpp)
                    mp = " ".join(mptmp)
                    mp = mp.replace(htsep,"+")
                    # print(mp)
                    
                    subwordvocab.train_pos(mp,tags=tag_dict_,words=mtk)
                    subwordvocab.get_bigram([mtk])
                # if stopcnt == (count + validation_step) * args.BATCH:
                #     break
            # print(f"bigramcount:{subwordvocab.bigramcount}")
            cnt = 0
            sumv = 0
            for k,v in subwordvocab.bigramcount.items():
                if v > 55:#80:
                    sumv += v
                    # print(f"{k}={v}")
                    subwordvocab.bigram2index[k] = subwordvocab.bigramindex
                    subwordvocab.index2bigram[subwordvocab.bigramindex] = k
                    subwordvocab.bigramindex += 1
                    cnt += 1
            subwordvocab.bigramcount = {}
            # print(len(subwordvocab.bigram2index))
            # print(cnt)
            # print(sumv/cnt)
            # # print(f"avg bigramcount:{sum(subwordvocab.bigramcount.values())/len(subwordvocab.bigramcount)}")
            # exit()
                    # break
        if not args.subword:            
            with open("posvocab.pkl","wb") as f:  
                pickle.dump(posvocab,f)
        else:
            # print(subwordvocab.pos2index.keys())
            with open("subwordvocab.pkl","wb") as f:  
                pickle.dump(subwordvocab,f)        
            # print("dumpsubword")
    else:
        if not args.subword:
            with open("posvocab.pkl","rb") as f:
                posvocab = pickle.load(f)
        else:
            with open("subwordvocab.pkl","rb") as f:
                subwordvocab = pickle.load(f)

    if not args.subword:
        # print("asbccc",len(posvocab.pos2index))
        # exit()

        tag_len = len(posvocab.index2pos)
    else:
        print("tagcount:",len(subwordvocab.pos2index))
        # exit()

        tag_len = len(subwordvocab.index2pos)
        # print(tag_len,len(subwordvocab.pos2index))
        # print(subwordvocab.sep)
        
        # exit()
    # exit()
    if not args.subword: 
        postagger = PosLSTM(posvocab)
    else:
        postagger = PosLSTM(subwordvocab)
    s_epoch = args.sepoch
    postagger = postagger.to(device)
    # for n,p in postagger.named_parameters():
    #     # print(n)
    #     if n.startswith("model.encoders.") and ("bias" in n or "norm" in n):
    #         # print(n)
    #         p.requires_grad=False
    # exit()
    # postagger = torch.compile(postagger)
    if not args.infer:
        # postagger = torch.compile(postagger, mode='max-autotune')
        # postagger = torch.compile(postagger)
        True
    if not args.infer and s_epoch > 0:
        checkpoint = torch.load('{}/model_{}'.format(args.model_path,s_epoch),weights_only=True)
        state_dict = remove_prefix_from_state_dict(checkpoint["model"])
        postagger.load_state_dict(state_dict)
    
    
    # import torch._dynamo
    # out = torch._dynamo.explain(postagger)(torch.randint(12,(2,10)).to("cuda"),torch.randint(12,(2,10)).to("cuda"))
    # print(out)
    # prinnext(postagger.parameters()).device)
    # exit()
    # postagger = torch.compile(postagger, mode='max-autotune')
    if not args.infer:
        print(device)
        config = {"maxlen":420,"max_grad_norm":1,"epoch":10,"batch_size":50}
        
        loss = nn.CrossEntropyLoss(ignore_index=subwordvocab.pos2index["[PAD]"])#nn.NLLLoss(ignore_index=subwordvocab.pos2index["[PAD]"])
        
        # postagger.load_state_dict(torch.load(base+"/model",weights_only=True))
        optimizer = torch.optim.AdamW(postagger.parameters(),lr=args.lr)
        # optimizer = torch.optim.Adagrad(postagger.parameters(),lr=args.lr)
        warmup_step = int( count_data * 0.5 )
        if s_epoch > 0:
            optimizer.load_state_dict(checkpoint["optimizer"])
            loss.load_state_dict(checkpoint["loss"])
        
        from transformers import get_linear_schedule_with_warmup
        import torch.optim as optim

        # Assuming you have an optimizer and model defined
        # optimizer = optim.AdamW(model.parameters(), lr=1e-5)

        num_warmup_steps = int((count_data*EPOCH)*0.05)#500  # Number of steps for linear warmup
        num_training_steps = count_data * EPOCH#10000 # Total number of training steps

        lr_scheduler = None
        # get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=num_warmup_steps,
        #     num_training_steps=num_training_steps
        # )
        # n_data = htdataset()
        from datetime import datetime
        from tqdm import tqdm
    
        datapath = os.path.join(os.environ["DATASET"],"ht_dataset","modu","result","postagger")
    
        if worker_dataset:
            # proc = Process(target=multiproc,args=(n_data))
            if not args.subword:
                n_data = htdataset(args=args,worker=False)
                for i in tqdm(range(count_data+validation_step+1+10+1)):
                    bi,data,masks,segments,y = next(n_data) 
                    data = data.cpu().numpy()
                    y = y.cpu().numpy()
                    
                    np.save(os.path.join(datapath,f"{i}_x.npy"),data)
                    np.save(os.path.join(datapath,f"{i}_y.npy"),y)
            else:
                print("HI")
                htdataset(args=args,worker=True)
                exit()
            # exit()
        import torch
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
        traincount = 0
        evalcount = 0
        if args.sepoch > 0:
            traincount = (args.sepoch * count_data) // 100
            evalcount = args.sepoch * (count_data // args.eval_step)
        # with # model.load_state_dict(torch.load("selfmodel_13",weights_only=True))
        # scaler = torch.amp.grad_scaler()
        scaler = torch.amp.GradScaler("cuda")
        # if not args.subword: 
        #     postagger = PosLSTM(posvocab)
        # else:
        #     postagger = PosLSTM(subwordvocab)
        # s_epoch = args.sepoch
        # postagger = postagger.to(device)
        # with torch.amp.autocast("cuda",dtype=torch.bfloat16):
        if True:
            postagger.train()
            for e in range(s_epoch,s_epoch+EPOCH):
                running_loss = 0
                
                start = datetime.now()
                
                # n_data = htdataset()
                
                for index in range(count_data):
                    try:
                        start_b = datetime.now()
                        # bi,data,masks,segments,y = next(n_data)
                        # print(index)
                        # data = np.load(os.path.join(datapath,f"{index}_x.npy"),mmap_mode="r")
                        # y = np.load(os.path.join(datapath,f"{index}_y.npy"),mmap_mode="r")
                        data,y,bi = datasetload(datapath,index)
                        # bi = np.load(os.path.join(datapath,f"{index}_bi.npy"))
                        data = torch.tensor(data)
                        y = torch.tensor(y)
                        x = [data,torch.tensor([[0]]),torch.tensor([[0]])]
                        y = [y,[]]   
                        bi = torch.tensor(bi)#torch.tensor([[0]])
                        # bi = torch.tensor([[0]])
                        loss__ = train(config,optimizer,loss,postagger,bi,x,y,device,subwordvocab,scaler,lr_scheduler)
                        if (index + 1) % 15 == 0:
                            torch.cuda.empty_cache()
                        bcur = datetime.now()
                        cur = bcur - start
                        cur = str(cur).split(".")[0]
                        total = (bcur - start_b) * count_data
                        total = str(total).split(".")[0]
                        print("\r",str(e+1)+"/"+str(s_epoch+EPOCH),str(index+1)+"/"+str(count_data),f"{loss__:6f}",f"{cur}/{total}",end="")
                        running_loss += loss__
                        if (index+1) % 100 == 0:
                            writer.add_scalar("Loss/Train",loss__,traincount)
                            # print(lr_scheduler.get_last_lr()[-1])
                            if lr_scheduler:
                                writer.add_scalar("LR/Train",lr_scheduler.get_last_lr()[-1],traincount)
                            # writer.flush()
                            traincount += 1
                        # if (index+1) % 100 == 0:
                        #     torch.cuda.empty_cache()
                        if (index + 1) % args.eval_step == 0:
                            # print("\n-----------------")
                            # print(f"\n{index+1} batch loss:{loss__}")
                            postagger.eval()
                            # eval()
                            print()
                            acc = eval(None,datapath,count_data,device,postagger,subwordvocab,torch.tensor([]),writer=writer,loss=loss,logstep=evalcount,logname="Eval",batch=True)
                            evalcount += 1
                            # writer.add_scalar("Accuracy/Eval",acc,evalcount)
                            # writer.flush()
                            # print("-----------------")
                            resbiotk = 0
                            resbioeoj = 0
                            length = 0
                            evalcnt = 0
                            for bx,by in zip(tqdm(bioX[:30]),bioY[:30]):
                                # print(len(bx),len(by))
                                tkavg, eojavg = eval_bio(subwordvocab,postagger,bx,by,device)
                                resbiotk += tkavg
                                resbioeoj += eojavg
                                length += len(bx)
                                evalcnt += 1
                            # print()
                                
                                # if evalcnt == 100:
                                #     break
                            print()
                            # exit()
                            writer.add_scalar("Bio/TkAvg",resbiotk/length,evalcount)
                            writer.add_scalar("Bio/EojAvg",resbioeoj/length,evalcount)
                            # writer.flush()
                            postagger.train()
                    except Exception as ex:
                        print(ex)
                        import traceback
                        traceback.print_exc()
                        print()
                        print(index)
                        exit()
                # print()
                # print("\n\r",e,index,loss__,(running_loss / ((count_data))))
                # torch.cuda.empty_cache()
                postagger.eval()

                acc = eval(None,datapath,count_data,device,postagger,subwordvocab,torch.tensor([]),writer=writer,loss=loss,logstep=e,logname="Validation")
                resbiotk = 0
                resbioeoj = 0
                length = 0
                evalcnt = 0
                print()
                print("validation")
                for bx,by in zip(tqdm(bioX[30:]),bioY[30:]):
                    # print(len(bx))
                    tkavg, eojavg = eval_bio(subwordvocab,postagger,bx,by,device)
                    resbiotk += tkavg
                    resbioeoj += eojavg
                    length += len(bx)
                    # evalcnt += 1
                    # if evalcnt == 20:
                        # break
                print()
                writer.add_scalar("Bio/TkValAvg",resbiotk/length,e)
                writer.add_scalar("Bio/EojValAvg",resbioeoj/length,e)
                # writer.flush()
                postagger.train()
                if not os.path.exists(args.model_path):
                    os.makedirs(args.model_path)
                chkpoint = {
                    "model":postagger.state_dict(),
                    "loss":loss.state_dict(),
                    "optimizer":optimizer.state_dict()
                }
                torch.save(chkpoint, '{}/model_{}'.format(args.model_path,e+1))#{}'.format(e))
            
            chkpoint = {
                "model":postagger.state_dict(),
                "loss":loss.state_dict(),
                "optimizer":optimizer.state_dict()
            }
            torch.save(chkpoint, '{}/model'.format(args.model_path))
        # torch.save(optimizer.state_dict(), '{}/optimizer'.format(args.model_path))
        import pickle
        if not subword:
            with open(os.path.join(args.model_path,"posvocab.pkl"),"wb") as f:
                pickle.dump(posvocab,f)
        else:
            with open(os.path.join(args.model_path,"subwordvocab.pkl"),"wb") as f:
                pickle.dump(subwordvocab,f)
        writer.close()
    else:
        os.environ["HT"] = "C:\\Users\\ty341\\OneDrive\\Desktop\\kjm\\headtail\\postagger_model"
        # mfn = os.path.join(os.environ["KJM"],"jupyter","gemma2","sx_x.txt")
        # tfn = os.path.join(os.environ["KJM"],"jupyter","gemma2","sx_y.txt")
        # import tagging_lstm as tagging
        testi = open("test.input","w",encoding="utf-8")
        testo = open("test.output","w",encoding="utf-8")
        # f1_tagindex = {"NNP":0,
        #                "Josa":1,
        #                "VV":2,
        #                "Eomi":3,
        #                "XSV":4,
        #                "XSA":5,
        #                "VCP":6,
        #                "VX":7,
        #                "MM":8,
        #                "XSN":9,
        #                "VA":10,
        #                "NNB":11,
        #                "NP":12,
        #                "MAG":13,
        #                "MAJ":14,
        #                "SN":15,
        #                "VCN":16,
        #                "NR":17,
        #                "SL":18,
        #                "IC":19,
        #                "XPN":20,
        #                "XR":21,
        #                "Fail":0
        #             }
        f1_tagindex = {           
            'XSN':0, 
            'Josa':1, 
            'NNP':2, 
            'VA':3, 
            'Eomi':4, 
            'MM':5, 
            'VV':6, 
            'MAG':7, 
            'NNB':8, 
            'MAJ':9, 
            'VX':10, 
            'VCP':11, 
            'XSV':12, 
            'SN':13, 
            'SL':14, 
            'XSA':15, 
            'IC':16, 
            'NP':17, 
            'VCN':18, 
            'NR':19, 
            'UNC':20, 
            'UNT':21, 
            'XR':22, 
            'UNA_EF':23, 
            'XPN':24, 
            'UNA':25,
            'Fail':2
        }
        with open(mfn,encoding="utf-8") as m, open(tfn,encoding="utf-8") as t:
            X = []
            X_ = []
            Y = []
            # print(m,t)
            lcnt = 0
            batch_cnt = 0
            result_tkavg = 0
            result_eojavg = 0
            # postagger = torch.compile(postagger)
            # postagger = torch.compile(postagger, mode='max-autotune')
            if args.sepoch > 0:
                state_dict = torch.load('{}/model_{}'.format(args.model_path,args.sepoch),weights_only=True)["model"]
                postagger.load_state_dict(state_dict)
            else:
                state_dict = torch.load('{}/model{}'.format(args.model_path,args.sepoch),weights_only=True)["model"]
                postagger.load_state_dict(state_dict)
            postagger.eval()
            # print(2,count,validation_step,count * args.BATCH)
            f1_preds = []
            f1_trues = []
            count = len(m.readlines())
            m.seek(0)
            pbar = tqdm(total=count-((count_data+validation_step+200)*args.BATCH))
            with torch.no_grad():
                for mm,tt in zip(m,t):
                    mm = mm.strip()
                    tt = tt.strip()
                    lcnt += 1
                    # if lcnt == 15000:
                    #     break
                    if lcnt <= (count_data+validation_step+200) * args.BATCH: 
                        continue
                    pbar.update(1)
                    # mm = mm.strip()#.replace("+","_").replace("/","_")
                    # # mmtmp = []
                    # import re
                    mm = mm.replace(htsep,"+")
                    tt = tt.replace(htsep,"+")
                    
                    # import re
                    # if re.search(r"(name)(\d+)",mm):
                    #     continue
                    
                    # if re.search(r"(address)(\d+)",mm):
                    #     continue
                    # ehanflag = False
                    # for mmm in mm.split():
                    #     if (re.search(r"[가-힣]+",mmm) and re.search(r"[a-zA-Z]+",mmm)) or (re.search(r"[가-힣]+",mmm) and re.search(r"\d+",mmm)) or (re.search(r"[a-zA-Z]+",mmm) and re.search(r"\d+",mmm)):
                    #         ehanflag = True 
                    # if ehanflag:
                    #     continue                           
                    X.append(mm)
                    X_.append(mm.replace("+"," "))
                    Y.append(tt)
                    # print(mm)
                    # print(tt)
                    testi.write(mm+"\n")
                    # testi.write("<<EOD>>\n")
                    # testo.write("<<EOD>>\n")
                    testo.write(tt+"\n")
                    if len(X) % 100 >= 0 and len(X) != 0:
                        tkavg, eojavg,f1_pred,f1_true = eval_bio(subwordvocab,postagger,X,Y,device,f1_tagindex)
                        result_tkavg += tkavg
                        result_eojavg += eojavg
                        batch_cnt += len(X)
                        f1_preds += f1_pred
                        f1_trues += f1_true
                        X = []
                        Y = []
                        X_ = []
                    # if batch_cnt >= 50000:
                    #     break
            pbar.close()
        if len(X) > 0:
            tkavg, eojavg,f1_pred,f1_true = eval_bio(subwordvocab,postagger,X,Y,device,f1_tagindex)
            f1_preds += f1_pred
            f1_trues += f1_true
            result_tkavg += tkavg
            result_eojavg += eojavg
            batch_cnt += len(X)
        print("tkavg:",result_tkavg / batch_cnt)
        print("eojavg:",result_eojavg / batch_cnt)
        testi.close()
        testo.close()
        from sklearn.metrics import f1_score
        f1_micro = f1_score(f1_trues, f1_preds, average='micro')
        f1_macro = f1_score(f1_trues, f1_preds, average='macro')
        f1_weighted = f1_score(f1_trues, f1_preds, average='weighted')
        f1_per_class = f1_score(f1_trues, f1_preds, average=None)

        print(f"F1-score (micro): {f1_micro:3.4f}")
        print(f"F1-score (macro): {f1_macro:3.4f}")
        print(f"F1-score (weighted): {f1_weighted:3.4f}")
        print(f"F1-score per class: {f1_per_class}")
            #         if len(X) % 100 == 0 and len(X) != 0:
            #             # poss = tagging.predictbi_pt(postagger,X_,X,device)
            #             poss = infer(subwordvocab,postagger,X,device)
            #             # print(poss)
            #             # exit()
            #             batch_cnt += len(X)
            #             for iii,(xx,pr,tr) in enumerate(zip(X,poss,Y)):

            #                 pr = pr.split()

            #                 tr = tr.split()
            #                 # trt = tr
            #                 tkcorrect = 0
            #                 eojcorrect = 0
            #                 ys = []
            #                 for pr_ in pr:
            #                     tmp = []
            #                     # if pr_ == "+":
            #                     #     continue
            #                     for pr__ in pr_.split("+"):
            #                         # print(tr)
            #                         # print(pr)
            #                         tmp.append(pr__.split("/")[1])
            #                     ys.append("+".join(tmp))
            #                     # ys.append("+".join(tmp))
            #                 pr = " ".join(ys)
            #                 pr = pr.split()
                            
            #                 xx = xx.split()

            #                 tklength = 0
            #                 for ii,(xxx,ppr, ttr) in enumerate(zip(xx,pr,tr)):
            #                     # ppr
            #                     # tklen += 1
            #                     if ppr == ttr:
            #                         eojcorrect += 1
            #                     # print(ppr,ttr)
            #                     for xxxx,models,trues in zip(xxx.split("+"),ppr.split("+"),ttr.split("+")):
            #                         if models == trues:
            #                             tkcorrect += 1
            #                         # else:
            #                         elif False and models[0] == "S":
            #                             print("--------")
            #                             print(xx)
            #                             print(pr)
            #                             print(tr)
            #                             print(ppr)
            #                             print(ttr)
            #                             print(xxxx,models,trues)
            #                             # exit()
            #                         tklength += 1

            #                 result_tkavg += tkcorrect / tklength
            #                 result_eojavg += eojcorrect / len(tr)
            #                 # print(result_avg)
            #                 # exit()
            #             X = []
            #             X_ = []
            #             Y = []
                    
            #         # if batch_cnt >= 200000:
            #         #     break
            # # print()
            # print(result_tkavg)
            # print("tkavg",result_tkavg / batch_cnt)
            # print("eojavg",result_eojavg / batch_cnt)
            # print()

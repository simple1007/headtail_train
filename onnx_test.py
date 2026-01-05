from headtail import make_mp, file_analysis,tagged_reduce
from headtail_util import get_mp_tags
from datetime import datetime
from inferutils import make_input, PickableInferenceSession#, pos_predict_onnx
from inferutils import pos_predict_onnx,w2i,tok_predict_onnx
from tqdm import tqdm

import numpy as np
import pickle
import torch
import re


def predict_pos(x,sess2):
    xs,inputs = make_input(x,subwordvocab,maxlength=350,type_="np")
    unfold = torch.tensor(inputs["input_ids"].tolist())#.detach().clone().cpu().numpy().tolist())
    unfold = unfold.unfold(1,3,1).detach().clone().cpu().numpy().tolist()
    input_name = sess2.sess.get_inputs()[0].name
    input_name2 = sess2.sess.get_inputs()[1].name
    output_name = sess2.sess.get_outputs()[0].name
    
    y = sess2.run(None, {input_name: np.array(inputs["input_ids"],dtype=np.float32),input_name2:np.array(unfold,dtype=np.float32)})
    y = pos_predict_onnx(subwordvocab,xs,y,inputs,origin=False)
    # exit()
    return y

def predict_tok(x,sess):
    BI = []
    # x = xx
    x_temp = x
    x_len = len(x)
    x = w2i(x)
    x = np.array(x,dtype=np.float32)    

    BI = np.array(BI)
    input_name = sess.sess.get_inputs()[0].name
    input_name2 = sess.sess.get_inputs()[1].name
    output_name = sess.sess.get_outputs()[0].name
    xt = x.tolist()
    unfold = torch.tensor(xt)
    unfold = unfold.unfold(1,3,1).detach().cpu().tolist()
    unfold = np.array(unfold,dtype=np.float32)
    y = sess.run(None, {input_name: x,input_name2:unfold})
    y = tok_predict_onnx(y,x_temp)
    
    return y

if __name__ == "__main__":
    input_ = "postagger_model/test.input"
    output_ = "postagger_model/test.output"
    posinfer = False
    if posinfer:
        model_name = "super_resolution_pos.onnx"
        model_name = "pos-model-uint8.onnx"
        model_name = "pos-model-uint4.onnx"
        sess2 = PickableInferenceSession(model_name)
        with open("postagger_model/ht_postagger_model3/subwordvocab.pkl","rb") as f:
            subwordvocab = pickle.load(f)
        with open(input_,encoding="utf-8") as in_, open("test.label","w",encoding="utf-8") as out:
            length = len(in_.readlines())
            in_.seek(0)
            pbar = tqdm(total=length)
            for l in in_:
                l = l.strip()
                out.write(predict_pos([l],sess2)[0]+"\n")
                pbar.update(1)
            pbar.close()
            in_.seek(0)
            tmpX = in_.readlines()
        
        eoj = 0
        tk = 0
        linenum = 0
        with open(output_,encoding="utf-8") as true, open("test.label",encoding="utf-8") as pred:
            for idx,(pr,tr) in enumerate(zip(pred,true)):
                pr = pr.strip("+")

                prline = []
                pr = pr.strip()
                tr = tr.strip()

                eojtmp = 0
                eojlen = 0
                tktmp = 0
                tklen = 0
                
                for prr, trr in zip(pr.split(),tr.split()):
                    
                    try:
                        tklen += 1
                        if "+" in trr:
                            th,tt = trr.split("+")
                            th = tagged_reduce(th).strip()
                            tt = tagged_reduce("_"+tt).strip("_")
                        else:
                            th,tt = trr,""
                            th = tagged_reduce(th)

                        if prr == "+":
                            continue
                        ph,pt = get_mp_tags(prr)
                        if tt!="":
                            tklen += 1

                    except Exception as ex:
                        import traceback
                        traceback.print_exc()
                        exit()
                    # tklen += 1
                    if ph[1] == th:
                        tktmp += 1
                    
                    eojlen += 1
                    if tt != "":
                        if len(pt) == 0:
                            pt.append("UNK")
                            pt.append("Fail")
                        if ph[1]+"+"+pt[1] == th+"+"+tt:
                            
                        # print(prr,trr)
                            eojtmp += 1
                    else:
                        if ph[1] == th:
                            eojtmp += 1
                    if tt != "":
                        # tklen += 1
                        if pt[1] == tt:
                            tktmp += 1
                eoj += (eojtmp / eojlen)
                tk += (tktmp / tklen)
                linenum += 1
        print(linenum)
        print("eoj:",eoj/linenum)
        print("tk:",tk/linenum) 
    
    else:
        model_name = "super_resolution_tok.onnx"
        model_name = "tok-model-uint8.onnx"
        model_name = "tok-model-uint4.onnx"
        sess = PickableInferenceSession(model_name)
        with open(input_,encoding="utf-8") as in_, open("test.label","w",encoding="utf-8") as out:
            tmpX = in_.readlines()
            pbar = tqdm(total=len(tmpX))
            in_.seek(0)
            linenum = 0
            eoj = 0
            for l in in_:
                true = l.strip()
                line = l.strip().replace("+","")
                pred = predict_tok([line],sess)[0]
                linenum += 1
                eojtmp = 0
                eojlen = 0
                # print(pred,line)
                for prr,trr in zip(pred.split(),true.split()):
                    eojlen += 1
                    if prr == trr:
                        eojtmp += 1
            
                eoj += (eojtmp/eojlen)
                pbar.update(1)
            pbar.close()
            print("tokeoj:",eoj / linenum)
            
        
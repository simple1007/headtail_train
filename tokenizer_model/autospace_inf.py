# from tensorflow.keras.layers import LSTM, Input, Bidirectional, Embedding,TimeDistributed,Dense,Concatenate
# from tensorflow.keras import Model
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


import torch
import torch.nn as nn
import pickle
import numpy as np
# import tensorflow as tf
import os
import argparse
from transformer import Encoder,TransformerConfig
parser = argparse.ArgumentParser(description="Train Head-Tail Tokenizer")

parser.add_argument("--BATCH",type=int,help="Train Data BATCH SIZE",default=256)
parser.add_argument("--MAX_LEN",type=int,help="Sequence Data MAX LENGTH",default=500)
parser.add_argument("--epoch_step",type=int,help="Train Data Epoch Step",default=200000//3)
parser.add_argument("--validation_step",type=int,help="Validation Data Epoch Step",default=1000)
parser.add_argument("--EPOCH",type=int,help="Train Epoch SIZE",default=5)
# parser.add_argument("--BATCH",type=int,help="Train BATCH SIZE",default=50)
parser.add_argument("--model_name",type=str,help="Tokenizer Model Name",default="lstm_bigram_tokenizer.model")
parser.add_argument("--GPU_NUM",type=str,help="Train GPU NUM",default="0")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_NUM
max_len = args.MAX_LEN#300
EPOCH = args.EPOCH#6
BATCH_SIZE = args.BATCH#50
count_data = args.epoch_step#4000
validation_data = args.validation_step#400

count = 0
with open("httk_x.txt",'r',encoding='utf-8') as x_f:
    for l in x_f:
        count += 1
print(count)

with open('lstm_vocab.pkl','rb') as f:
    lstm_vocab = pickle.load(f)

tag_len = 3
        
class TK_Model2(nn.Module):
    def __init__(self):
        super(TK_Model2,self).__init__()
        self.config = TransformerConfig(pad_idx=0,numheads=4,dmodel=128, dff=512,nlayer=4,maxlen=max_len,vocab_size=len(lstm_vocab),dropout=0.1,warmup_steps=100)
        self.model = Encoder(self.config,last_layer=False).to(device)
        self.auto_output = torch.nn.Linear(self.config.dmodel,tag_len)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self,x):
        enc = self.model(x)
        output = self.auto_output(enc)
        output = self.softmax(output)
        return output

if __name__ == "__main__":
    import copy
    import re
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    print(device)
    model = TK_Model2()
    
    model.to(device)
    model.load_state_dict(torch.load('autospace_model/model'))
    model.eval()
    
    dataset = "C:\\Users\\ty341\\OneDrive\\Desktop\\kjm\\jupyter\\naver_tropicalfish_crawling\\naver_kin_crawling\\homedari.txt"
    output = "./homedari_auto.txt"
    with torch.no_grad():
        # while True:
        from tqdm import tqdm
        # with 
        with open(dataset,encoding="utf-8") as f, open(output,"w",encoding="utf-8") as ff:
            count = 0
            for _ in f:
                count += 1
            f.seek(0)
            with tqdm(total=count) as pbar:
                for txt in f:
                    # txt = input("Input: ")
                    txt = txt.strip()
                    # txt = re.sub(r"[\.+]{2,}"," ",txt)
                    # txt = re.sub(r"[\!+]{2,}"," ",txt)
                    # txt = re.sub(r"[\?+]{2,}"," ",txt)
                    # txt = re.sub(r" +"," ",txt)
                    last = txt[-1]
                    lastchk = False
                    # if last in [".","!","?"]:
                    #     txt = txt[:-1]
                    #     lastchk = True
                    # txt = re.sub(r"[^a-z|A-Z|가-힣|\d]+","",txt)
                    if lastchk:
                        txt = txt + last
                    inputt = [lstm_vocab[t] if t in lstm_vocab else 1 for t in txt][:300]
                    
                    inputt = torch.tensor([inputt],dtype=torch.long).to(device)
                    
                    res = model(inputt)
                    
                    res = torch.argmax(res,dim=-1).view(-1).cpu().numpy().tolist()
                    txt_ = list(copy.deepcopy(txt))[:300]
                    txt = txt[:300]
                    # print(len(txt_),len(res))
                    for i in range(len(txt)):
                        # print(i,end=" ")
                        if res[i] == 1:
                            txt_[i] = txt_[i] + " "
                    # print()
                    autotxt = "".join(txt_).strip()
                    autotxt = re.sub(" +"," ",autotxt)
                    ff.write(autotxt+"\n")
                    ff.flush()
                    pbar.update(1)
                    # exit()
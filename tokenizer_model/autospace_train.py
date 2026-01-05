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
parser.add_argument("--MAX_LEN",type=int,help="Sequence Data MAX LENGTH",default=300)
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
# x_f.seek(0)
# from tqdm
count = count // BATCH_SIZE
count_data = count
count_data = int(count * 0.9) 
validation_data = int(count * 0.05)
with open('bigram_vocab.pkl','rb') as f:
    bigram = pickle.load(f)
# print(bigram)
print(count)
print(count_data)
print(validation_data)
# exit()
def dataset2(flag=False):
    with open("autospace_x.txt",'r',encoding='utf-8') as x_f:
        with open("autospace_y.txt",'r',encoding='utf-8') as y_ff:   
            X = []
            Y = []
            lstm_X = []
            lstm_Y = []
            # BI = []
            Auto = []
            file_num = 0
            # bigram_index = 0
            x_f.seek(0)
            y_ff.seek(0)
            if flag:
                start = ((count_data +1) * BATCH_SIZE)
                # end = (count + 1) * BATCH_SIZE
                end = (count_data + validation_data+ 1) * BATCH_SIZE#start + (validation_data* BATCH_SIZE)
                for ii in range(start):
                    # for jjj,jj in zip(x_f,y_ff):
                        # True
                    x_f.readline()
                    y_ff.readline()
            else:
                start = 0 
                end = ((count_data+1) * BATCH_SIZE) 
            length = []
            for _ in range(start,end):#zip(x_f,y_ff):
                
                file_num = 0
                x = x_f.readline()
                y = y_ff.readline()
                x = x.replace('\n','')
                y = y.replace('\n','')
                # length.append(len(x))
                # if flag:
                    # print(f"\r{_}/{validation_data*BATCH_SIZ}")
                if True: #len(x) <= max_len and len(y) <= max_len:
                    # bi_npy = []
                    lstm_x = [lstm_vocab[i] if i in lstm_vocab else 1 for i in x]
                    lstm_y = [int(i) for i in y]
                    length.append(min(len(lstm_x),max_len))
                    
                    lstm_X.append(lstm_x)
                    lstm_Y.append(lstm_y)

                if len(lstm_X) == BATCH_SIZE:
                    mx_length = max(length)
                    
                    tmpX = []
                    tmpY = []
                    # tmpAuto = []
                    for lx,ly in zip(lstm_X,lstm_Y):
                        lstm_x = lx[:mx_length] + [0] * (mx_length - len(lx))
                        lstm_y = ly[:mx_length] + [2] * (mx_length - len(ly))
                        
                        tmpX.append(lstm_x)
                        tmpY.append(lstm_y)
                        
                    # print(tmpX)    
                    lstm_X = np.array(tmpX)
                    lstm_Y = np.array(tmpY)
                    
                    lstm_X = torch.tensor(lstm_X)
                    lstm_Y = torch.tensor(lstm_Y)
                    yield lstm_X, lstm_Y
                    lstm_X = []
                    lstm_Y = []
                    file_num += 1
                    length = []
                    

def dataset():
    for _ in range(EPOCH):
        for i in range(count_data):
            data = np.load('token_data/%05d_lstm_x.npy' % i)
            data_bigram = np.load('token_data/%05d_bigram.npy' % i)
            y = np.load('token_data/%05d_lstm_y.npy' % i)
            data = torch.tensor(data)
            data_bigram = torch.tensor(data_bigram)
            y = torch.tensor(y)
            yield [data,data_bigram],y

def validation():
    for _ in range(EPOCH*2):
        for i in range(count_data,count_data+validation_data):
            data = np.load('token_data/%05d_lstm_x.npy' % i)
            data_bigram = np.load('token_data/%05d_bigram.npy' % i)
            y = np.load('token_data/%05d_lstm_y.npy' % i)
            data = torch.tensor(data)
            data_bigram = torch.tensor(data_bigram)
            y = torch.tensor(y)
            yield [data,data_bigram],y

with open('lstm_vocab.pkl','rb') as f:
    lstm_vocab = pickle.load(f)

tag_len = 3
class TK_Model(nn.Module):
    def __init__(self):
        super(TK_Model,self).__init__()
        self.emb = nn.Embedding(len(lstm_vocab.keys()),128)
        
        self.emb_bi = nn.Embedding(len(bigram.keys()),128)

        self.bilstm = nn.LSTM(128,256,bidirectional=True,batch_first=True)
        self.bilstm_bi = nn.LSTM(128,256,bidirectional=True,batch_first=True)

        self.output_tag = nn.Linear(256 * 4,tag_len)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self,x,bi):
        emb = self.emb(x)
        emb_bi = self.emb_bi(bi)

        bilstm,_ = self.bilstm(emb)
        bilstm_bi,_ = self.bilstm_bi(emb_bi)

        concat = torch.concat([bilstm,bilstm_bi],dim=-1)#torch.concat([bilstm,bilstm_bi],dim=-1)
        output = self.output_tag(concat)
        output = self.softmax(output)

        return output
        
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

train_data = dataset()
val_data = validation()

def train(config,optimizer,loss,model,x,y,device):
    model.train()
    
    x = x.to(device)
    # y = y.to(device)

    yp_tmp = model(x)
    y_pred = yp_tmp#.to(device)
    y_pred = y_pred.view(-1, tag_len)
    
    y_tmp = y
    y = y_tmp.to(device)
    y = y.view(-1)
    y = y.type(torch.LongTensor).to(device)
    loss_ = loss(y_pred,y)

    optimizer.zero_grad()
    loss_.backward()
    
    # torch.nn.utils.clip_grad_norm_(model.parameters(),config["max_grad_norm"])

    optimizer.step()

    return loss_.item()

def eval(val_dataloader,device,model):
    with torch.no_grad():
        count = 0
        result = 0
        # result2 = 0
        for index in range(validation_data):
            xbi,y_ = next(val_dataloader)
            
            # bi_ = xbi[1].to(device)
            x_ = xbi.to(device)
            
            y_tmp = y_
            
            y_ = y_tmp
            y_ = y_.to(device)
            y_ = y_.type(torch.LongTensor).to(device)
            # yy = model(x_,bi_)
            yy = model(x_)

            yp_tmp = yy
            yy = yp_tmp
            
            yy = torch.argmax(yy,dim=-1)
            for yp__, y__ in zip(yy,y_):
                y__ = y__[y__!=2]
                yp__ = yp__[:y__.shape[0]]
                comp = (y__ == yp__).view(-1)
                
                result += (comp[comp==True].shape[0] / (y__.shape[0] + 1e-9))
            print(f"\r{index+1}/{validation_data}",end="")
        print(result)
        print("auto avg",result/(validation_data*BATCH_SIZE))
        
def train2(config,optimizer,loss,model,bi,x,y,device):
    model.train()
    
    bi = bi.to(device)
    x = x.to(device)
    
    y = y.to(device)

    y_pred = model(x)
    # print(y_pred.shape)
    y_pred = y_pred.view(-1, 4)

    y = y.view(-1)
    # print(y.shape)
    # exit()
    y = y.type(torch.LongTensor).to(device)
    loss_ = loss(y_pred,y)

    optimizer.zero_grad()
    loss_.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(),config["max_grad_norm"])

    optimizer.step()

    return loss_.item()

def eval2(val_dataloader,device,model):
    with torch.no_grad():
        count = 0
        result = 0
        for index in range(args.validation_step):
            xbi,y_ = next(val_dataloader)
            # x_ = [data,masks,segments]
            # count+=1
            bi_ = xbi[1].to(device)
            x_ = xbi[0].to(device)
            
            y_ = y_.to(device)
            y_ = y_.type(torch.LongTensor).to(device)
            yy = model(x_)

            yy = yy.to(device)
            # print(y_.shape)
            # exit()
            _,yy = torch.topk(yy,k=1,dim=-1)
            # print(yy.shape)
            yy = yy.view(yy.shape[0],yy.shape[1])
            # exit()
            for r,_ in enumerate(y_):
                # print(r,c)
                y__ = y_[r]
                y__ = y__[y__ == 3]
                yy_ = yy[r][:y__.shape[0]]
                comp = torch.eq(yy_,y__)
                # print(y__)
                # print(comp)
                # index_sep = (y_ == 1).nonzero()
                count += y__.shape[0]
                comp = comp[comp==True].view(-1)
                # print(comp.shape)
                # exit()
                result += (comp.shape[0]/y__.shape[0])
        print(result)
        print("avg",result/(args.validation_step*50))

if __name__ == "__main__":
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    print(device)
    config = {"maxlen":420,"max_grad_norm":1,"epoch":10,"batch_size":32}
    
    loss = nn.NLLLoss(ignore_index=3)
    loss2 = nn.NLLLoss(ignore_index=2)
    
    model = TK_Model2()
    
    model.to(device)
    # model.load_state_dict(torch.load('tokenizer_model/model'))
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)#5e-5)#8e-5)#5e-5)
    
    # n_data = dataset()
    # n_validation = validation()
    from tqdm import tqdm
    from datetime import datetime
    for e in range(5,5+EPOCH):
        running_loss = 0
        model.train()
        n_data = dataset2()
        def timestring(td):
            td_in_seconds = td.total_seconds()
            hours, remainder = divmod(td_in_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            hours = int(hours)
            minutes = int(minutes)
            seconds = int(seconds)
            
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        start = datetime.now()
        for index in range(count_data):
            start_b = datetime.now()
            x,y= next(n_data)
            loss__ = train(config,optimizer,loss,model,x,y,device)
            running_loss += loss__
            end = datetime.now()
            cur = end - start
            cur_working = timestring(cur)#cur.strftime("%H:%M:%S")
            # total = cur
            end_b = datetime.now()
            # for _ in range(count_data)
            total = (end_b - start_b) * count_data
            # for i in range(count_data-1):
            #     total += (start_b - end_b)
                
            total_working = timestring(total)#total.strftime("%H:%M:%S")
            print(f"\repoch:{e}",",",f"{index+1}/{count_data}",f"batch_loss:{loss__:.6f}",",",f"total_loss:{(running_loss / ((index+1))):.6f}",",",f"{cur_working}/{total_working}",end="")
            # if index+1 == count_data:
                # break
        print()
        model.eval()
        n_validation=dataset2(flag=True)
        print()
        eval(n_validation,device,model)
        if not os.path.exists("autospace_model"):
            os.makedirs("autospace_model")
        torch.save(model.state_dict(), 'autospace_model/model_{}'.format(e))#{}'.format(e))
    torch.save(model.state_dict(), 'autospace_model/model')


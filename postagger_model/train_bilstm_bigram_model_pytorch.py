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

parser = argparse.ArgumentParser(description="Train Head-Tail Tokenizer")

parser.add_argument("--BATCH",type=int,help="Train Data BATCH SIZE",default=256)
parser.add_argument("--MAX_LEN",type=int,help="Sequence Data MAX LENGTH",default=300)
parser.add_argument("--epoch_step",type=int,help="Train Data Epoch Step",default=4000)
parser.add_argument("--validation_step",type=int,help="Validation Data Epoch Step",default=400)
parser.add_argument("--EPOCH",type=int,help="Train Epoch SIZE",default=6)
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

root = "."

with open(root + os.sep + 'lstm_vocab.pkl','rb') as f:
    lstm_vocab = pickle.load(f)
with open(root + os.sep + 'bigram_vocab.pkl','rb') as f:
    bigram = pickle.load(f)

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

class TK_Model(nn.Module):
    def __init__(self):
        super(TK_Model,self).__init__()
        self.emb = nn.Embedding(len(lstm_vocab.keys()),100)
        
        self.emb_bi = nn.Embedding(len(bigram.keys()),100)

        self.bilstm = nn.LSTM(100,64,bidirectional=True)
        self.bilstm_bi = nn.LSTM(100,64,bidirectional=True)

        self.output_tag = nn.Linear(64 * 4,4)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self,x,bi):
        emb = self.emb(x)
        emb_bi = self.emb_bi(bi)

        bilstm,_ = self.bilstm(emb)
        bilstm_bi,_ = self.bilstm_bi(emb_bi)

        concat = torch.concat([bilstm,bilstm_bi],dim=-1)
        output = self.output_tag(concat)
        output = self.softmax(output)

        return output       

train_data = dataset()
val_data = validation()

def train(config,optimizer,loss,model,bi,x,y,device):
    model.train()
    
    bi = bi.to(device)
    x = x.to(device)
    
    y = y.to(device)

    y_pred = model(x,bi)
    y_pred = y_pred.view(-1, 5)
    y = y.view(-1)
    y = y.type(torch.LongTensor).to(device)
    loss_ = loss(y_pred,y)

    optimizer.zero_grad()
    loss_.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(),config["max_grad_norm"])

    optimizer.step()

    return loss_.item()

def eval(val_dataloader,device,model):
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
            yy = model(x_,bi_)

            yy = yy.to(device)
            # print(y_.shape)
            # exit()
            _,yy = torch.topk(yy,k=1,dim=-1)
            # print(yy.shape)
            yy = yy.view(yy.shape[0],yy.shape[1])
            index_sep = (y_ == 4).nonzero()
            # print(y_)
            # exit()
            for r,c in index_sep:#enumerate(y_):
                # print(r,c)
                y__ = y_[r][:c]
                yy_ = yy[r][:c]
                comp = torch.eq(yy_,y__)
                # print(y__)
                # print(comp)
                index_sep = (y_ == 1).nonzero()
                count += y__.shape[0]
                comp = comp[comp==True].view(-1)
                # print(comp.shape)
                # exit()
                result += (comp.shape[0]/y__.shape[0])
        print(result)
        print("avg",result/(args.validation_step*50))
        # exit()

if __name__ == "__main__":
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    print(device)
    config = {"maxlen":420,"max_grad_norm":1,"epoch":10,"batch_size":32}
    
    loss = nn.NLLLoss()
    
    model = TK_Model()
    
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(),lr=8e-5)#5e-5)#8e-5)#5e-5)
    
    n_data = dataset()
    n_validation = validation()
    from tqdm import tqdm
    for e in range(EPOCH):
        running_loss = 0
        model.train()
        
        for index in range(count_data):
            xbi,y= next(n_data)
            loss__ = train(config,optimizer,loss,model,xbi[1],xbi[0],y,device)
            running_loss += loss__
            if index % 100 == 0 and index != 0:
                print(e,index,loss__,(running_loss / ((index))))
        model.eval()
        eval(n_validation,device,model)
        if not os.path.exists("tokenizer_model"):
            os.makedirs("tokenizer_model")
        chkpoint = {
            "model":model.state_dict(),
            "loss":loss.state_dict(),
            "optimizer":optimizer.state_dict()
        }
        torch.save(chkpoint, 'tokenizer_model/model_{}'.format(e))#{}'.format(e))
    torch.save(model.state_dict(), 'tokenizer_model/model')


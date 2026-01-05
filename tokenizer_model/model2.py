# import sys
# sys.path.append("C:\\Users\\ty341\\OneDrive\\Desktop\\kjm\\runpod\\runpod\\transformer")
from transformer import TransformerConfig, Encoder, PositionalEncoding
from torch import nn
# from torchcrf import CRF
import torch
import math

class BiEmbedding(nn.Module):
    def __init__(self,config: TransformerConfig):
        super(BiEmbedding,self).__init__()
        self.emb = nn.Embedding(config.vocab_size, config.dmodel) 
        """self.biemb = nn.Embedding(config.vocab_size, config.dmodel)"""
        self.pe = PositionalEncoding(config)#.to(device)
        self.config = config
        """self.lstm = nn.LSTM(config.dmodel,config.dmodel//2,batch_first=True,bidirectional=True)
        self.layer = nn.Linear(config.dmodel,config.dmodel)"""
        self.norm = nn.LayerNorm(config.dmodel)
        # with open("subwordvocab2.pkl","rb") as f:
            # self.subwordvocab = pickle.load(f)
    def forward(self,x):
        emb = self.emb(x)
        # emb = self.layer(emb)
        # bigram = self.setbigram(x)
        emb = emb * math.sqrt(self.config.dmodel)# + bigram
        emb = self.pe(emb,emb.shape[1])
        # emb = self.norm(emb)#3
        
        return emb
    
    def setbigram(self,data):
        # self.bigram = x
        # data_ = data.unfold(1,3,1)
        data_ = data.unfold(1,5,1)
        # cls = torch.full((data.shape[0],1,2),subwordvocab.cls).to(data.device)
        cls = torch.concat((data[:,0:1],data[:,0:1],data[:,0:1],data[:,0:1]),dim=1)#torch.full((data.shape[0],1,2))
        cls = cls.unsqueeze(1)
        # print(cls.shape)
        # exit()
        pad = torch.full((data.shape[0],1,4),0).to(data.device)
        data_ = torch.concat((data_[:,:,0:1],data_[:,:,1:2],data_[:,:,3:4],data_[:,:,4:5]),dim=2)
        # data_ = data_[:,:,0:1]+data_[:,:,1:2]+data_[:,:,3:4]+data_[:,:,4:5]
        
        data_ = torch.concat((cls,data_,pad,pad,pad),dim=1)
        # print(data_.shape)
        data_ = self.biemb(data_)
        # data_ = torch.concat((data_[:,:,0,:],data_[:,:,1,:],data_[:,:,2,:],data_[:,:,3,:]),dim=2)
        data_ = data_[:,:,0,:]+data_[:,:,1,:]+data_[:,:,2,:]+data_[:,:,3,:]
        data_ = data_ / 4
       
        # print(data_.shape)
        # exit()
        # print(data.shape,data_.shape)
        # exit()
        
        lstm,_ = self.lstm(data_)
        # print(lstm.shape)
        # print(_[0].shape,_[1].shape)
        # exit()
        
        return lstm

class TK_Model_Mini2(nn.Module):
    def __init__(self,max_len,lstm_vocab,bigram,tag_len=1,device=torch.device("cuda")):
        super(TK_Model_Mini2,self).__init__()
        # self.model = SelfEncoder(lstm_vocab)
        # self.config = self.model.config
        dmodel = 128
        self.emb = nn.Embedding(len(lstm_vocab),dmodel)
        self.lstm = nn.LSTM(dmodel,dmodel//2,bidirectional=True,batch_first=True)
        """self.biemb = nn.Embedding(self.config.vocab_size, self.config.dmodel)"""
        self.sublayer = nn.Linear(dmodel,dmodel//4)
        self.act = nn.GELU()
        self.vocab_output = torch.nn.Linear(dmodel//4,1)
        """self.ngramlstm = torch.nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)"""
        self.tag_len = tag_len
        
    def forward(self,x):#,unfold):
        if x.dtype != torch.long:
            # print("fff")
            x = x.type(torch.LongTensor).to(x.device)
        
        """# emb = self.emb(x)
        enc = self.model(x,is_softmax=True)
        lstm = self.setbigram(x,unfold)
        
        # lstm,_  = self.lstm(enc)
        # enc = self.layer(lstm)
        enc = enc + lstm# * 0.6
        output = self.vocab_output(enc)"""
        enc1 = self.emb(x)#self.model(x,is_softmax=True)
        """enc2 = self.model2(unfold,is_softmax=True)"""
        lstm,_ = self.lstm(enc1)#+enc2)
        output = self.sublayer(lstm)
        output = self.act(output)
        output = self.vocab_output(output)
        output = torch.nn.functional.sigmoid(output)
        return [output]
    
    def setbigram(self,data,unfold):
        # self.bigram = x
        # data_ = data.unfold(1,3,1)
        data_ = unfold#data.unfold(1,3,1)
        # cls = torch.full((data.shape[0],1,2),subwordvocab.cls).to(data.device)
        cls = torch.concat((data[:,0:1],data[:,0:1]),dim=1)#torch.full((data.shape[0],1,2))
        cls = cls.unsqueeze(1)
        # print(cls.shape)
        # exit()
        pad = torch.full((data.shape[0],1,2),0).to(data.device)
        data_ = torch.concat((data_[:,:,0:1],data_[:,:,1:2]),dim=2)
        # data_ = data_[:,:,0:1]+data_[:,:,1:2]+data_[:,:,3:4]+data_[:,:,4:5]
        
        data_ = torch.concat((cls,data_,pad),dim=1)
        # print(data_.shape)
        data_ = self.biemb(data_)
        # data_ = torch.concat((data_[:,:,0,:],data_[:,:,1,:],data_[:,:,2,:],data_[:,:,3,:]),dim=2)
        data_ = data_[:,:,0,:]+data_[:,:,1,:]#+data_[:,:,2,:]+data_[:,:,3,:]
        data_ = data_ / 2
        
        data_, _ = self.ngramlstm(data_)
        
        return data_
    
class TK_Model_Mini2CPP(nn.Module):
    def __init__(self,max_len,lstm_vocab,bigram,tag_len=1,device=torch.device("cuda")):
        super(TK_Model_Mini2CPP,self).__init__()
        dmodel = 128
        self.emb = nn.Embedding(len(lstm_vocab),dmodel)
        self.lstm = nn.LSTM(dmodel,dmodel//2,bidirectional=True,batch_first=True)
        """self.biemb = nn.Embedding(self.config.vocab_size, self.config.dmodel)"""
        self.sublayer = nn.Linear(dmodel,dmodel//4)
        self.act = nn.GELU()
        self.vocab_output = torch.nn.Linear(dmodel//4,1)
        """self.ngramlstm = torch.nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)"""
        self.tag_len = tag_len
        
    def forward(self,x):#,unfold):
        if x.dtype != torch.long:
            x = x.type(torch.LongTensor).to(x.device)
        if x.dtype != torch.long:
            # print("fff")
            x = x.type(torch.LongTensor).to(x.device)
        
        """# emb = self.emb(x)
        enc = self.model(x,is_softmax=True)
        lstm = self.setbigram(x,unfold)
        
        # lstm,_  = self.lstm(enc)
        # enc = self.layer(lstm)
        enc = enc + lstm# * 0.6
        output = self.vocab_output(enc)"""
        enc1 = self.emb(x)#self.model(x,is_softmax=True)
        """enc2 = self.model2(unfold,is_softmax=True)"""
        lstm,_ = self.lstm(enc1)#+enc2)
        output = self.sublayer(lstm)
        output = self.act(output)
        output = self.vocab_output(output)
        output = torch.nn.functional.sigmoid(output)
        result = torch.zeros(output.shape,dtype=torch.int)
        result[output>=0.5] = 1
        # print(result.dtype)
        # print(result)
        return [result]
    

class SelfEncoder(nn.Module):
    def __init__(self,posvocab):
        super(SelfEncoder,self).__init__()
        
        self.config = TransformerConfig(pad_idx=posvocab["[PAD]"],numheads=4,group_attn=False,dmodel=64, dff=256,nlayer=3,maxlen=2000,vocab_size=len(posvocab),dropout=0.02,warmup_steps=100)
        
        # Encoder Model
        self.model = Encoder(self.config,last_layer=False,embedding=BiEmbedding,pre_ln=True)#.to(device)
        
        '''
        # #LSTM Model
        self.emb = nn.Embedding(len(posvocab),self.config.dmodel)
        self.model = nn.LSTM(self.config.dmodel,self.config.dmodel//2,batch_first=True,num_layers=2,bidirectional=True)
        self.layer = nn.Linear(self.config.dmodel,self.config.vocab_size)
        '''
        
        self.softmax = nn.LogSoftmax(dim=-1)
        self.lstm_vocab = posvocab
        self.pad = self.lstm_vocab["[PAD]"]
        self.cls = self.lstm_vocab["[SOS]"]
        self.sep = self.lstm_vocab["[EOS]"]
        
    
    def forward(self,x,is_softmax=False):
        # emb = self.emb(x)
        #LSTM MODEL
        # return self.forward2(x,is_softmax=is_softmax)
        
        enc_out = self.model(x)
        # unfold = self.setbigram(x,unfold)
        # enc_out = self.norm(enc_out+unfold)
        
        if not is_softmax:
            vocab = self.layer(enc_out)
            
            out = self.softmax(vocab)
        
            return out
        
        return enc_out
    
    def forward2(self,x,is_softmax=False):
        emb = self.emb(x)
        enc_out,_ = self.model(emb)
            
        if not is_softmax:
            vocab = self.layer(enc_out)
        
            out = self.softmax(vocab)
            
            return out

        return enc_out
        
    def setbigram(self,data,unfold):
        data_ = unfold#data.unfold(1,3,1)
        cls = torch.full((data.shape[0],1,2),self.cls).to(data.device)
        pad = torch.full((data.shape[0],1,2),self.pad).to(data.device)
        data_ = torch.concat((data_[:,:,0:1],data_[:,:,2:3]),dim=2)
        data_ = torch.concat((cls,data_,pad),dim=1)
        data_ = self.biemb(data_)
        data_ = data_[:,:,0,:]+data_[:,:,1,:]#torch.concat((data_[:,:,0,:],data_[:,:,1,:]),dim=-1)
        lstm,_ = self.lstm(data_)
        
        return lstm

# import sys
# sys.path.append("C:\\Users\\ty341\\OneDrive\\Desktop\\kjm\\runpod\\runpod\\transformer")
from transformer import TransformerConfig, Encoder, PositionalEncoding
from torch import nn
# from torchcrf import CRF
import torch
import math
class TK_Model2(nn.Module):
    def __init__(self,max_len,lstm_vocab,tag_len=4,device=torch.device("cuda")):
        super(TK_Model2,self).__init__()
        self.config = TransformerConfig(pad_idx=0,numheads=8,ngroup=4,group_attn=True,dmodel=256, dff=1024,nlayer=6,maxlen=max_len,vocab_size=len(lstm_vocab),dropout=0.1,warmup_steps=100)
        self.model = Encoder(self.config,last_layer=False,embedding=BiEmbedding)#.to(device)
        # self.lstm = nn.LSTM(self.config.dmodel,64,batch_first=True)
        self.vocab_output = torch.nn.Linear(self.config.dmodel,tag_len)
        # self.auto_output = torch.nn.Linear(self.config.dmodel,3)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.tag_len = tag_len
        
    def forward(self,x):
        enc = self.model(x)
        # enc,_ = self.lstm(enc)
        output = self.vocab_output(enc)
        output = self.softmax(output)
        
        # output2 = self.auto_output(enc)
        # output2 = self.softmax(output2)
        return [output]
    
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
        # emb = emb * math.sqrt(self.config.dmodel)# + bigram
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
    

class TK_Model3(nn.Module):
    def __init__(self,max_len,lstm_vocab,tag_len=4,device=torch.device("cuda")):
        super(TK_Model3,self).__init__()
        self.config = TransformerConfig(pad_idx=0,numheads=4,ngroup=2,group_attn=True,dmodel=256, dff=512,nlayer=4,maxlen=max_len,vocab_size=len(lstm_vocab),dropout=0.1,warmup_steps=100)
        self.model = Encoder(self.config,last_layer=False,embedding=BiEmbedding)#.to(device)
        # self.lstm = nn.LSTM(self.config.dmodel,64,batch_first=True)
        self.vocab_output = torch.nn.Linear(64,tag_len)
        # self.auto_output = torch.nn.Linear(self.config.dmodel,3)
        self.softmax = nn.LogSoftmax(dim=-1)
        # self.lstm = nn.LSTM(self.config.dmodel,64,batch_first=True,bidirectional=True)
        self.layer = nn.Linear(self.config.dmodel,64)
        self.tag_len = tag_len
        # self.crf = CRF(self.tag_len,batch_first=True)
        
    def forward(self,x):
        
        # print(x.shape)
        # x = x.type(torch.LongTensor).to(x.device)
        if x.dtype != torch.long:
            x = x.type(torch.LongTensor).to(x.device)
            # print("변경")
        enc = self.model(x)
        # lstm,_  = self.lstm(enc)
        # enc = self.layer(lstm)
        enc = self.layer(enc)
        output = self.vocab_output(enc)
        output = self.softmax(output)
        return [output]
        # enc = self.vocab_output(enc)
        # enc = self.softmax(enc)
        # print(x.shape,y.shape)
        # exit()
        # enc,_ = self.lstm(enc)
        if y.shape[0] > 0:
            output = self.crf(enc,y)
            # print(output)
            # exit()
        else:
            output = self.decode(enc)
        # output = self.vocab_output(enc)
        # output = self.softmax(output)
        
        # output2 = self.auto_output(enc)
        # output2 = self.softmax(output2)
        return [output]

# from headtail.postagger_model import SelfEncoder
class TK_Model_MiniOld(nn.Module):
    def __init__(self,max_len,lstm_vocab,tag_len=4,device=torch.device("cuda")):
        super(TK_Model_MiniOld,self).__init__()
        self.config = TransformerConfig(pad_idx=0,numheads=4,ngroup=4,group_attn=True,dmodel=256, dff=800,nlayer=4,maxlen=350,vocab_size=len(lstm_vocab),dropout=0.1,warmup_steps=100)
        self.model = Encoder(self.config,last_layer=False,embedding=BiEmbedding)#.to(device)
        # self.lstm = nn.LSTM(self.config.dmodel,64,batch_first=True)
        # self.model = SelfEncoder(posvocab)
        # self.config = self.model.config
        # self.model.load_state_dict(torch.load("./posttager_model/selfmodel_29",weights_only=True)).to("cuda")
        # self.model = self.model.model
        self.biemb = nn.Embedding(self.config.vocab_size, self.config.dmodel)
        self.vocab_output = torch.nn.Linear(self.config.dmodel,tag_len)
        self.ngramlstm = torch.nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)
        # self.auto_output = torch.nn.Linear(self.config.dmodel,3)
        self.softmax = nn.LogSoftmax(dim=-1)
        # self.lstm = nn.LSTM(self.config.dmodel,64,batch_first=True,bidirectional=True)
        # self.layer = nn.Linear(self.config.dmodel,64)
        self.tag_len = tag_len
        # self.crf = CRF(self.tag_len,batch_first=True)
        
    def forward(self,x):
        # print(x.shape)
        # x = x.type(torch.LongTensor).to(x.device)
        if x.dtype != torch.long:
            x = x.type(torch.LongTensor).to(x.device)
            # print("변경")
        enc = self.model(x)
        lstm = self.setbigram(x)
        
        # lstm,_  = self.lstm(enc)
        # enc = self.layer(lstm)
        enc = enc + lstm# * 0.6
        output = self.vocab_output(enc)
        output = self.softmax(output)
        return [output]
    
    def setbigram(self,data):
        # self.bigram = x
        # data_ = data.unfold(1,3,1)
        data_ = data.unfold(1,3,1)
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

class TK_Model_Mini_Auto(nn.Module):
    def __init__(self,max_len,lstm_vocab,tag_len=1,device=torch.device("cuda")):
        super(TK_Model_Mini_Auto,self).__init__()
        # self.config = TransformerConfig(pad_idx=0,numheads=4,ngroup=2,group_attn=True,dmodel=128, dff=800,nlayer=4,maxlen=350,vocab_size=len(lstm_vocab),dropout=0.1,warmup_steps=100)
        # self.model = nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)#Encoder(self.config,last_layer=False,embedding=BiEmbedding)#.to(device)
        # self.config = self.model.config
        dmodel = 256
        self.emb = nn.Embedding(len(lstm_vocab), dmodel)
        # self.lstm = nn.LSTM(self.config.dmodel,64,batch_first=True)
        # self.model = SelfEncoder(lstm_vocab)
        
        # self.model.load_state_dict(torch.load("C:/Users/ty341/OneDrive/Desktop/kjm/headtail/tokenizer_model/selfmodel_mini_10",weights_only=True))#.to("cuda")
        # self.model = self.model.model
        self.lstm = nn.LSTM(dmodel,dmodel//2,bidirectional=True,batch_first=True,num_layers=4)
        """self.biemb = nn.Embedding(self.config.vocab_size, self.config.dmodel)"""
        self.output = nn.Linear(dmodel,dmodel)
        self.vocab_output = torch.nn.Linear(dmodel,1)
        self.vocab_output2 = torch.nn.Linear(dmodel,1)
        """self.ngramlstm = torch.nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)"""
        # self.auto_output = torch.nn.Linear(self.config.dmodel,3)
        # self.softmax = nn.LogSoftmax(dim=-1)
        # self.lstm = nn.LSTM(self.config.dmodel,64,batch_first=True,bidirectional=True)
        # self.layer = nn.Linear(self.config.dmodel,64)
        self.tag_len = tag_len
        # self.crf = CRF(self.tag_len,batch_first=True)

        
    def forward(self,x,unfold):
        # print(x.shape)
        # x = x.type(torch.LongTensor).to(x.device)
        if x.dtype != torch.long:
            x = x.type(torch.LongTensor).to(x.device)
            unfold = unfold.type(torch.LongTensor).to(x.device)
            # print("변경")
        # emb = self.emb(x)
        emb = self.emb(x)
        # enc = self.model(x,is_softmax=True)
        """lstm = self.setbigram(x,unfold)"""
        
        # lstm,_  = self.lstm(enc)
        # enc = self.layer(lstm)
        """enc = enc + lstm# * 0.6"""
        enc,_ = self.lstm(emb)
        enc = self.output(enc)
        output = self.vocab_output(enc)
        output2 = self.vocab_output2(enc)
        # output = self.softmax(output)
        return [output,output2]
    
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


class TK_Model_Mini2(nn.Module):
    def __init__(self,max_len,lstm_vocab,bigram,tag_len=1,device=torch.device("cuda")):
        super(TK_Model_Mini2,self).__init__()
        self.model = SelfEncoder(lstm_vocab)
        self.config = self.model.config
        # self.lstm = nn.LSTM(self.config.dmodel,self.config.dmodel//2,bidirectional=True,batch_first=True)
        """self.biemb = nn.Embedding(self.config.vocab_size, self.config.dmodel)"""
        self.sublayer = torch.nn.Linear(self.config.dmodel,self.config.dmodel//4)
        self.act = torch.nn.GELU()
        self.layer_norm = nn.LayerNorm(self.config.dmodel // 4)
        self.vocab_output = torch.nn.Linear(self.config.dmodel//4,1)
        """self.ngramlstm = torch.nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)"""
        self.tag_len = tag_len
        self.init_()

    def init_(self):
        torch.nn.init.trunc_normal_(self.sublayer.weight, std=0.02)
        self.sublayer.bias = torch.nn.Parameter(torch.zeros(self.config.dmodel//4))

        torch.nn.init.trunc_normal_(self.vocab_output.weight, std=0.02)
        self.vocab_output.bias = torch.nn.Parameter(torch.zeros(1))
        
        
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
        enc1 = self.model(x,is_softmax=True)
        """enc2 = self.model2(unfold,is_softmax=True)"""
        # lstm,_ = self.lstm(enc1)#+enc2)
        enc1 = self.sublayer(enc1)
        enc1 = self.act(enc1)
        enc1 = self.layer_norm(enc1)
        output = self.vocab_output(enc1)
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
        self.model = SelfEncoder(lstm_vocab)
        self.config = self.model.config
        # self.lstm = nn.LSTM(self.config.dmodel,self.config.dmodel//2,bidirectional=True,batch_first=True)
        """self.biemb = nn.Embedding(self.config.vocab_size, self.config.dmodel)"""
        self.sublayer = torch.nn.Linear(self.config.dmodel,self.config.dmodel//4)
        self.act = torch.nn.GELU()
        self.layer_norm = nn.LayerNorm(self.config.dmodel // 4)
        self.vocab_output = torch.nn.Linear(self.config.dmodel//4,1)
        """self.ngramlstm = torch.nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)"""
        self.tag_len = tag_len
        self.init_()
    
    def init_(self):
        torch.nn.init.trunc_normal_(self.sublayer.weight, std=0.02)
        self.sublayer.bias = torch.nn.Parameter(torch.zeros(self.config.dmodel//4))

        torch.nn.init.trunc_normal_(self.vocab_output.weight, std=0.02)
        self.vocab_output.bias = torch.nn.Parameter(torch.zeros(1))
        
    def forward(self,x):#,unfold):
        if x.dtype != torch.long:
            # print("fff")
            x = x.type(torch.LongTensor).to(x.device)
        
        enc1 = self.model(x,is_softmax=True)
        enc1 = self.sublayer(enc1)
        enc1 = self.act(enc1)
        enc1 = self.layer_norm(enc1)
        output = self.vocab_output(enc1)
        output = torch.nn.functional.sigmoid(output)

        result = torch.zeros(output.shape,dtype=torch.int)
        result[output>=0.5] = 1

        return [result]
    

class TK_Model_Mini3(nn.Module):
    def __init__(self,max_len,lstm_vocab,tag_len=4,device=torch.device("cuda")):
        super(TK_Model_Mini3,self).__init__()
        # self.config = TransformerConfig(pad_idx=0,numheads=4,ngroup=2,group_attn=True,dmodel=128, dff=800,nlayer=4,maxlen=350,vocab_size=len(lstm_vocab),dropout=0.1,warmup_steps=100)
        # self.model = nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)#Encoder(self.config,last_layer=False,embedding=BiEmbedding)#.to(device)
        # self.emb = nn.Embedding(self.config.vocab_size, self.config.dmodel)
        # self.lstm = nn.LSTM(self.config.dmodel,64,batch_first=True)
        self.model = SelfEncoder(lstm_vocab)
        self.config = self.model.config
        # self.model.load_state_dict(torch.load("C:/Users/ty341/OneDrive/Desktop/kjm/headtail/tokenizer_model/selfmodel_mini_10",weights_only=True))#.to("cuda")
        # self.model = self.model.model
        self.biemb = nn.Embedding(self.config.vocab_size, self.config.dmodel)
        self.vocab_output = torch.nn.Linear(self.config.dmodel,1)
        self.ngramlstm = torch.nn.LSTM(self.config.dmodel,self.config.dmodel//2,batch_first=True,bidirectional=True,num_layers=2)
        # self.auto_output = torch.nn.Linear(self.config.dmodel,3)
        self.softmax = nn.LogSoftmax(dim=-1)
        # self.lstm = nn.LSTM(self.config.dmodel,64,batch_first=True,bidirectional=True)
        # self.layer = nn.Linear(self.config.dmodel,64)
        self.tag_len = tag_len
        # self.crf = CRF(self.tag_len,batch_first=True)
        
    def forward(self,x,unfold):
        # print(x.shape)
        # x = x.type(torch.LongTensor).to(x.device)
        if x.dtype != torch.long:
            x = x.type(torch.LongTensor).to(x.device)
            unfold = unfold.type(torch.LongTensor).to(x.device)
            # print("변경")
        # emb = self.emb(x)
        enc = self.model(x,is_softmax=True)
        lstm = self.setbigram(x,unfold)
        
        # lstm,_  = self.lstm(enc)
        # enc = self.layer(lstm)
        enc = enc + lstm# * 0.6
        output = self.vocab_output(enc)
        # output = self.softmax(output)
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



class TK_Model_Mini(nn.Module):
    def __init__(self,max_len,lstm_vocab,tag_len=4,device=torch.device("cuda")):
        super(TK_Model_Mini2,self).__init__()
        self.config = TransformerConfig(pad_idx=0,numheads=4,ngroup=4,group_attn=True,dmodel=128, dff=256,nlayer=4,maxlen=350,vocab_size=len(lstm_vocab),dropout=0.1,warmup_steps=100)
        self.model = nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)#Encoder(self.config,last_layer=False,embedding=BiEmbedding)#.to(device)
        self.emb = nn.Embedding(self.config.vocab_size, self.config.dmodel)
        # self.lstm = nn.LSTM(self.config.dmodel,64,batch_first=True)
        # self.model = SelfEncoder(posvocab)
        # self.config = self.model.config
        # self.model.load_state_dict(torch.load("./posttager_model/selfmodel_29",weights_only=True)).to("cuda")
        # self.model = self.model.model
        self.biemb = nn.Embedding(self.config.vocab_size, self.config.dmodel)
        self.vocab_output = torch.nn.Linear(self.config.dmodel,tag_len)
        self.ngramlstm = torch.nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)
        # self.auto_output = torch.nn.Linear(self.config.dmodel,3)
        self.softmax = nn.LogSoftmax(dim=-1)
        # self.lstm = nn.LSTM(self.config.dmodel,64,batch_first=True,bidirectional=True)
        # self.layer = nn.Linear(self.config.dmodel,64)
        self.tag_len = tag_len
        # self.crf = CRF(self.tag_len,batch_first=True)
        self.norm = torch.nn.LayerNorm(self.config.dmodel)
    def forward(self,x,unfold):
        # print(x.shape)
        # x = x.type(torch.LongTensor).to(x.device)
        if x.dtype != torch.long:
            x = x.type(torch.LongTensor).to(x.device)
            unfold = unfold.type(torch.LongTensor).to(x.device)
            # print("변경")
        emb = self.emb(x)
        enc,_ = self.model(emb)
        lstm = self.setbigram(x,unfold)
        
        # lstm,_  = self.lstm(enc)
        # enc = self.layer(lstm)
        enc = self.norm(enc + lstm)# * 0.6
        output = self.vocab_output(enc)
        output = self.softmax(output)
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


class EomiList(nn.Module):
    def __init__(self,config: TransformerConfig,tokenizer,sepid):
        self.eomi_emb = nn.Embedding(config.vocab_size,config.dmodel)
        self.eomi_lstm = nn.LSTM(config.dmodel,config.dmodel//2,batch_first=True,bidirectional=True)
        self.tokenizer = tokenizer
        self.layer = nn.Linear(config.dmodel,config.dmodel//4)
        
        self.prob = nn.Linear(config.dmodel//4,config.dmodel)
        self.config = config
        self.sepid = sepid
    def forward(self,x):
        # seps = (x == self.sepid).nonzero()
        
        emb = self.eomi_emb(x)
        lstm,state = self.eomi_lstm(emb)
        ff = self.layer(state[0])
        prob = self.prob(ff)
        # prob = prob[seps]
        # ff[~seps] = self.tokenizer.pad_id
        return prob


    
class SelfEncoder(nn.Module):
    def __init__(self,posvocab):
        super(SelfEncoder,self).__init__()
        
        self.config = TransformerConfig(pad_idx=posvocab["[PAD]"],numheads=4,group_attn=False,dmodel=128, dff=256,nlayer=3,maxlen=420,vocab_size=len(posvocab),dropout=0.001,warmup_steps=100)
        
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

"""
from ht_utils import HTInputTokenizer
from transformers import AutoTokenizer

# import math

import os
from inferutils import w2i
class Dataset(torch.utils.data.Dataset):
    def __init__(self,posvocab:HTInputTokenizer):
        self.htsep = "hththththt"#posvocab.htsep
        # self.tagsep = tagsep
        self.lstm_vocab = posvocab
        filename = os.path.join(os.environ["DATASET"],"ht_dataset","modu","result","delkiwimorphs.txt")
        self.datas = []
        
        with open(filename,encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                l = l.replace("+","_").replace("/","_")
                l = l.replace(self.htsep,"")
                l_ = l.split()
                if len(l_) > 3:
                    # print(l)
                    self.datas.append([l,[]])
                if len(self.datas) == 500000:
                    break            
        
        # with open(os.path.join(os.environ["DATASET"],"naverblog_sentsplit250205.txt"),encoding="utf-8") as f:
        #     for l in f:
        #         l = l.strip()
        #         l = l.replace("+","_").replace("/","_")
        #         l = l.replace(self.htsep,"")
        #         l_ = l.split()
        #         if len(l_) > 3:
        #             # print(l)
        #             self.datas.append([l,[]])
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self,idx):
        x = self.datas[idx][0]
        y = self.datas[idx][1]
        # print(x)
        input_ = w2i(x)#self.tokenizer(x,return_tensors="pt",truncation=True,max_length=600)
        # print(input_.shape)
        # print(input_)
        # print(len(x))
        # exit()
        input_ = torch.tensor(input_,dtype=torch.long)
        if True:#len(y) == 0:
            output_ = input_.clone()
            prob = torch.full(input_.shape,0.3)
            # prob = prob == 
            # htsepprob = prob == self.tokenizer.convert_tokens_to_ids("+")
            # print(self.tokenizer.decode(self.tokenizer.encode("▁")))
            # sosprob = (input_ == self.lstm_vocab["[SOS]"]) | (input_ == self.lstm_vocab["[EOS]"])
            
            spaceprob = input_ == self.lstm_vocab["▁"]

            # exit()
            # prob[htsepprob] = 0
            # prob[sosprob] = 0
            prob[spaceprob] = 0
            prob[0,0] = 0
            prob[0,-1] = 0
            prob = torch.bernoulli(prob)
            
            prob = prob == 1
            input_[prob] = self.lstm_vocab["[MASK]"]
            # input_ = self.tokenizer.decode(input_["input_ids"][0][1:-1])
            # output_ = self.tokenizer.decode(output_[0][1:-1])
            output_[~prob] = self.lstm_vocab["[PAD]"]
            self.datas[idx][1] = output_
            # print(input_.shape)
            # exit()
        # print(input_["input_ids"][0].shape,self.datas[idx][1].shape)
        return {"input":input_[0].numpy().tolist(),"output":self.datas[idx][1][0].numpy().tolist(),"pad":self.lstm_vocab["[PAD]"]}

def collate_fn(batch):
    maxlen = 0
    
    for b in batch:
        maxlen = max(maxlen,len(b["input"]))
    maxlen = min(400,maxlen)
    # print(maxlen)
    res = {"input":[],"output":[]}
    for b in batch:
        in_ = b["input"] + [b["pad"]] * (maxlen - len(b["input"]))
        in_ = in_[:maxlen]
        out_ = b["output"] + [b["pad"]] * (maxlen - len(b["output"]))
        out_ = out_[:maxlen]
        # print(in_)
        # exit()
        # print(in_,out_)
        # print(in_)
        res["input"].append(in_)
        res["output"].append(out_)
        
    res["input"] = torch.tensor(res["input"])
    res["output"] = torch.tensor(res["output"])
    res["unfold"] = res["input"].clone().detach().unfold(1,3,1)
    return res

if __name__ == "__main__":
    # from torch.utils.data import DataLoader
    import pickle
    with open("lstm_vocab.pkl","rb") as f:
        posvocab = pickle.load(f)
    dataset = Dataset(posvocab)
    training_data, test_data, val_data = torch.utils.data.random_split(dataset,[0.898,0.1,0.002])
    train_dataloader = torch.utils.data.DataLoader(training_data,collate_fn=collate_fn, batch_size=50, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data,collate_fn=collate_fn, batch_size=50, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data,collate_fn=collate_fn, batch_size=50, shuffle=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    # scaler = torch.amp.GradScaler("cuda")
    model = SelfEncoder(posvocab).to("cuda")#,dtype=torch.bfloat16)
    
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
    # model.load_state_dict(torch.load("selfmodel_13",weights_only=True))
    # with torch.amp.autocast("cuda",dtype=torch.bfloat16):
    print(len(training_data),len(test_data),len(val_data))
    if True:
        validstep = 0
        trainstep = 0
        for e in range(20):
            loss__ = 0
            traincount = 0
            model.train()
            trainloss = 0
            for t in train_dataloader:
                # torch.cuda.empty_cache()
                input_ = t["input"].to("cuda")
                output_ = t["output"].to("cuda")
                unfold_ = t["unfold"].to("cuda")
                pred = model(input_)
                
                loss = torch.nn.functional.nll_loss(pred.view(-1,model.config.vocab_size),output_.view(-1),ignore_index=posvocab["[PAD]"])
                
                optimizer.zero_grad()
                # scaler.scale(loss).backward()
                loss.backward()
                optimizer.step()
                loss__ += loss.item()
                traincount += 1
                # if traincount % 100 == 0:
                
                print(f"\r{traincount:06d}/{len(train_dataloader):06d} ,epoch :{e}, {loss__/traincount}",end="")
                # loss__ = 0
                trainloss += loss.item()
                if traincount % 100 == 0:
                    torch.cuda.empty_cache()
                    trainstep += 1
                    writer.add_scalar("SelfLoss/Train",trainloss/100,trainstep)
                    # writer.add_scalar("Avg/Valid",valloss/len(val_dataloader),validstep)
                    writer.flush()
                    trainloss = 0
                    
                if traincount % 500 == 0:
                    model.eval()
                    valloss = 0
                    valresult = 0
                    for t in val_dataloader:
                        
                        input_ = t["input"].to("cuda")
                        output_ = t["output"].to("cuda")
                        unfold_ = t["unfold"].to("cuda")
                        
                        pred = model(input_)
                        loss = torch.nn.functional.nll_loss(pred.view(-1,model.config.vocab_size),output_.view(-1),ignore_index=posvocab["[PAD]"])
                        # loss.requires_grad = False
                        
                        valloss += loss.item()
                        pred = torch.argmax(pred,dim=-1).view(-1)
                        output_ = output_.view(-1)
                        
                        pad = output_ != posvocab["[PAD]"]
                        res = pred[pad] == output_[pad]
                        
                        resultcount = pred[pad][res].shape[0]
                        resultcount = resultcount / output_[pad].shape[0]
                        valresult += resultcount
                    # print("---------------------validation-------------------------")
                    # print(f"\nvalloss: {valloss/len(val_dataloader)}")
                    # print(f"\nvalavg: {valresult/len(val_dataloader)}")
                    # print("--------------------------------------------------------")
                    validstep += 1
                    writer.add_scalar("SelfLoss/Valid",valloss/len(val_dataloader),validstep)
                    writer.add_scalar("SelfAvg/Valid",valresult/len(val_dataloader),validstep)
                    writer.flush()
                    model.train()
            # print(f"\nloss: {loss__/len(test_dataloader)}")
            # print(f"\navg: {result/len(test_dataloader)}")
            # torch.save(model.state_dict(),f"selfmodel_mini_{e}")
            result = 0
            loss__ = 0
            model.eval()
            
            for t in test_dataloader:
                input_ = t["input"].to("cuda")
                output_ = t["output"].to("cuda")
                unfold_ = t["unfold"].to("cuda")
                
                pred = model(input_)
                loss = torch.nn.functional.nll_loss(pred.view(-1,model.config.vocab_size),output_.view(-1),ignore_index=posvocab["[PAD]"])
                # loss.requires_grad = False
                
                loss__ += loss.item()
                pred = torch.argmax(pred,dim=-1).view(-1)
                output_ = output_.view(-1)
                
                pad = output_ != posvocab["[PAD]"]
                res = pred[pad] == output_[pad]
                
                resultcount = pred[pad][res].shape[0]
                resultcount = resultcount / output_[pad].shape[0]
                result += resultcount

            
            # print(f"\nloss: {loss__/len(test_dataloader)}")
            # print(f"\navg: {result/len(test_dataloader)}")
            writer.add_scalar("SelfLoss/Test",loss__/len(test_dataloader),e+1)
            writer.add_scalar("SelfAvg/Test",result/len(test_dataloader),e+1)
            writer.flush()
            torch.save(model.state_dict(),f"selfmodel_mini_{e}")
            # print(t["input"].shape)
            # print(t["input"])
            # print(t["output"])
            # print(len(t["input"][0]))
            # print(len(t["output"][0]))
            # exit()
    writer.close()"""
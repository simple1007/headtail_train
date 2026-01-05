from torch import nn
from transformer import PositionalEncoding, TransformerConfig
import torch

def remove_prefix_from_state_dict(state_dict, prefix='_orig_mod.'):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    return new_state_dict

class PosLSTM(nn.Module):
    def __init__(self,posvocab):
        super(PosLSTM,self).__init__()
        # print(tag_len)
        self.emb = nn.Embedding(len(posvocab.index2word),64)#bilm.emb#nn.Embedding(len(bigram.keys()),80)

        self.emb_lstm = nn.LSTM(64,128,bidirectional=True,batch_first=True)

        self.embbi = nn.Embedding(len(posvocab.index2uni),64)#bilm.emb#nn.Embedding(len(bigram.keys()),80)

        self.embbi_lstm = nn.LSTM(64,128,bidirectional=True,batch_first=True)

        # self.sub_layer = nn.LSTM(256*2,256) 
        # self.sub_layer = nn.LSTM(128,256,batch_first=True) 
        self.output = nn.Linear(128,len(posvocab.index2pos))
        self.output_bio = nn.Linear(128,5)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,bi,data,masks,segments):
        emb = self.emb(data)
        
        emb_lstm,_ = self.emb_lstm(emb)
        emb_lstm = emb_lstm[:,:,:128] + emb_lstm[:,:,128:]
        
        embbi = self.embbi(bi)
        embbi_lstm, _ = self.embbi_lstm(embbi)
        embbi_lstm = embbi_lstm[:,:,:128] + embbi_lstm[:,:,128:]
        
        # output2,_ = self.sub_layer(emb_lstm+embbi_lstm)
        output = self.output(embbi_lstm+emb_lstm)
        output = self.softmax(output)

        output_bio = self.output_bio(embbi_lstm+emb_lstm)
        output_bio = self.softmax(output_bio)
        return output,output_bio

class PosLSTM2(nn.Module):
    def __init__(self,posvocab):
        super(PosLSTM,self).__init__()
        # print(tag_len)
        self.emb = nn.Embedding(len(posvocab.index2word),64)#bilm.emb#nn.Embedding(len(bigram.keys()),80)

        self.emb_lstm = nn.LSTM(64,128,bidirectional=True,batch_first=True)

        self.embbi = nn.Embedding(len(posvocab.index2uni),64)#bilm.emb#nn.Embedding(len(bigram.keys()),80)

        self.embbi_lstm = nn.LSTM(64,128,bidirectional=True,batch_first=True)

        # self.sub_layer = nn.LSTM(256*2,256) 
        self.sub_layer = nn.LSTM(128,256,batch_first=True) 
        self.output = nn.Linear(256,len(posvocab.index2pos))
        self.output2 = nn.Linear(256,len(posvocab.index2pos))
        
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,bi,data,p):
        emb = self.emb(data)
        
        emb_lstm,_ = self.emb_lstm(emb)
        emb_lstm = emb_lstm[:,:,:128] + emb_lstm[:,:,128:]
        
        embbi = self.embbi(bi)
        embbi_lstm, _ = self.embbi_lstm(embbi)
        embbi_lstm = embbi_lstm[:,:,:128] + embbi_lstm[:,:,128:]
        
        output_,_ = self.sub_layer(emb_lstm+embbi_lstm)
        output = self.output(output_)
        output = self.softmax(output)

        output2 = self.output2(output_)
        output2 = self.softmax(output2)

        return output
    

# from torchcrf import CRF
from transformer import Encoder,PositionalEncoding
import pickle
import os

class BiEmbedding2(nn.Module):
    def __init__(self,config: TransformerConfig):
        super(BiEmbedding2,self).__init__()
        self.emb = nn.Embedding(config.vocab_size, config.dmodel)
        # self.pe = PositionalEncoding(config)#.to(device)
        # self.emblayer = nn.Linear(config.dmodel,config.dmodel)
        self.config = config
        # self.pe = nn.Embedding(self.config.maxlen,self.config.dmodel)
        
        self.norm = torch.nn.LayerNorm(self.config.dmodel)
        # self.biemb = nn.Embedding(config.vocab_size,self.config.dmodel)
        # self.lstm = nn.LSTM(self.config.dmodel,self.config.dmodel//2,batch_first=True,bidirectional=True)
        # self.layernorm = nn.LayerNorm(self.config.dmodel)
        self.dropout = nn.Dropout(config.dropout)
        # self.layer = nn.Linear(self.config.dmodel,self.config.dmode)
        # if not subwordvocab:
        if os.environ.get("HT","") != "":
            with open(os.environ["HT"]+os.sep+"subwordvocab.pkl","rb") as f:
                self.subwordvocab = pickle.load(f)
        else:
            with open("subwordvocab.pkl","rb") as f:
                self.subwordvocab = pickle.load(f)
        # else:
        #     self.subwordvocab = subwordvocab
    def setData(self,data):
        self.data = data
    def forward(self,unfold):
        data = self.data
        data_ = unfold#data.unfold(1,3,1)
        cls = torch.full((data.shape[0],1,2),self.subwordvocab.cls).to(unfold.device)
        pad = torch.full((data.shape[0],1,2),self.subwordvocab.pad).to(unfold.device)
        data_ = torch.concat((data_[:,:,0:1],data_[:,:,2:3]),dim=2)
        data_ = torch.concat((cls,data_,pad),dim=1)
        data_ = self.emb(data_)
        data_ = data_[:,:,0,:]+data_[:,:,1,:]#torch.concat((data_[:,:,0,:],data_[:,:,1,:]),dim=-1)
        
        # lstm,_ = self.lstm(data_)
        # print(data.shape)
        # print(data_.shape)
        # exit()
        return data_
        
        cls = torch.full((x.shape[0],1,2),self.subwordvocab.cls).to(x.device)
        pad = torch.full((x.shape[0],1,2),self.subwordvocab.pad).to(x.device)
        x = torch.concat((x[:,:,0:1],x[:,:,2:3]),dim=2)
        x = torch.concat((cls,x,pad),dim=1)
        x = self.emb(x)
        x = x[:,:,0,:]+x[:,:,1,:]#torch.concat((data_[:,:,0,:],data_[:,:,1,:]),dim=-1)
        # lstm,_ = self.lstm(data_)
        emb = self.emb(x)
        
        return emb
import math
class BiEmbedding(nn.Module):
    def __init__(self,config: TransformerConfig):
        super(BiEmbedding,self).__init__()
        self.emb = nn.Embedding(config.vocab_size, config.dmodel)
        self.pe = PositionalEncoding(config)#.to(device)
        self.config = config
        if os.environ.get("HT","") != "":
            with open(os.environ["HT"]+os.sep+"subwordvocab.pkl","rb") as f:
                self.subwordvocab = pickle.load(f)
        else:
            with open("subwordvocab.pkl","rb") as f:
                self.subwordvocab = pickle.load(f)

    def forward(self,x):
        emb = self.emb(x)
        emb = self.pe(emb,emb.shape[1])
        return emb


class PosLSTM3(nn.Module):
    def __init__(self,posvocab):
        super(PosLSTM3,self).__init__()
        self.config = TransformerConfig(pad_idx=posvocab.pad,numheads=4,ngroup=2,group_attn=True,dmodel=256, dff=512,nlayer=3,maxlen=600,vocab_size=15001,dropout=0.1,warmup_steps=100)
        self.model = Encoder(self.config,last_layer=False,embedding=BiEmbedding)#.to(device)
        # self.lstm = nn.LSTM(self.config.dmodel,64,bidirectional=True,batch_first=True)
        # self.biemb = nn.Embedding(15001,self.config.dmodel//2)
        # self.lstm = nn.LSTM(self.config.dmodel,128,batch_first=True,bidirectional=True)
        # self.layernorm = nn.LayerNorm(self.config.dmodel)
        # self.lstm = nn.LSTM(self.config.dmodel,64,batch_first=True,bidirectional=True)
        # self.layer = nn.Linear(128,self.config.dmodel)
        self.vocab_output = torch.nn.Linear(self.config.dmodel,len(posvocab.pos2index))
        # self.bi_output = torch.nn.Linear(self.config.dmodel,self.config.dmodel)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.tag_len = len(posvocab.pos2index)
        
    def forward(self,data):
        # data_ = data.unfold(1,3,1)
        # cls = torch.full((data.shape[0],1,2),subwordvocab.cls).to(data.device)
        # pad = torch.full((data.shape[0],1,2),subwordvocab.pad).to(data.device)
        # data_ = torch.concat((data_[:,:,0:1],data_[:,:,2:3]),dim=2)
        # data_ = torch.concat((cls,data_,pad),dim=1)
        # data_ = self.biemb(data_)
        # data_ = torch.concat((data_[:,:,0,:],data_[:,:,1,:]),dim=-1)
        # lstm,_ = self.lstm(data_)
        
        # self.model.emb.setbigram(lstm)
        if data.dtype != torch.long:
            data = data.type(torch.LongTensor).to(data.device)
        enc = self.model(data,group=True)
        # enc,_ = self.lstm(enc)
        # enc = self.layer(enc)
        
        # print(enc)
        # exit()
        # exit()
        # dense = self.layer((enc*0.8)+(lstm*0.2))
        # enc = self.layernorm(lstm+enc)
        output = self.vocab_output(enc)
        # output2 = self.vocab_output(lstm)
        output = self.softmax(output)
        # output2 = self.softmax(output2)
        # print(torch.sum(output,dim=-1))
        # print(torch.log(torch.exp(output[0])))
        # print(output[0])
        # exit(i
        # output = torch.exp(output)
        # output2 = torch.exp(output2)
        # output = (output * 0.7) + (output2 * 0.3)
        # output = torch.log(output)
        return output

class PosLSTMMiniOld(nn.Module):
    def __init__(self,posvocab):
        super(PosLSTMMiniOld,self).__init__()
        # self.config = TransformerConfig(pad_idx=posvocab.pad,numheads=4,ngroup=4,group_attn=True,dmodel=256, dff=800,nlayer=4,maxlen=600,vocab_size=15001,dropout=0.1,warmup_steps=100)
        self.model = SelfEncoder(posvocab)
        self.config = self.model.config

        self.model.load_state_dict(torch.load(os.path.join(os.environ["KJM"],"headtail","postagger_model","selfmodel_29"),weights_only=True))
        self.model = self.model.model
        #Encoder(self.config,last_layer=False,embedding=BiEmbedding)#.to(device)
        self.subwordvocab = posvocab
        self.norm = torch.nn.LayerNorm(self.config.dmodel)
        self.biemb = nn.Embedding(self.config.vocab_size,self.config.dmodel)
        self.lstm = nn.LSTM(self.config.dmodel,self.config.dmodel//2,batch_first=True,bidirectional=True)

        self.vocab_output = torch.nn.Linear(self.config.dmodel,len(posvocab.pos2index))
        self.softmax = nn.LogSoftmax(dim=-1)
        self.tag_len = len(posvocab.pos2index)
        
    def forward(self,data):
        # print(data)
        if data.dtype != torch.long:
            data = data.type(torch.LongTensor).to(data.device)
        enc = self.model(data,group=True)
        lstm = self.setbigram(data)
        norm = self.norm(lstm+enc)
        output = self.vocab_output(norm)
        output = self.softmax(output)
        return output
    def setbigram(self,data):
        data_ = data.unfold(1,3,1)
        cls = torch.full((data.shape[0],1,2),self.subwordvocab.cls).to(data.device)
        pad = torch.full((data.shape[0],1,2),self.subwordvocab.pad).to(data.device)
        data_ = torch.concat((data_[:,:,0:1],data_[:,:,2:3]),dim=2)
        data_ = torch.concat((cls,data_,pad),dim=1)
        data_ = self.biemb(data_)
        data_ = data_[:,:,0,:]+data_[:,:,1,:]#torch.concat((data_[:,:,0,:],data_[:,:,1,:]),dim=-1)
        lstm,_ = self.lstm(data_)
        
        return lstm
import copy
class PosLSTMMini2(nn.Module):
    def __init__(self,posvocab):
        super(PosLSTMMini2,self).__init__()
        self.subwordvocab = posvocab
        self.model = SelfEncoder(posvocab.pad)
        self.config = self.model.config
        
        self.sublayer = torch.nn.Linear(self.config.dmodel,self.config.dmodel//4)
        self.act = torch.nn.GELU()
        self.layer_norm = nn.LayerNorm(self.config.dmodel // 4)

        self.vocab_output = nn.Linear(self.config.dmodel//4,len(posvocab.pos2index))
        self.tag_len = len(posvocab.pos2index)
        self.init_()
    
    def init_(self):
        torch.nn.init.trunc_normal_(self.sublayer.weight, std=0.02)
        self.sublayer.bias = torch.nn.Parameter(torch.zeros(self.config.dmodel//4))

        torch.nn.init.trunc_normal_(self.vocab_output.weight, std=0.02)
        self.vocab_output.bias = torch.nn.Parameter(torch.zeros(self.tag_len))

    def forward(self,data,unfold):
        if data.dtype != torch.long:
            data = data.type(torch.LongTensor).to(data.device)
        
        enc = self.model(data,is_softmax=True)
        enc = self.sublayer(enc)
        enc = self.act(enc)
        output = self.layer_norm(enc)
        output = self.vocab_output(output)
        return output
    
class PosLSTMMini2CPP(nn.Module):
    def __init__(self,pad,outsize):
        super(PosLSTMMini2CPP,self).__init__()
        # self.subwordvocab = posvocab
        self.model = SelfEncoder(pad=pad)
        self.config = self.model.config

        self.norm = nn.LayerNorm(self.config.dmodel)
        self.dropout = nn.Dropout(0.1)
        self.output = nn.LSTM(self.config.dmodel,self.config.dmodel//2,batch_first=True,bidirectional=True)
        self.vocab_output = nn.Linear(self.config.dmodel,outsize)
        self.tag_len = outsize#len(posvocab.pos2index)
        
    def forward(self,data):
        if data.dtype != torch.long:
            data = data.type(torch.LongTensor).to(data.device)
        
        # data = torch.tensor([a]).to(data.device)
        enc = self.model(data,is_softmax=True)

        output,_ = self.output(enc)
        output = self.vocab_output(output)
        output_softmax = torch.nn.functional.log_softmax(output,dim=-1)
        output_softmax = torch.argmax(output_softmax,dim=-1)
        # print(output_softmax)
        # print(output_softmax.shape)
        # print(output.shape)
        # output_softmax = nn.functional.softmax(output,dim=-1)
        # output_softmax = torch.argmax(output_softmax,dim=-1)
        return [output_softmax]#.reshape(output.shape)

class PosLSTMMini2OLD(nn.Module):
    def __init__(self,posvocab):
        super(PosLSTMMini2OLD,self).__init__()
        # self.config = TransformerConfig(pad_idx=posvocab.pad,numheads=4,ngroup=4,group_attn=True,dmodel=256, dff=800,nlayer=4,maxlen=600,vocab_size=15001,dropout=0.1,warmup_steps=100)
        self.model = SelfEncoder(posvocab)
        self.config = self.model.config
        print(self.config)
        # self.model.load_state_dict(torch.load(os.path.join(os.environ["KJM"],"headtail","postagger_model","selfmodel_mini_9"),weights_only=True))
        # self.emb = self.model.emb
        # self.model = self.model.model
        
        #Encoder(self.config,last_layer=False,embedding=BiEmbedding)#.to(device)
        self.subwordvocab = posvocab
        self.norm = torch.nn.LayerNorm(self.config.dmodel)
        self.biemb = nn.Embedding(self.config.vocab_size,self.config.dmodel)
        self.lstm = nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)#nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)

        self.vocab_output = nn.Linear(self.config.dmodel,len(posvocab.pos2index))
        self.softmax = nn.LogSoftmax(dim=-1)
        self.tag_len = len(posvocab.pos2index)
        
    def forward(self,data,unfold):
        # print(data)
        if data.dtype != torch.long:
            data = data.type(torch.LongTensor).to(data.device)
            unfold = unfold.type(torch.LongTensor).to(data.device)
        # emb = self.emb(data)
        enc = self.model(data,is_softmax=True)
        lstm = self.setbigram(data,unfold)
        # norm = self.norm(lstm+enc)
        output = self.vocab_output(enc+lstm)
        output = self.softmax(output)
        return output
    
    def setbigram(self,data,unfold):
        data_ = unfold#data.unfold(1,3,1)
        cls = torch.full((data.shape[0],1,2),self.subwordvocab.cls).to(data.device)
        pad = torch.full((data.shape[0],1,2),self.subwordvocab.pad).to(data.device)
        data_ = torch.concat((data_[:,:,0:1],data_[:,:,2:3]),dim=2)
        data_ = torch.concat((cls,data_,pad),dim=1)
        data_ = self.biemb(data_)
        data_ = data_[:,:,0,:]+data_[:,:,1,:]#torch.concat((data_[:,:,0,:],data_[:,:,1,:]),dim=-1)
        lstm,_ = self.lstm(data_)
        
        return lstm


class PosLSTMMini3(nn.Module):
    def __init__(self,posvocab):
        super(PosLSTMMini3,self).__init__()
        self.model = SelfEncoderMini(posvocab)
        self.config = self.model.config

        self.model.load_state_dict(torch.load(os.path.join(os.environ["KJM"],"headtail","postagger_model","selfmodel_mini_9"),weights_only=True))

        self.vocab_output = torch.nn.Linear(self.config.dmodel,len(posvocab.pos2index))
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self,data,unfold):
        if data.dtype != torch.long:
            data = data.type(torch.LongTensor).to(data.device)
            unfold = unfold.type(torch.LongTensor).to(data.device)
        enc = self.model(data,unfold,is_softmax=True)
        output = self.vocab_output(enc)
        output = self.softmax(output)
        return output
    
class SelfEncoderMini(nn.Module):
    def __init__(self,posvocab):
        super(SelfEncoderMini,self).__init__()
        self.config = TransformerConfig(pad_idx=posvocab.pad,numheads=8,ngroup=4,group_attn=True,dmodel=256, dff=800,nlayer=4,maxlen=600,vocab_size=15001,dropout=0.1,warmup_steps=100)
        self.emb = nn.Embedding(self.config.vocab_size,self.config.dmodel)
        self.model = nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)#Encoder(self.config,last_layer=False,embedding=BiEmbedding)#.to(device)
        self.layer = nn.Linear(self.config.dmodel,self.config.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.subwordvocab = posvocab
        # self.norm = torch.nn.LayerNorm(self.config.dmodel)
        # self.biemb = nn.E(self.config.vocab_size,self.config.dmodel)
        # self.lstm = nn.LSTM(self.config.dmodel,self.config.dmodel,batch_first=True)

    
    def forward(self,x,unfold,is_softmax=False):
        emb = self.emb(x)
        enc_out,_ = self.model(emb)
        # unfold = self.setbigram(x,unfold)
        # enc_out = self.norm(enc_out+unfold)
        
        # if not is_softmax:
        vocab = self.layer(enc_out)
        
        out = self.softmax(vocab)
    
        return out
        
        # return enc_out
    
    def setbigram(self,data,unfold):
        data_ = unfold#data.unfold(1,3,1)
        cls = torch.full((data.shape[0],1,2),self.subwordvocab.cls).to(data.device)
        pad = torch.full((data.shape[0],1,2),self.subwordvocab.pad).to(data.device)
        data_ = torch.concat((data_[:,:,0:1],data_[:,:,2:3]),dim=2)
        data_ = torch.concat((cls,data_,pad),dim=1)
        data_ = self.biemb(data_)
        data_ = data_[:,:,0,:]+data_[:,:,1,:]#torch.concat((data_[:,:,0,:],data_[:,:,1,:]),dim=-1)
        lstm,_ = self.lstm(data_)
        
        return lstm
    


    
class SelfEncoder(nn.Module):
    def __init__(self,pad,vocab_size=None,biemb=False):
        super(SelfEncoder,self).__init__()
        self.config = TransformerConfig(pad_idx=pad,numheads=4,group_attn=False,dmodel=128, dff=256,nlayer=2,maxlen=600,vocab_size=10001,dropout=0.01,warmup_steps=100)
        
        if vocab_size:
            self.config.vocab_size = vocab_size

        if biemb:
            self.model = Encoder(self.config,last_layer=False,embedding=BiEmbedding2)
        else:
            self.model = Encoder(self.config,last_layer=False,embedding=BiEmbedding)#.to(device)
        self.layer = nn.Linear(self.config.dmodel,self.config.vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x,is_softmax=False):
        # emb = self.emb(x)
        enc_out = self.model(x)
        # unfold = self.setbigram(x,unfold)
        # enc_out = self.norm(enc_out+unfold)
        
        if not is_softmax:
            vocab = self.layer(enc_out)
            
            out = self.softmax(vocab)
        
            return out
        
        return enc_out
    
    # def setbigram(self,data,unfold):
    #     data_ = unfold#data.unfold(1,3,1)
    #     cls = torch.full((data.shape[0],1,2),self.subwordvocab.cls).to(data.device)
    #     pad = torch.full((data.shape[0],1,2),self.subwordvocab.pad).to(data.device)
    #     data_ = torch.concat((data_[:,:,0:1],data_[:,:,2:3]),dim=2)
    #     data_ = torch.concat((cls,data_,pad),dim=1)
    #     data_ = self.biemb(data_)
    #     data_ = data_[:,:,0,:]+data_[:,:,1,:]#torch.concat((data_[:,:,0,:],data_[:,:,1,:]),dim=-1)
    #     lstm,_ = self.lstm(data_)
        
    #     return lstm

class SelfEncoder2(nn.Module):
    def __init__(self,posvocab):
        super(SelfEncoder2,self).__init__()
        # self.config = TransformerConfig(pad_idx=posvocab.pad,numheads=4,ngroup=2,group_attn=True,dmodel=256, dff=512,nlayer=4,maxlen=600,vocab_size=15001,dropout=0.1,warmup_steps=100)
        self.config = TransformerConfig(pad_idx=posvocab.pad,numheads=8,group_attn=False,dmodel=128, dff=1024,nlayer=3,maxlen=600,vocab_size=15001,dropout=0.1,warmup_steps=100)
        
        self.emb = nn.Embedding(self.config.vocab_size,self.config.dmodel)
        self.model = nn.LSTM(self.config.dmodel,self.config.dmodel//2,batch_first=True,bidirectional=True,num_layers=2)#Encoder(self.config,last_layer=False,embedding=BiEmbedding)#.to(device)
        self.layer = nn.Linear(self.config.dmodel,self.config.vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
        self.subwordvocab = posvocab
        
        # self.norm = torch.nn.LayerNorm(self.config.dmodel)
        # self.biemb = nn.Embedding(self.config.vocab_size,self.config.dmodel)
        # self.lstm = nn.LSTM(self.config.dmodel,self.config.dmodel//2,batch_first=True,bidirectional=True)

    
    def forward(self,x,is_softmax=False):
        # emb = self.emb(x)
        emb = self.emb(x)
        enc_out,_ = self.model(emb)
        # unfold = self.setbigram(x,unfold)
        # enc_out = self.norm(enc_out+unfold)
        
        if not is_softmax:
            vocab = self.layer(enc_out)
            
            out = self.softmax(vocab)
        
            return out
        
        return enc_out
    
    def setbigram(self,data,unfold):
        data_ = unfold#data.unfold(1,3,1)
        cls = torch.full((data.shape[0],1,2),self.subwordvocab.cls).to(data.device)
        pad = torch.full((data.shape[0],1,2),self.subwordvocab.pad).to(data.device)
        data_ = torch.concat((data_[:,:,0:1],data_[:,:,2:3]),dim=2)
        data_ = torch.concat((cls,data_,pad),dim=1)
        data_ = self.biemb(data_)
        data_ = data_[:,:,0,:]+data_[:,:,1,:]#torch.concat((data_[:,:,0,:],data_[:,:,1,:]),dim=-1)
        lstm,_ = self.lstm(data_)
        
        return lstm
from ht_utils import HTInputTokenizer
from transformers import AutoTokenizer

# import math

class Dataset(torch.utils.data.Dataset):
    def __init__(self,posvocab:HTInputTokenizer):
        self.htsep = posvocab.htsep
        # self.tagsep = tagsep
        self.tokenizer:AutoTokenizer = posvocab.tokenizer
        filename = os.path.join(os.environ["DATASET"],"ht_dataset","modu","result","delkiwimorphs.txt")
        self.datas = []
        
        with open(filename,encoding="utf-8") as f:
            for l in f:
                l = l.strip()
                l = l.replace("+","_").replace("/","_")
                l = l.replace(self.htsep," + ")
                l_ = l.split()
                if len(l_) > 3:
                    # print(l)
                    self.datas.append([l,[]])
                if len(self.datas) == 300000:
                    break
        
        # linecnt = 0
        # with open(os.path.join(os.environ["DATASET"],"naverblog_sentsplit250205.txt"),encoding="utf-8") as f:
        #     for l in f:
        #         l = l.strip()
        #         l = l.replace("+","_").replace("/","_")
        #         l = l.replace(self.htsep,"")
        #         l_ = l.split()
        #         if len(l_) > 3:
        #             # print(l)
        #             self.datas.append([l,[]])
        #             linecnt += 1
        #         if linecnt == 100000:
        #             break
    
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self,idx):
        x = self.datas[idx][0]
        y = self.datas[idx][1]
        
        input_ = self.tokenizer(x,return_tensors="pt",truncation=True,max_length=600)
        if True:#len(y) == 0:
            output_ = input_["input_ids"].clone()
            prob = torch.full(input_["input_ids"].shape,0.15)
            # prob = torch.bernoulli(prob)
            # prob = prob == 
            htsepprob = input_ == self.tokenizer.convert_tokens_to_ids("+")
            # print(self.tokenizer.decode(self.tokenizer.encode("▁")))
            spaceprob = input_ == self.tokenizer.convert_tokens_to_ids("▁")
            # sosprob = (input_ == self.tokenizer.sep_token_id) | (self.tokenizer.cls_token_id)
            # exit()
            
            prob[htsepprob] = 0
            prob[spaceprob] = 0
            prob[0,0] = 0
            # prob[0,-1] = 0
            sep = input_ == self.tokenizer.convert_tokens_to_ids("[SEP]")
            prob[sep] = 0
            prob = torch.bernoulli(prob)

            probmlm = prob * 0.8
            probmlm = torch.bernoulli(probmlm)
            
            probtmp = prob
            probtmp[probmlm==1] = 0
            probrandom = torch.bernoulli(probtmp*0.5)
            
            # prob = probmlm == 1
            input_["input_ids"][probmlm==1] = self.tokenizer.convert_tokens_to_ids("[MASK]")
            input_["input_ids"][probrandom==1] = torch.randint_like(input_["input_ids"][probrandom==1],0,10000)#[random for _ in range((probrandom==1).shape[1])  ]
            # input_ = self.tokenizer.decode(input_["input_ids"][0][1:-1])
            # output_ = self.tokenizer.decode(output_[0][1:-1])
            prob = prob == 1
            output_[~prob] = self.tokenizer.convert_tokens_to_ids("[PAD]")
            self.datas[idx][1] = output_
        # print(input_["input_ids"][0].shape,self.datas[idx][1].shape)
        return {"input":input_["input_ids"][0].numpy().tolist(),"output":self.datas[idx][1][0].numpy().tolist(),"pad":self.tokenizer.convert_tokens_to_ids("[PAD]")}

def collate_fn(batch):
    maxlen = 0
    
    for b in batch:
        maxlen = max(maxlen,len(b["input"]))
    maxlen = min(600,maxlen)
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
        res["input"].append(in_)
        res["output"].append(out_)
    res["input"] = torch.tensor(res["input"])
    res["output"] = torch.tensor(res["output"])
    res["unfold"] = res["input"].clone().detach().unfold(1,3,1)
    return res

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    # from torch.utils.data import DataLoader
    import pickle
    with open("subwordvocab.pkl","rb") as f:
        posvocab = pickle.load(f)
    dataset = Dataset(posvocab)
    training_data, test_data, val_data = torch.utils.data.random_split(dataset,[0.898,0.1,0.002])
    train_dataloader = torch.utils.data.DataLoader(training_data,collate_fn=collate_fn, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_data,collate_fn=collate_fn, batch_size=64, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_data,collate_fn=collate_fn, batch_size=64, shuffle=True)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()
    # scaler = torch.amp.GradScaler("cuda")
    model = SelfEncoder2(posvocab).to("cuda")#,dtype=torch.bfloat16)
    model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
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
                
                loss = torch.nn.functional.nll_loss(pred.view(-1,model.config.vocab_size),output_.view(-1),ignore_index=posvocab.tokenizer.convert_tokens_to_ids("[PAD]"))
                
                optimizer.zero_grad()
                # scaler.scale(loss).backward()
                
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(),7)
                optimizer.step()
                loss__ += loss.item()
                traincount += 1
                # if traincount % 100 == 0:
                
                print(f"\r{traincount:06d}/{len(train_dataloader):06d} ,epoch :{e}, {loss__/traincount}",end="")
                # loss__ = 0
                trainloss += loss.item()
                if traincount % 500 == 0:
                    torch.cuda.empty_cache()
                    trainstep += 1
                    writer.add_scalar("SelfLoss/Train",trainloss/500,trainstep)
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
                        loss = torch.nn.functional.nll_loss(pred.view(-1,model.config.vocab_size),output_.view(-1),ignore_index=posvocab.tokenizer.convert_tokens_to_ids("[PAD]"))
                        # loss.requires_grad = False
                        
                        valloss += loss.item()
                        pred = torch.argmax(pred,dim=-1).view(-1)
                        output_ = output_.view(-1)
                        
                        pad = output_ != posvocab.tokenizer.convert_tokens_to_ids("[PAD]")
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
                loss = torch.nn.functional.nll_loss(pred.view(-1,model.config.vocab_size),output_.view(-1),ignore_index=posvocab.tokenizer.convert_tokens_to_ids("[PAD]"))
                # loss.requires_grad = False
                
                loss__ += loss.item()
                pred = torch.argmax(pred,dim=-1).view(-1)
                output_ = output_.view(-1)
                
                pad = output_ != posvocab.tokenizer.convert_tokens_to_ids("[PAD]")
                res = pred[pad] == output_[pad]
                
                resultcount = pred[pad][res].shape[0]
                resultcount = resultcount / output_[pad].shape[0]
                result += resultcount


            
            # print(f"\nloss: {loss__/len(test_dataloader)}")
            # print(f"\navg: {result/len(test_dataloader)}")
            writer.add_scalar("SelfLoss/Test",loss__/len(test_dataloader),e+1)
            writer.add_scalar("SelfAvg/Test",result/len(test_dataloader),e+1)
            writer.flush()
            # state_dict = remove_prefix_from_state_dict(model.state_dict())
            torch.save(model.state_dict(),f"selfmodel_mini_{e}")
            # print(t["input"].shape)
            # print(t["input"])
            # print(t["output"])
            # print(len(t["input"][0]))
            # print(len(t["output"][0]))
            # exit()
    writer.close()

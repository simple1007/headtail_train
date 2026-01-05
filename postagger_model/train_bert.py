# from transformers import TFBertForTokenClassification
# from tensorflow.keras.layers import LSTM,Input,Dropout, Bidirectional,Embedding,TimeDistributed,Dense
# from tensorflow.keras.models import Model
# from transformers import create_optimizer
# from transformers import TFBertModel

import pickle
# import tensorflow_addons as tfa
import numpy as np
import datetime
# import tensorflow as tf
import pickle
import os
import argparse
# from gensim.models import Word2Vec
from transformers import BertModel, DistilBertModel, AdamW
from tokenization_kobert import KoBertTokenizer
# from peft import get_peft_model, LoraConfig, TaskType
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import random_split
from torch.quantization import QuantStub, DeQuantStub, float_qparams_weight_only_qconfig, default_qconfig
#from bigramlm import BiGramLM#, vocab_size
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert') # monologg/distilkobert??
# bert_model = BertModel.from_pretrained('monologg/kobert')
# model = BertModel.from_pretrained('monologg/kobert')
model = DistilBertModel.from_pretrained('monologg/distilkobert')
# print(TaskType)
# peft_config = LoraConfig(
#     task_type=TaskType.FEATURE_EXTRACTION ,inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
# )
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# model = get_peft_model(model, peft_config)
# model = model.merge_and_unload()
# print(count_parameters(model)) #¬©
# inputs = tokenizer("HU")
# print(inputs)
import torch
import torch.nn as nn
# inputs["input_ids"] = torch.tensor([inputs["input_ids"]])
# inputs["token_type_ids"] = torch.tensor([inputs["token_type_ids"]])
# inputs["attention_mask"] = torch.tensor([inputs["attention_mask"]])
# print(model(**inputs).last_hidden_state.shape)
#with open("ht_lm_dict.pkl","rb") as f:
#:    map_dict = pickle.load(f)
with open('kcc150_all_tag_dict.pkl','rb') as f:
    tag_dict = pickle.load(f)
#vocab_size = len(map_dict)
tag_len = len(tag_dict.keys())

parser = argparse.ArgumentParser(description="Postagger")

parser.add_argument("--MAX_LEN",type=int,help="MAX Sequnce Length",default=200)
parser.add_argument("--BATCH",type=int,help="BATCH Size",default=50)
parser.add_argument("--EPOCH",type=int,help="EPOCH Size",default=5)
parser.add_argument("--epoch_step",type=int,help="Train Data Epoch Step",default=4000)
parser.add_argument("--validation_step",type=int,help="Validation Data Epoch Step",default=240)
parser.add_argument("--hidden_state",type=int,help="BiLstm Hidden State",default=tag_len*2)
parser.add_argument("--GPU_NUM",type=str,help="Train GPU NUM",default="0")
parser.add_argument("--model_name",type=str,help="Tokenizer Model Name",default="tkbigram_one_first_alltag_bert_tagger.model")

args = parser.parse_args()

if False:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_NUM

EPOCH = args.EPOCH
count_data = args.epoch_step#4000
validation_step = args.validation_step
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
max_len = args.MAX_LEN
# bilm = BiGramLM(vocab_size)
# bilm.load_state_dict(torch.load("lm/lm_5"))
# bilm = bilm.to(device)

with open("train_data.txt") as f:
    tk_datas = f.readlines()
#w2v = Word2Vec.load("word2vec.model")
#k2i = w2v.wv.key_to_index

#mk = max(k2i.values()) + 1
#k2i["[UNK]"] = mk

#i2k = w2v.wv.index_to_key
# print(i2k[-1])
#i2k.append("[UNK]")

#weight = w2v.wv.vectors
#zerovec = np.array([np.zeros(weight[0].shape[0],dtype=np.float64)])
# exit()
# print(weight.shape)
#weight = np.concatenate([weight,zerovec],axis=0)
# print(weight.shape)
# print(model.wv.most_similar(unk))
# print(type(model.wv.vectors[0]))
# print(k2i["[UNK]"])
# print(i2k)
# print(weight[mk])
#w2v.wv.index_to_key = i2k
#w2v.wv.key_to_index = k2i
#w2v.wv.vectors = weight
from collections import deque
from pos_vocab import PosVocab,TokenPosVocab

posvocab = PosVocab(max_len)#PosVocab(max_len)
# length = []
#with open("morphs.txt",encoding="utf-8") as m, open("tags.txt",encoding="utf-8") as p:
#    for i in range((count_data+validation_step) * args.BATCH):
        # print(m.readline(),p.readline())
        # continue
#        mtk, mp = m.readline(),p.readline()
        # length.append(len(mtk.strip()))
#        posvocab.make_dict(mtk,mp)
# print(max(length),sum(length)/len(length),sorted(length,reverse=True)[:40])
# exit()
#tag_len = len(posvocab.index2pos)
def htdataset():
    for _ in range(EPOCH):
        X = []
        Y = []
        with open("morphs.txt",encoding="utf-8") as m, open("tags.txt",encoding="utf-8") as p:
            for i in range((count_data+validation_step) * args.BATCH):
                # print(m.readline(),p.readline())
                # continue
                mtk, mp = m.readline(),p.readline()
                posvocab.make_dict(mtk,mp)
                X.append(mtk)
                Y.append(mp)
                
                if len(X) != 0 and len(X) % args.BATCH == 0:
                    X,Y = posvocab.make_batch(X,Y)
                    yield torch.tensor([[0]]),torch.tensor(X),torch.tensor([[0]]),torch.tensor([[0]]),torch.tensor(Y)
                     
                if len(X) != 0 and len(X) % args.BATCH == 0:
                    X = []
                    Y = []
def dataset():
    for _ in range(EPOCH):
        for i in range(count_data):
            masks = []
            segments = []
            
            data = np.load('kcc150_data/%05d_x.npy' % i)
            y = np.load('kcc150_data/%05d_y.npy' % i)
            bi = np.load('kcc150_data/%05d_tk_bigram.npy' % i)
            lmx = []
            # tks = tk_datas[args.BATCH*i:args.BATCH*i+args.BATCH]
            for d in data:
                # text = ["[SOS]"] + list(tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(d)).replace("[CLS]","").replace("[SEP]","").replace("[PAD]","").strip())[:198] + ["[EOS]"]#.split()
                # text = ["[SOS]"] + list(tks_.strip())[:198] + ["[EOS]"]#.split()
                #text = tks_.strip().replace("+"," ").split()
                #lmx_ = []
                #text = deque(text)
                # print(d)
                #tks = tokenizer.convert_ids_to_tokens(d)
                
                #tempx = []
                #for tk in tks:
                #    if tk.startswith("?"):
                #        tempx.append(text.popleft())
                #    else:
                #        tempx.append("[UNK]")
                # exit()
                # print(text)
                # print(tks_)
                #text = tempx
                #for text_ in text:
                #    if text_ in w2v.wv.key_to_index:
                #        lmx_.append(w2v.wv.key_to_index[text_])
                #    else:
                #        lmx_.append(w2v.wv.key_to_index["[UNK]"])
                #lmx_ = lmx_ + ([w2v.wv.key_to_index["[UNK]"]] * (max_len - len(lmx_)))
                #lmx.append(lmx_[:200])
                # print(text)
                # lmx = torch.tensor([lmx]).to(device)
                # emb = bilm.emb(lmx)
                # out = bilm.lstm(emb)[0]
                # print(out.shape)
                # exit()
                d_ = [t for t in d if t != 1]
                mask = [1]*len(d_) + [0] * (max_len-len(d_))
                masks.append(mask)
                segment = [0] * max_len
                segments.append(segment)
            masks = np.array(masks)
            segments = np.array(segments)
            bi = torch.tensor(bi)
            data = torch.tensor(data)
            masks = torch.tensor(masks)
            segments = torch.tensor(segments)
            #lmx = torch.tensor(lmx)
            # print(lmx.shape)
            y = torch.tensor(y)
            yield bi,data,masks,segments,y
# next(dataset())
# nnnn = dataset()
# for i in range(2000):
#     print(next(nnnn)[0])
# exit()
def validation():
    for _ in range(EPOCH):
        for i in range(count_data,count_data+validation_step):
            masks = []
            segments = []
            lmx = []
            # print('kcc150_data/%05d_x.npy' % i)
            data = np.load('kcc150_data/%05d_x.npy' % i)
            y = np.load('kcc150_data/%05d_y.npy' % i)
            bi = np.load('kcc150_data/%05d_tk_bigram.npy' % i)
            # tks = tk_datas[args.BATCH*i:args.BATCH*i+args.BATCH]
            for d in data:
                #text = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(d)).replace("[CLS]","").replace("[SEP]","").replace("[PAD]","").strip()
                # lmx_ = []
                #text = text.strip().replace("+"," ").split()
                #lmx_ = []
                #text = deque(text)
                # print(d)
                #tks = tokenizer.convert_ids_to_tokens(d)
                
                #tempx = []
                #for tk in tks:
                #    if tk.startswith("?"):
                #        tempx.append(text.popleft())
                #    else:
                #        tempx.append("[UNK]")
                # exit()
                # print(text)
                # print(tks_)
                #text = tempx
                # print(text)
                # print(tks_)
                #for text_ in text:
                #    if text_ in w2v.wv.key_to_index:
                #        lmx_.append(w2v.wv.key_to_index[text_])
                #    else:
                #        lmx_.append(w2v.wv.key_to_index["[UNK]"])
                #lmx_ = lmx_ + ([w2v.wv.key_to_index["[UNK]"]] * (max_len - len(lmx_)))
                #lmx.append(lmx_[:200])
                # for text_ in text:
                #     if text_ in map_dict:
                #         lmx_.append(map_dict[text_])
                # lmx_ = lmx_ + ([map_dict["[PAD]"]] * (max_len - len(lmx_)))
                # lmx.append(lmx_)
                d_ = [t for t in d if t != 1]
                mask = [1]*len(d_) + [0] * (max_len-len(d_))
                masks.append(mask)
                segment = [0] * max_len
                segments.append(segment)
            masks = np.array(masks)
            segments = np.array(segments)

            # masks = np.array(masks)
            # segments = np.array(segments)
            bi = torch.tensor(bi)
            data = torch.tensor(data)
            masks = torch.tensor(masks)
            segments = torch.tensor(segments)
            #lmx = torch.tensor(lmx)
            y = torch.tensor(y)
            yield bi,data,masks,segments,y


with open('kcc150_all_tokenbi.pkl','rb') as f:
    bigram = pickle.load(f)

# class EmbeddingWithLinear(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.emb = torch.nn.Embedding(num_embeddings=10, embedding_dim=12)
#         self.fc = torch.nn.Linear(5, 5)

#     def forward(self, indices, linear_in):
#         return self.emb(indices), self.fc(linear_in)
#class PosTaggerModelPeftTorch(nn.Module):
#    def __init__(self,model,maxlen,hidden_state,tag_len):
    #     super(PosTaggerModelPeftTorch,self).__init__()
 
    #     self.emb = nn.Embedding(len(posvocab.word2index),64)#bilm.emb#nn.Embedding(len(bigram.keys()),80)
 
    #     self.emb_lstm = nn.LSTM(64,128,bidirectional=True,batch_first=True)

    #     self.sub_layer = nn.Linear(128*2,len(posvocab.pos2index))

    #     self.softmax = nn.LogSoftmax(dim=-1)

    # def forward(self,bi,data,masks,segments):

    #     emb = self.emb(data)

    #     emb_lstm,_ = self.emb_lstm(emb)

    #     output = self.sub_layer(emb_lstm)
    #     output = self.softmax(output)

    #     return output
class PosTaggerModelPeftTorch(nn.Module):
    def __init__(self,model,maxlen,hidden_state,tag_len):
        super(PosTaggerModelPeftTorch,self).__init__()
        # self.quant1 = torch.ao.quantization.QuantStub()
        # self.quant2 = torch.ao.quantization.QuantStub()
        # self.quant3 = torch.ao.quantization.QuantStub()
        # self.quant4 = torch.ao.quantization.QuantStub()
        self.emb = nn.Embedding(len(bigram.keys()),128)#bilm.emb#nn.Embedding(len(bigram.keys()),80)
        # print(w2v.wv.vectors[0])
        # exit()
        # self.emb = nn.Embedding.from_pretrained(torch.FloatTensor(w2v.wv.vectors))
        # self.emb.qconfig = float_qparams_weight_only_qconfig
        # self.emb.qconfig = float_qparams_weight_only_qconfig
        # self.qconfig = default_qconfig
        self.emb_lstm = nn.LSTM(128,768,batch_first=True)
        # self.emb_lstm = nn.LSTM(100,128)
        # self.emb_lstm1 = nn.Linear(80,128)
        # self.emb_lstm2 = nn.Linear(80,128)
        # self.emb_lstm3 = nn.Linear(80,128)
        # emb_lstm = tf.expand_dims(emb_lstm,1)
        self.bert = model
        # self.bilstm = nn.LSTM(hidden_state,)
        # self.sub_layer = nn.Linear(768,64)
        self.bilstm = nn.LSTM(768,64*2,batch_first=True)
        self.output_tag = nn.Linear(64 * 2,tag_len)

        self.softmax = nn.LogSoftmax(dim=-1)
        # self.dequant = torch.ao.quantization.DeQuantStub()
        # token_inputs = tf.keras.layers.Input((self.max_len,), dtype=tf.int32, name='input_word_ids')
        # mask_inputs = tf.keras.layers.Input((self.max_len,), dtype=tf.int32, name='input_masks')
        # segment_inputs = tf.keras.layers.Input((self.max_len,), dtype=tf.int32, name='input_segment')
        # outputs = bmodel.bert(input_ids=token_inputs,attention_mask = mask_inputs,token_type_ids = segment_inputs)[0]

        # outputs = tf.keras.layers.Concatenate(axis=-2)([emb_lstm, outputs[:,1:,:]])

        # lstm = Bidirectional(LSTM(self.hidden_state,return_sequences=True,dropout=0.1))(outputs)
        # lstm = TimeDistributed(Dense(self.tag_len, activation='softmax'))(lstm)

        # self.bert = model
    def forward(self,bi,data,masks,segments):
        emb = self.emb(bi)
        emb_lstm,_ = self.emb_lstm(emb)
        bert_output = self.bert(input_ids=data,attention_mask = masks)[0]#,token_type_ids = segments)[0]

        # bert_output = self.sub_layer(bert_output)
        
        # exit()
        emb_lstm = emb_lstm[:,-1,:].unsqueeze(1)
        # print(emb_lstm.shape,bert_output[:,1:,:].shape)
        output = torch.concat([emb_lstm,bert_output[:,1:,:]],dim=-2)

        output,_ = self.bilstm(output)#(output)
        output = self.output_tag(output)

        output = self.softmax(output)

        return output


def train(config,optimizer,scheduler,loss,model,bi,x,y,device):
    model.train()
    # model.train()
    bi = bi.to(device)
    x[0] = x[0].to(device)
    x[1] = x[1].to(device)
    x[2] = x[2].to(device)
    
    # ht = ht.to(device)

    y = y.to(device)

    y_pred = model(bi,x[0],x[1],x[2])
    # print(y_pred[0].shape,y_female.shape)
    # y_pred = y_pred.view(y_pred.shape[0],y_pred.shape[1] * y_pred.shape[2])
    y_pred = y_pred.view(-1, tag_len)
    y = y.view(-1)
    y = y.type(torch.LongTensor).to(device)
    loss_ = loss(y_pred,y)
    # print(y_pred.shape,y.shape)
    # exit()

    optimizer.zero_grad()
    loss_.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),config["max_grad_norm"])

    optimizer.step()
    scheduler.step()

    return loss_.item()

def eval(val_dataloader,device,model):
    with torch.no_grad():
        count = 0
        result = 0
        print("what")
        from tqdm import tqdm
        for index in tqdm(range(validation_step)):
            bi_,data,masks,segments,y_ = next(val_dataloader)
            x_ = [data,masks,segments]
            # count+=1
            bi_ = bi_.to(device)
            
            x_[0] = x_[0].to(device)
            x_[1] = x_[1].to(device)
            x_[2] = x_[2].to(device)
           
            y_ = y_.to(device)
            y_ = y_.type(torch.LongTensor).to(device)
            yy = model(bi_,x_[0],x_[1],x_[2])

            yy = yy.to(device)
            
            # loss_ = loss(yy,y_)
            
            # vl = (l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9 + l10) / 10
            # vs += loss_.item()
            _,yy = torch.topk(yy,k=1,dim=-1)
            # print(yy.shape)
            # print(y_.shape)
            yy = yy.view(yy.shape[0],yy.shape[1])
            index_sep = (y_ == 1).nonzero()
            # print(    index_sep.shape)
            # print(y_)
            # index_sep_pred = (yy == 3).nonzero()
            # print(index_sep.shape)
            # print(y.shape)
            # print(index_sep.shape)
            # print(index_sep[0])
            
            # exit()
            
            for r, c in index_sep:
                # print(r,c)
                y__ = y_[r][:c]
                # print(y__)
                yy_ = yy[r][:c]
                # print(yy.shape)
                comp = torch.eq(yy_,y__)
                # print(comp[comp==True])
                count += c#y__.shape[0]
                # print(y__.shape,yy_.shape,comp.shape)
                comp = comp[comp==True].view(-1)
                # print(comp.shape)
                result += (comp.shape[0]/c)
        print(result)
        # result = result.cpu().numpy()
        print("avg",result/(validation_step*50))
        # exit()

if __name__ == "__main__":
    base = "bert"
    from tqdm import tqdm

    print(device)
    # bert_model.to(device)
    # bert_model.eval()
    # import pickle
    # with open('ht_model/w2i.pkl','rb') as f:
    #     w2i = pickle.load(f)
    # reader = fileread('hatescore_unsmile.tsv')
    # reader = reader.sample(frac=1)
    # data_size = reader.shape[0]
    config = {"maxlen":420,"max_grad_norm":1,"epoch":10,"batch_size":32}
    
    # mrc_data = HateSpeechDataset(reader, config["maxlen"])
    s_epoch = 0
    loss = nn.NLLLoss(ignore_index=3)
    
    postagger = PosTaggerModelPeftTorch(model,max_len,args.hidden_state,tag_len)
    # postagger.load_state_dict(torch.load("pos_model/model_24"))
    # result = postagger(torch.tensor([[1,2,3,4]]),torch.tensor([[1,2,3,4]]),torch.tensor([[1,1,1,1]]),torch.tensor([[0,0,0,0]]))
    # print(result.shape)
    # qconfig_dict = {'fc' : default_dynamic_qconfig}
    # model = EmbeddingWithLinear()
    # quantize_dynamic(model, qconfig_dict, inplace=True)

    postagger.to(device)
    # model = BERTHTFineTune(bert_model)
    # model = model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in postagger.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in postagger.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        # {'params': [p for n, p in postagger.named_parameters() if True], 'weight_decay': 0.01}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters,lr=8e-5)#5e-5)#8e-5)#5e-5)
    warmup_step = int( count_data * 50 * 0.1 )
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,num_training_steps=count_data * args.BATCH * EPOCH)
    n_data = dataset()
    n_validation = validation()
    # n_data = htdataset()
    
    from tqdm import tqdm
    # while False:
    for e in range(EPOCH):
        running_loss = 0
        postagger.train()
        
        # for index in tqdm(range(12910//batch_size)):
        # for index, batch in enumerate(tqdm(dataloader)):
        for index in range(count_data):
            bi,data,masks,segments,y = next(n_data)
            x = [data,masks,segments]
            # x, y = batch
            loss__ = train(config,optimizer,scheduler,loss,postagger,bi,x,y,device)
            running_loss += loss__
            if index % 100 == 0 and index != 0:
                print(e,index,loss__,(running_loss / ((index))))
            # break
        postagger.eval()
       
        eval(n_validation,device,postagger)
        if not os.path.exists(base):
            os.makedirs(base)
        # postagger.eval()
        # postagger = postagger.to(torch.device("cpu"))
        # backend = "fbgemm"
        # postagger.qconfig = torch.quantization.get_default_qconfig(backend)
        # # torch.backends.quantized.engine = backend
        # # postagger = postagger.to(torch.device("cpu")) 
        # model_static_quantized = torch.quantization.prepare(postagger)
        # model_static_quantized = torch.quantization.convert(model_static_quantized)
        # model_int8 = model_static_quantized
        # # run the model, relevant calculations will happen in int8
        # # res = model_int8(input_fp32)
        # print(count_parameters(model_int8),end=" ")
        # print(count_parameters(model_int8))
        torch.save(postagger.state_dict(), '{}/model_{}'.format(base,s_epoch+e))#{}'.format(e))
        # postagger = postagger.to(device)
    # eval(config,test_dataloader,device)
    eval(n_validation,device,postagger)
    torch.save(postagger.state_dict(), '{}/model'.format(base))
    import pickle
    #with open("posvocab.pkl","wb") as f:
    #    pickle.dump(posvocab,f)

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
parser.add_argument("--epoch_step",type=int,help="Train Data Epoch Step",default=200000)
parser.add_argument("--validation_step",type=int,help="Validation Data Epoch Step",default=1000)
parser.add_argument("--EPOCH",type=int,help="Train Epoch SIZE",default=5)
# parser.add_argument("--BATCH",type=int,help="Train BATCH SIZE",default=50)
parser.add_argument("--model_name",type=str,help="Tokenizer Model Name",default="lstm_bigram_tokenizer.model")
parser.add_argument("--GPU_NUM",type=str,help="Train GPU NUM",default="0")
parser.add_argument("--infer",help="Train GPU NUM",action="store_true")
parser.add_argument("--model_path",help="Train GPU NUM",type=str)
parser.add_argument("--sepoch",help="Train GPU NUM",type=int,default=0)
parser.add_argument("--lr",help="Train GPU NUM",type=float,default=1e-3)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_NUM
max_len = args.MAX_LEN#300
EPOCH = args.EPOCH#6
BATCH_SIZE = args.BATCH#50
count_data = args.epoch_step#4000
validation_data = args.validation_step#400

count = 0
with open("wauto_x.txt",'r',encoding='utf-8') as x_f:
    for l in x_f:
        count += 1
print(count)
# count = 550000
with open('lstm_vocab.pkl','rb') as f:
    lstm_vocab = pickle.load(f)
# x_f.seek(0)
# from tqdm
count = count // BATCH_SIZE
count_data = count
count_data = int(count * 0.9) 
validation_data = int(count * 0.1)
with open('bigram_vocab.pkl','rb') as f:
    bigram = pickle.load(f)
# print(bigram)
print(count)
print(count_data)
print(validation_data)
# exit()
def make_batch(length,lstm_X,lstm_Y,lstm_Y2,BI):
    mx_length = max(length)
    mx_length = max(10,mx_length)
    tmpX = []
    tmpY = []
    tmpY2 = []
    tmpBI = []
    for lx,ly,ly2 in zip(lstm_X,lstm_Y,lstm_Y2):
        lstm_x = lx[:mx_length] + [0] * (mx_length - len(lx))
        lstm_y = ly[:mx_length] + [2] * (mx_length - len(ly))
        lstm_y2 = ly2[:mx_length] + [2] * (mx_length - len(ly2))
        tmpX.append(lstm_x)
        tmpY.append(lstm_y)
        tmpY2.append(lstm_y2)
        
    for bi in BI:
        bi_npy = bi[:mx_length] + [0] * (mx_length - len(bi))
        tmpBI.append(bi_npy)
    
    lstm_X = np.array(tmpX)
    lstm_Y = np.array(tmpY)
    lstm_Y2 = np.array(tmpY2)
    BI = np.array(tmpBI)

    
    lstm_X = torch.tensor(lstm_X)
    BI = torch.tensor(BI)
    lstm_Y = torch.tensor(lstm_Y)
    lstm_Y2 = torch.tensor(lstm_Y2)
    return [lstm_X,BI], [lstm_Y,lstm_Y2]


# if os.path.exists("../aho.pkl"):
#     with open("../aho.pkl","rb") as f:
#         eomiaho = pickle.load(f)
def make_data(x,y,length,lstm_X,lstm_Y,lstm_Y2,BI):
    bi_npy = []
    # for xx in x.split():
    #     eomiaho.s
    for i in range(len(x)-1):
        bi = x[i:i+2]
        # if bi in bigram:
        bi_npy.append(bigram[bi])
        # else:
        #     bi_npy.append(bigram["[UNK]"])
    if False:
        bi_npy = bi_npy[:max_len] + [0] * (max_len - len(bi_npy))
    lstm_x = [lstm_vocab[i] if i in lstm_vocab else 1 for i in x]
    if False:
        lstm_x = lstm_x[:max_len] + [0] * (max_len - len(lstm_x))
    # print(y)
    # exit()
    lstm_y = [1 if i == "2" else 0 for i in y]
    # print(y)
    # print(x)
    # exit()
    lstm_y2 = [1 if i == "1" else 0 for i in y]
    # print(lstm_y)
    # print(lstm_y2)
    # print(lstm_x)
    # exit()
    if False:
        lstm_y = y[:max_len] + [3] * (max_len -len(y))
    length.append(min(len(lstm_x),max_len))
    lstm_X.append(lstm_x)
    lstm_Y.append(lstm_y)
    lstm_Y2.append(lstm_y2)
    BI.append(bi_npy)

def dataset2(flag=False):
    with open("wauto_x.txt",'r',encoding='utf-8') as x_f:
        with open("wauto_y.txt",'r',encoding='utf-8') as y_ff:   
            X = []
            Y = []
            lstm_X = []
            lstm_Y = []
            lstm_Y2 = []
            BI = []
            file_num = 0
            x_f.seek(0)
            y_ff.seek(0)
            if flag:
                start = ((count_data +1) * BATCH_SIZE)
                end = (count_data + validation_data+ 1) * BATCH_SIZE
                for ii in range(start):
                    x_f.readline()
                    y_ff.readline()
            else:
                start = 0 
                end = ((count_data+1) * BATCH_SIZE) 
            length = []
            for _ in range(start,end):
                if False:
                    print(_)
                file_num = 0
                x = x_f.readline()
                y = y_ff.readline()
                x = x.replace('\n','')
                y = y.replace('\n','')
                if True: #len(x) <= max_len and len(y) <= max_len:
                    make_data(x,y,length,lstm_X,lstm_Y,lstm_Y2,BI)
                    
                if len(lstm_X) == BATCH_SIZE:
                    yield make_batch(length,lstm_X,lstm_Y,lstm_Y2,BI)
                    BI = []
                    lstm_X = []
                    lstm_Y = []
                    lstm_Y2 = []
                    file_num += 1
                    length = []
                    # Auto = []
                    
def dataset2_test(flag=True):
    with open("wauto_x.txt",'r',encoding='utf-8') as x_f:
        with open("wauto_y.txt",'r',encoding='utf-8') as y_ff:   
            X = []
            Y = []
            lstm_X = []
            lstm_Y = []
            lstm_Y2 = []
            BI = []
            file_num = 0
            x_f.seek(0)
            y_ff.seek(0)
            if flag:
                start = ((count_data +1+10) * BATCH_SIZE)
                end = (count_data + validation_data +10+ 1) * BATCH_SIZE
                for ii in range(start):
                    x_f.readline()
                    y_ff.readline()
            else:
                start = 0 
                end = ((count_data+1) * BATCH_SIZE) 
            length = []
            for _ in range(start,end):
                if False:
                    print(_)
                file_num = 0
                x = x_f.readline()
                y = y_ff.readline()
                x = x.replace('\n','')
                y = y.replace('\n','')
                if True: #len(x) <= max_len and len(y) <= max_len:
                    make_data(x,y,length,lstm_X,lstm_Y,lstm_Y2,BI)
                    
                if len(lstm_X) == BATCH_SIZE:
                    yield make_batch(length,lstm_X,lstm_Y,lstm_Y2,BI)
                    BI = []
                    lstm_X = []
                    lstm_Y = []
                    lstm_Y2 = []
                    file_num += 1
                    length = []
                    # Auto = []


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
    for _ in range(EPOCH):
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


train_data = dataset()
val_data = validation()

def train(config,optimizer,loss,model,x,bi,y,device):
    model.train()
    
    # bi = bi.to(device)
    x = x.to(device)
    unfold = x.unfold(1,3,1).to(device)
    # print(x)
    # exit()
    yp_tmp = model(x,unfold)
    
    y_pred = yp_tmp[0]#.to(device)
    y_pred = y_pred.view(-1)#, model.tag_len)
    
    y_pred2 = yp_tmp[1]
    y_pred2 = y_pred2.view(-1)

    y_tmp = y
    y = y_tmp[0].to(device)
    y = y.view(-1)
    y = y.type(torch.FloatTensor).to(device)
    # if loss:
    y_removepad = y.clone()
    y_removepad[y_removepad==2] = 0
    loss_1 = loss[0](y_pred,y_removepad)

    y2 = y_tmp[1].to(device)
    y2 = y2.view(-1)
    y2 = y2.type(torch.FloatTensor).to(device)

    y_removepad2 = y2.clone()
    y_removepad2[y_removepad2==2] = 0

    loss_2 = loss[1](y_pred2,y_removepad2)

    loss_ = (loss_1 + loss_2) / 2
    optimizer.zero_grad()
    loss_.backward()
    
    optimizer.step()

    return loss_.item()

def train_crf(config,optimizer,loss,model,x,bi,y,device):
    model.train()
    
    # bi = bi.to(device)
    x = x.to(device)

    # print(x)
    # exit()
    
    # y_pred = yp_tmp[0]#.to(device)
    # y_pred = y_pred.view(-1, model.tag_len)
    
    y_tmp = y
    y = y_tmp[0].to(device)
    # y = y.view(-1)
    y = y.type(torch.LongTensor).to(device)
    # y = torch.nn.functional.one_hot(y,num_classes=4).to(device)
    # print(y.shape)
    # exit()
    # if loss:
    # loss_ = loss(y_pred,y)
    loss_ = model(x,y=y)[0]#.cpu().numpy().tolist())
    # print(loss_[0])
    # exit()
    optimizer.zero_grad()
    loss_.backward()
    
    optimizer.step()

    return loss_.item()

# def train_crf(config,optimizer,loss,model,x,bi,y,device):
#     model.train()
    
#     # bi = bi.to(device)
#     x = x.to(device)

#     # print(x)
#     # exit()
    
#     # y_pred = yp_tmp[0]#.to(device)
#     # y_pred = y_pred.view(-1, model.tag_len)
    
#     y_tmp = y
#     y = y_tmp[0].to(device)
#     # y = y.view(-1)
#     y = y.type(torch.LongTensor).to(device)
#     # if loss:
#     # loss_ = loss(y_pred,y)

#     optimizer.zero_grad()
#     # loss_.backward()
    
#     optimizer.step()

#     return loss_.item()

def eval_crf(val_dataloader,device,model):
    with torch.no_grad():
        count = 0
        result = 0
        result2 = 0
        for index in range(validation_data):
            xbi,y_ = next(val_dataloader)
            
            bi_ = xbi[1].to(device)
            x_ = xbi[0].to(device)
            
            y_tmp = y_
            
            y_ = y_tmp[0]
            y_ = y_.to(device)
            y_ = y_.type(torch.LongTensor).to(device)
            # yy = model(x_,bi_)
            yy = model(x_)[0]

            yp_tmp = yy
            yy = torch.tensor(yp_tmp[0])
            
            yy = yy.to(device)
            yy = torch.argmax(yy,dim=-1)
            for r,_ in enumerate(y_):
                y__ = y_[r]
                y__ = y__[y__!=3]
                yy_ = yy[r][:y__.shape[0]]
                htprob = ((y__ == 2) | (y__ == 0)).nonzero()
                predprob = yy_[htprob[:,0]]
                trueprob = y__[htprob[:,0]]
                comp = (predprob == trueprob)
                comp = comp[comp==True].view(-1)
                result += (comp.shape[0]/(trueprob.shape[0]+1e-9))
            

            print(f"\r{index+1}/{validation_data}",end="")
        print(result)
        print("tk avg",result/(validation_data*BATCH_SIZE))
        

def eval(val_dataloader,device,model):
    with torch.inference_mode():
        count = 0
        result = 0
        result2 = 0
        c = 0
        for index in range(validation_data):
            xbi,y_ = next(val_dataloader)
            
            bi_ = xbi[1].to(device)
            x_ = xbi[0].to(device)
            
            y_tmp = y_
            
            y_ = y_tmp[0]
            y_ = y_.to(device)
            # y_ = y_.type(torch.LongTensor).to(device)
            # yy = model(x_,bi_)
            unfold = x_.unfold(1,3,1).to(device)
            yy = model(x_,unfold)

            yp_tmp = yy
            yy = yp_tmp[0]
            
            yy = yy.to(device)
            result += calc_avg(yy,y_)
            c += yy.shape[0]

            y_2 = y_tmp[1]
            y_2 = y_2.to(device)
            yy2 = yp_tmp[1]
            result2 += calc_avg(yy2,y_2)

            # result2 = (result1 + result2) / 2
            # yy = torch.argmax(yy,dim=-1)
            """yy[yy>=0.5] = 1
            yy[yy<0.5] = 0
            
            yy = yy.reshape(yy.shape[0],yy.shape[1])
            # c = 0
            for r,_ in enumerate(y_):
                y__ = y_[r]
                y__ = y__[y__!=3]
                yy_ = yy[r][:y__.shape[0]]
                yy_ = yy[r]
                htprob = (y__ == 2).nonzero()
                predprob = yy_[htprob[:,0]]
                trueprob = y__[htprob[:,0]]
                comp = (predprob == trueprob)
                comp = comp[comp==True].view(-1)
                # c+= 1
                # if trueprob.shape[0] > 0 and comp.shape[0] == 0:
                    # c+=1
                    # conti
                if trueprob.shape[0] == 0:
                    continue
                c+=1
                result += (comp.shape[0]/(trueprob.shape[0]+1e-9))"""
            

            print(f"\r{index+1}/{validation_data}",end="")
        print(result)
        print("tk avg",result/c)#(validation_data*BATCH_SIZE))
        print("auto avg",result2/c)

def calc_avg(yy,y):
    # yy[yy>=0.5] = 1
    # yy[yy<0.5] = 0
    
    # y = y * 4
    # y = y.type(torch.LongTensor).to("cuda")
    # y[(0. <= y) & (y <=0.25)] = 1
    # y[(0.25 < y) & (y <= 0.5)] = 2
    # y[(0.5 < y) & (y <= 0.75)] = 3
    # y[0.75 <= y] = 4

    yy[yy < 0.5] = 0
    yy[yy >= 0.5] = 1
    # yy[(0. <= yy) & (yy <= 1.)] = 1
    # yy[(1. < yy) & (yy <= 2.)] = 2
    # yy[(2. < yy) & (yy <= 3.)] = 3
    # yy[3. < yy] = 4
    
    # print(y)
    # print(yy)
    # exit()
    yy = yy.reshape(yy.shape[0],yy.shape[1])
    y = y.type(torch.LongTensor).to(y.device)
    resultavg = 0.0
    for yy_,y_ in zip(yy,y):
        padidx = (y_ == 2)
        y_ = y_[:padidx.shape[0]]
        yy_ = yy_[:padidx.shape[0]]
        # print(y_.shape)
        # print(yy.shape)
        # exit()
        httag = (y_ == 1)

        cmplabel = y_[httag] == yy_[httag]
        cmplabel = cmplabel.type(torch.LongTensor).to(y_.device)

        resultavg += torch.sum(cmplabel)/(y_[httag].shape[0] + 1e-9)
    
    return resultavg.item()

def eval_test(val_dataloader,device,model):
    with torch.inference_mode():
        count = 0
        result = 0
        result2 = 0
        c= 0
        for index in range(10):
            xbi,y_ = next(val_dataloader)
            
            bi_ = xbi[1].to(device)
            x_ = xbi[0].to(device)
            
            y_tmp = y_
            
            y_ = y_tmp[0]
            y_ = y_.to(device)
            # y_ = y_.type(torch.LongTensor).to(device)
            # yy = model(x_,bi_)
            # try:
            unfold = x_.unfold(1,3,1).to(device)
            # except:
                # print(x_.shape)
                # exit()
            yy = model(x_,unfold)

            yp_tmp = yy
            yy = yp_tmp[0]
            
            yy = yy.to(device)
            result += calc_avg(yy,y_)
            c += yy.shape[0]

            
            y_2 = y_tmp[1]
            y_2 = y_2.to(device)
            yy2 = yp_tmp[1]
            
            yy2 = yy2.to(device)
            result2 += calc_avg(yy2,y_2)

            # result2 += result2
            """for r,_ in enumerate(y_):
                y__ = y_[r]
                y__ = y__[y__!=3]
                yy_ = yy[r][:y__.shape[0]]
                htprob = (y__ == 2).nonzero()
                predprob = yy_[htprob[:,0]]
                trueprob = y__[htprob[:,0]]
                comp = (predprob == trueprob)
                comp = comp[comp==True].view(-1)
                if trueprob.shape[0] == 0:
                    continue
                c += 1
                result += (comp.shape[0]/(trueprob.shape[0]+1e-9))"""
            

            print(f"\r{index+1}/{10}",end="")
        print(result)
        print("tk avg",result/(c+1e-20))
        print("auto avg",result2/(c+1e-20))
        # exit()
        
def train2(config,optimizer,loss,model,x,bi,y,device):
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
    y_removepad = y.clone()
    y_removepad[y_removepad==3] = 0
    loss_ = loss(y_pred,y_removepad)

    optimizer.zero_grad()
    loss_.backward()
    
    # torch.nn.utils.clip_grad_norm_(model.parameters(),config["max_grad_norm"])

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
    if not args.infer:
        from torch.utils.tensorboard import SummaryWriter
        from tqdm import tqdm
        from model import TK_Model_Mini_Auto as TK_Model2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
        print(device)
        writer = SummaryWriter()
        config = {"maxlen":420,"max_grad_norm":1,"epoch":10,"batch_size":32}
        
        loss = nn.MSELoss()#nn.NLLLoss(ignore_index=3)
        loss2 = nn.MSELoss()#nn.NLLLoss(ignore_index=3)
        # loss2 = nn.NLLLoss(ignore_index=2)
        
        model = TK_Model2(max_len,lstm_vocab)
        
        model.to(device)
        # model.load_state_dict(torch.load(f'{args.model_path}/model'))
        optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr)#2e-4)#5e-5)#8e-5)#5e-5)
        # train = train_crf
        # eval = eval_crf
        # n_data = dataset()
        # n_validation = validation()
        from tqdm import tqdm
        from datetime import datetime
        for e in range(EPOCH):
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
                xbi,y= next(n_data)
                
                loss__ = train(config,optimizer,[loss,loss2],model,xbi[0],xbi[1],y,device)
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
                # writer.add_scalars("Train/Loss",)
                print(f"\repoch:{e}",",",f"{index+1}/{count_data}",f"batch_loss:{loss__:.6f}",",",f"total_loss:{(running_loss / ((index+1))):.6f}",",",f"{cur_working}/{total_working}",end="")
                # if index+1 == count_data:
                    # break
                if (index+1) % 500==0:
                    test = dataset2_test()
                    model.eval()
                    eval_test(test,device,model)
                    model.train()
            print()
            model.eval()
            n_validation=dataset2(flag=True)
            print()
            eval(n_validation,device,model)
            if not os.path.exists(args.model_path):
                os.makedirs(args.model_path)
            torch.save(model.state_dict(), '{}/model_{}'.format(args.model_path,e))#{}'.format(e))
        torch.save(model.state_dict(), '{}/model'.format(args.model_path))
    else:
        import sys
        import os
        from tqdm import tqdm
        from model import TK_Model_Mini_Auto as TK_Model2
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
        print(device)
        sys.path.append(os.path.join("C:\\Users\\ty341\\Desktop\\headtail_train","headtail_train"))
        os.environ["HT"] = os.path.join("C:\\Users\\ty341\\Desktop\\headtail_train","headtail_train","postagger_model")
        score = 0
        testlen = 0
        model = TK_Model2(max_len,lstm_vocab)
        import tokenizer_ht as tokenizer
        model.to(device)
        # 3
        if args.sepoch > 0:
            # print(args.sepoch)
            model.load_state_dict(torch.load('{}/model_{}'.format(args.model_path,args.sepoch-1),weights_only=True))
        else:
            model.load_state_dict(torch.load('{}/model'.format(args.model_path),weights_only=True))
        # while True:
        while True:
            text = input("sent: ")
            print(tokenizer.predict4(model,[text.strip()],device))

        with open("httk_x.txt",encoding="utf-8") as fx: 
            with open("httk_y.txt",encoding="utf-8") as fy:
                # text = input("sentence: ")
                # print(tokenizer.predict2(model,[text],device))
                # if text == "exit":
                #     exit()
                X = []
                tX = []
                # count = 0
                counttmp = 0
                for x,y in zip(fx,fy):
                    counttmp += 1
                    if counttmp <= count * args.BATCH:
                        continue
                    
                    X.append(x.strip())#.replace("▁"," "))
                    tx = list(x.strip())
                    ty = list(y.strip())
                    for i in range(len(tx)):
                        if y[i] == "2":
                            tx[i] = "+" + tx[i]#+"+"
                    
                    tks = "".join(tx)#.replace("▁"," ")                        
                    tX.append(tks)    
                    
                    if len(X) % 100 == 0:
                        PRED = tokenizer.predict2(model,X,device)
                        # print(count)
                        # print(PRED[-1])
                        # exit()
                        for p,t in zip(PRED,tX):
                            # print(p)
                            # exit()
                            # print(t)
                            correct = 0
                            p = p.split()
                            t = t.split("▁")
                            # if p.count("+") >= 2:
                                # print(p)
                            for pp, tt in zip(p,t):
                                if pp == tt:
                                    correct += 1
                            testlen += 1
                            # print(p,t)
                            # exit()
                            score += (correct/len(t))
                        # if counttmp >= 100000:
                        #     break
                        X = []
                        tX = []                            
        print(score / testlen)
from collections import defaultdict
import torch
import os
class MakeLMDataset:
    def __init__(self):
        self.negdataset = defaultdict(int)
        self.tagsep = "@@@"
        self.htsep = "hththththt"
    
    def list2join(self,arr):
        return self.removehtsep(" ".join(arr))

    def removehtsep(self,txt):
        txt = txt.replace(self.htsep,"")
        return txt

    def make_dataset(self,txt):
        txt = txt.split()
        windows = []
        for i in range(1,len(txt)-1):
            txtcur = txt[i].replace(self.htsep,"+")
            ngram = self.list2join(txt[i-2:i]) + " " + "@" + txtcur + "@" + " " + self.list2join(txt[i+1:i+3])
            # ngram = ngram.
            # print(ngram)
            windows.append(ngram)
        return windows

    def make_negdataset(self,txt):
        self.negdataset
        txt = txt.split()
        for t in txt:
            t = self.removehtsep(t)
            self.negdataset[t] += 1
        
# txt = "[CLS] 백운규 산업통상자원부 장관hththththt이 20 일 오후 서울 세종대로 정부서울청사hththththt에서 안 린hththththt데 스웨덴 외교부 통상담당 장관hththththt과 악수하hththththt고 있hththththt다 [SEP]"
# make_dataset(txt)

# class MakeDataset(torch.utils.data.Dataset):
#     def __init__(self,pospath,negpath):

from headtail.postagger_model import SelfEncoder
class LmModel(torch.nn.Module):
    def __init__(self,posvocab):
        super(LmModel,self).__init__()
        self.model = SelfEncoder(posvocab)
        self.config = self.model.config
        
        self.model = self.model.model
        self.layer = torch.nn.Linear(self.config.dmodel,self.config.dmodel // 2)
        self.output = torch.nn.Linear(self.config.dmodel // 2,1)
        self.softmax = torch.nn.LogSoftmax(dim=-1)
    def forward(self,x):
        enc = self.model(x)
        layer = self.layer(enc)
        
        # output = self.softmax(torch.mean(layer,dim=1))
        output = self.output(torch.mean(layer,dim=1))
        
        return output
    
if __name__ == "__main__":
    md = MakeLMDataset()
    datapath = os.path.join(os.environ["DATASET"],"ht_dataset","modu","result","kiwimorphs_.txt")
    poslmdatapath = os.path.join(os.environ["DATASET"],"ht_dataset","modu","result","poslm.txt")
    neglmdatapath = os.path.join(os.environ["DATASET"],"ht_dataset","modu","result","neglm.txt")
    inpath = open(datapath,encoding="utf-8")
    count = len(inpath.readlines())
    inpath.close()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("../kcctokenizer")
    ids = tokenizer(["밥을 [SEP] 먹고","밥+을 [SEP] 먹고","밥+을 [SEP] 우주로","자전거+로는 [SEP] 달렸다.","자전거로+는 [SEP] 달렸다."],padding="longest",return_tensors="pt")["input_ids"].to("cuda")
    import pickle
    with open("../postagger_model/subwordvocab.pkl","rb") as f:
        posvocab = pickle.load(f)
    model = LmModel(posvocab).to("cuda")
    model.load_state_dict(torch.load("model_4",))
    print(model(ids))
    exit()
    # train = count * 0.9
    # val = count * 0.9
    
    epoch = 10
    makeflag = False
    import os
    import random
    if makeflag:
        def duplespace(txt):
            import re
            txt = re.sub(r" +"," ",txt)
            return txt
        
        with open(datapath,encoding="utf-8") as f, open(poslmdatapath,"w",encoding="utf-8") as posdataf,open(poslmdatapath.replace(".txt","_tmp.txt"),"w",encoding="utf-8") as posdataftmp:
            linecnt = 0
            for l in f:
                l = l.strip()
                posdata = md.make_dataset(l)
                data = []
                for pos_ in posdata:
                    data = []
                
                    for pos__ in pos_.split():
                        if pos__.startswith("@") and pos__.endswith("@"):
                            data.append(pos__)
                        else:
                            data.append(pos__)
                    
                    data = " ".join(data)
                    data = duplespace(data).strip()
                    data_ = data.replace("[SEP]"," ")
                    data_ = duplespace(data_).strip()
                    md.make_negdataset(data_)
                    posdataf.write(data+"\n")
                    posdataf.flush()
                
                for pd in posdata:
                    posdataftmp.write(pd+"\n")
                posdataftmp.flush()
                linecnt += 1
                # if linecnt == count:
                #     break
            # f.seek(0)
            # for l in f:
        words = list(md.negdataset.keys())
        with open(poslmdatapath.replace(".txt","_tmp.txt"),encoding="utf-8") as f, open(neglmdatapath,"w",encoding="utf-8") as negdataf:
            for l in f:
                ltmp = l
                l = l.split()
                idx = 0
                negsdata = []
                negcount = 0
                for t in l:
                    if t.startswith("@") and t.endswith("@"):#"+" in t:
                        negsdata.append(t)
                    else:
                        
                        wordid = random.randint(0,len(words)-1)
                        if words[wordid] not in ltmp.replace("+","").replace("@",""):
                            negsdata.append(words[wordid])
                        else:
                            while words[wordid] not in ltmp.replace("+","").replace("@",""):
                                wordid = random.randint(0,len(words)-1)
                            negsdata.append(words[wordid])
                    # idx += 1
                negsdata = duplespace(" ".join(negsdata))
                negdataf.write(negsdata+"\n")
                negdataf.flush()
    datas = []         
    with open(poslmdatapath,encoding="utf-8") as p, open(neglmdatapath,encoding="utf-8") as n:
        datacnt = 0
        for pp in p:
            datacnt += 1
            data = []
            inw = ""
            for ppp in pp.strip().split():
                if ppp.startswith("@") and ppp.endswith("@"):
                    ppp = ppp.strip("@")
                    inw = ppp
                # data.append(ppp)
        
            # datas.append((" ".join(data),1))
        # p.seek(0)
            for ppp in pp.strip().split():
                if not ppp.startswith("@") and not ppp.endswith("@"):
                    datas.append([inw+" [SEP] "+ppp,1])
        for nn in n:
            datacnt += 1
            data = []
            inw = ""
            for nnn in nn.strip().split():
                if nnn.startswith("@") and nnn.endswith("@"):
                    nnn = nnn.strip("@")
                    inw = nnn
            for nnn in nn.strip().split():
                if not nnn.startswith("@") and not nnn.endswith("@"):
                    datas.append([inw+" [SEP] "+nnn,0])
                # data.append(nnn)
            # datas.append((" ".join(data),0))
    
    print(datacnt)
    print(datas[0])
    import random
    random.shuffle(datas)
    print(datacnt)
    print(datas[0])
    # exit()
    X = []
    Y = []
    xtmp = []
    ytmp = []

    # ids = tokenizer(["나는 [SEP] 밥+을 [SEP] 먹고 학교에","나는 [SEP] 밥을 [SEP] 먹고 학교에"],padding="longest",return_tensors="pt")["input_ids"].to("cuda")
    # import pickle
    # with open("../postagger_model/subwordvocab.pkl","rb") as f:
    #     posvocab = pickle.load(f)
    # model = LmModel(posvocab).to("cuda")
    # model.load_state_dict(torch.load("model_4"))
    # print(model(ids))
    # exit()
    for idx,d in enumerate(datas):
        xtmp.append(d[0])
        ytmp.append(d[1])
        
        if len(xtmp) == 1024 or idx == len(datas) - 1:
            xtmp_ = tokenizer(xtmp,padding="longest")
            xtmp_ = xtmp_["input_ids"]#.to("cuda")
            ytmp_ = ytmp#.to("cuda")
            
            X.append(xtmp_)
            Y.append(ytmp_)
            
            xtmp = []
            ytmp = []
        
    train = int(len(X) * 0.8)
    test = [train, train + int(len(X) * 0.1)]
    # test = [train,train]
    val = [test[1],test[1]+ int(len(X) * 0.1)]
    # from head_tail.pos
    import pickle
    with open("../postagger_model/subwordvocab.pkl","rb") as f:
        posvocab = pickle.load(f)
    model = LmModel(posvocab).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
    from tqdm import tqdm
    for e in range(5):
        for i in range(len(X[:train])):
            model.train()
            x = torch.tensor(X[i],dtype=torch.long).to("cuda")
            y = torch.tensor(Y[i],dtype=torch.float).to("cuda")
            
            predy = model(x)
            
            loss = torch.nn.functional.mse_loss(predy.view(-1),y.view(-1))
            # print
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"\rTRAIN epoch: {e+1:02d}/10, batchidx: {i+1:08d}/{train:08d}, loss: {loss.item()}",end="")
            if (i+1) % 10000 == 0:
                model.eval()
                resultavg = 0 
                resultloss = 0
                for j in range(test[0],test[1]):
                    testx = X[j]
                    testy = Y[j]
                    # d_ = []
                    # for x in testx:
                    #     d = tokenizer.decode(x)
                    #     d_.append(d)
                    # d = d_
                    testx = torch.tensor(testx,dtype=torch.long).to("cuda")
                    testy = torch.tensor(testy,dtype=torch.float).to("cuda")
                    
                    predtest = model(testx)
                    # for dd, p in zip(d,predtest.view(-1)):
                    #     print(dd,p)
                    # exit()
                    loss = torch.nn.functional.mse_loss(predtest.view(-1),testy.view(-1))
                    
                    # predtest = torch.argmax(predtest,dim=-1)
                    # result = predtest.view(-1) == testy.view(-1)
                    
                    # predtest[result].shape
                    # resultavg += predtest.view(-1)[result].shape[0] / testx.shape[0]
                    resultloss += loss.item()
                print()
                print("TEST epoch:",e,"batchidx:",i+1,"avg:",resultavg/(test[1] - test[0]),"loss:",resultloss/(test[1] - test[0]))
        resultavg = 0
        resultloss = 0
        print()
        model.eval()
        for j in range(val[0],val[1]):
            valx = torch.tensor(X[j],dtype=torch.long).to("cuda")
            valy = torch.tensor(Y[j],dtype=torch.float).to("cuda")
        
            predval = model(valx)
            
            loss = torch.nn.functional.mse_loss(predval.view(-1),valy.view(-1))

            # predval = torch.argmax(predval,dim=-1)
            # result = predtest.view(-1) == testy.view(-1)
            
            # predtest[result].shape
            # resultavg += predval.view(-1)[result].shape[0] / valx.shape[0]
            resultloss += loss.item()
        print()
        print("VALIDATION epoch:",e,"avg:",resultavg/(val[1] - val[0]),"loss:",resultloss/(val[1] - val[0]))
        # print("VALIDATION epoch:",e,"mse:",resultavg/(val[1]-val[0]))
        torch.save(model.state_dict(),f"model_{e}")
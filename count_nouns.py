from collections import defaultdict
from tqdm import tqdm

import re
import os
import pickle

mincount = 1
minprob = 0.77
nng_pos = []
count = defaultdict(int)


def pos_extract(pos,arr=["NOUN"]):#,"VV","VA"]):
    # return True
    for i in arr:
        if i == pos:
            return True
    return False                
datapath = os.environ["DATASET"]
words = []
sentences = []
index = 0
from ht_utils import get_xsn
reshtpf, resrawf = open("resht.txt","w",encoding="utf-8"), open("resraw.txt","w",encoding="utf-8")
rawfilename = []
wordssep = []
for iddx,path in enumerate(tqdm([os.path.join(datapath,"kccword2vec_data_ht3.txt")])):#, \
    # os.path.join(datapath,"naverblog_sentsplit250205_ht3.txt"), \
    # os.path.join(datapath,"naverblog_sentsplit250213_ht3.txt"), \
    # os.path.join(datapath,"naverblog_sentsplit250307_ht3.txt")])):
    # os.path.join(datapath,"jsondataset_sentsplit_ht.txt")])):
    
    if "ht3" in path:
        rawpath = path.replace("_ht3","")
    # else:
        # rawpath = path.replace("_ht","")
    with open(path,encoding="utf-8") as htp, open(rawpath,encoding="utf-8") as rp:
        # htp_oneline, raws_oneline = [], []
        linecnt = 0
        for ht,rawdata in zip(htp,rp):
            ht,rawdata = ht.strip(),rawdata.strip()
            # if rawdata == "<<EOD>>":
            # htp_oneline = " ".join([ht])
            # raws_oneline = " ".join([rawdata])
            wordssep.append(ht)
            reshtpf.write(ht+"\n")
            resrawf.write(rawdata+"\n")
            # htp_oneline = []
            # raws_oneline = []
            rawfilename.append("kcc")
            #     continue
            # htp_oneline.append(ht)
            # raws_oneline.append(rawdata)
            linecnt += 1
            if linecnt == 8000:
                break

# print(len(rawfilename))            
with open(os.path.join(os.path.join(datapath,"jsondatasetfn.txt")),encoding="utf-8") as filenamef:
    for l in filenamef:
        # if l.strip() != "<<EOD>>":
        rawfilename.append(l.strip())
# print(len(rawfilename))

for iddx,path in enumerate(tqdm([os.path.join(datapath,"jsondataset_ht.txt")])):
    
    # if "ht3" in path:
    rawpath = path.replace("_ht","")
    # else:
        # rawpath = path.replace("_ht","")
    # print(path,rawpath)
    with open(path,encoding="utf-8") as htp, open(rawpath,encoding="utf-8") as rp:
        htp_oneline, raws_oneline = [], []
        htp_sep = []
        for ht,rawdata in zip(htp,rp):
            ht,rawdata = ht.strip(),rawdata.strip()
            if rawdata == "<<EOD>>":
                htp_sep = " [SEP] ".join(htp_oneline)
                htp_oneline = " ".join(htp_oneline)
                raws_oneline = " [SEP] ".join(raws_oneline)
                
                wordssep.append(htp_sep)
                reshtpf.write(htp_oneline+"\n")
                resrawf.write(raws_oneline+"\n")
                htp_oneline = []
                raws_oneline = []
                
                continue
            htp_oneline.append(re.sub(r"\s+"," ",ht).strip().replace(" + ","+").replace(" +","+").replace("+ ","+").strip())
            raws_oneline.append(rawdata)
            
reshtpf.close()
resrawf.close()
# exit()
linecnt = 0
for iddx,path in enumerate(tqdm(["resht.txt"])):
    # os.path.join(datapath,"naverblog_sentsplit250205_ht3.txttmp"), \
    # os.path.join(datapath,"naverblog_sentsplit250213_ht3.txttmp"), \
    # os.path.join(datapath,"naverblog_sentsplit250307_ht3.txttmp")])):
    # if iddx == 0:
    #     rawpath = os.path.join(datapath,"kccword2vec_data.txt")
    # else:
    rawpath = "resraw.txt"
        # rawpath = path.replace("_ht3.txttmp",".txt").replace("sentsplit","autospace")
    # print(path,rawpath)
    # exit()
    linecnt = 0
    with open(path,encoding="utf-8") as f, open(rawpath,encoding="utf-8") as rf:
        # print(len(f.readlines()),len(rf.readlines()))
        # exit()
        for l,raw in zip(f,rf):
            # if l.strip() == "EOD/SL":
            #     continue
            wordstmp = []
            l = re.sub(r"\s+"," ",l).strip().replace(" + ","+").replace(" +","+").replace("+ ","+").strip()
            for ll in l.strip().split():
                # ll 
                # print(ll)
                ll = ll.strip("+")
                lll = ll.split("+")
                # print(ll)
                # lll[0] = get_xsn(lll[0]).split("+")[0]
                try:
                    for llll in lll:
                        tks = llll.split("/")
                        if tks[1][0] != "S" and tks[0].strip() != "":
                            if pos_extract(tks[1]):
                                wordstmp.append(tks[0])
                                count[llll] += 1
                except:
                    print(ll)
                    print(lll)
                    print(llll)
                    print(l)
                    exit()
            # print(l,wordstmp)
            linecnt += 1
            
            # if len(wordstmp) > 0:
            words.append(wordstmp)
            sentences.append(raw)
            wordstmp = []
            # else:
            #     sentences.append(index)
            # inex += 1
    # print(linecnt)
dict_ = defaultdict(list)   
# print(linecnt)
# print(len(words),len(rawfilename))
# exit()
# exit()
for k,v in count.items():
    if len(k) >= mincount:
        dict_["token"].append(k)
        dict_["count"].append(v)
        
import pandas as pd

df = pd.DataFrame(dict_)
df.to_csv("C:\\Users\\ty341\\OneDrive\\Desktop\\dataset\\qadataset_tropical.csv",encoding="utf-8-sig")
# df = pd.read_csv("C:\\Users\\ty341\\OneDrive\\Desktop\\dataset\\tropical.csv")

# print(df["count"].max(),df["count"].min(),df["count"].mean(),df.query("count >= 44").shape,df.shape)
df_ = df["count"].value_counts()#.to_frame()#.reset_index()

df_ = pd.DataFrame({"count":df_.index,"nums":df_.values}).reset_index(drop=True)
# print(df_)
df_ = df_.sort_values(by=["nums"],axis=0,ascending=False)
from headtail import analysis

if False:
    df = df.query("count >= 5")
    with open("words.txt",encoding="utf-8") as f:
        tmp = {}
        for l in f:
            l = l.strip()
            if l in tmp:
                continue
            res = df.query(f"token == \"{l}\"")
            if res.shape[0] > 0:
                token = l
                count = res.loc[:,["count"]].values
                # for _ in range(cout[0][0]):
                print(l)
            tmp[l] = True
        
else:
    from gensim.models import Word2Vec
    model_filename = 'word2vec_qadataset.model'
    # model_filename = 'word2vec.model'
    
    if True:
        model = Word2Vec(words,workers=8,vector_size=100, window=5, hs=1, min_count=mincount, sg=1)        
        model.save(model_filename)

        # print(model_filename + ' 저장완료')
    else:
        three_under = open("thunder.txt","w",encoding="utf-8")
        for w_ in words:
            # print(())
            for w in w_:
                if 1 <= len(w) <= 3:
                    three_under.write(w+"\n")
        model = Word2Vec.load(model_filename)
    chk = {}
    def simword(word):
        global chk

        dict_ = word.split()
        
        for w in dict_:
        #     for _ in range(df.query(f"token==\"{w}\"").values[0][1]):
            if w not in chk:
                print(w)
                chk[k] = 1
        wordemb_exsist = False
        for w in dict_:
            if w not in model.wv:
                wordemb_exsist = True 
        if wordemb_exsist:
            return []               
        for w in model.wv.most_similar(word.split(),topn=2000):
            if w[1] > 0.8:
                if w[0] not in chk:

                    try:
                        # print(w[0])
                        # continue
                        if df.query(f"token==\"{w[0]}\"").shape[0] > 0 and df.query(f"token==\"{w[0]}\"").values[0][1] > mincount:
                            # for _ in range(df.query(f"token==\"{w[0]}\"").values[0][1] ):
                            print(w[0])
                            chk[w[0]] = 1
                    except:
                        import traceback
                        traceback.print_exc()
                        print("----------------------")
                        print(w)
                        print("----------------------")
                        exit()
                dict_.append(w[0])      
        
        return dict_
    # while True:
    #     txt = input("input: ")
    #     print(model.wv.most_similar(txt.split(),topn=2000))
        
    #     if txt == "exit":
    #         exit()
        
    result = []
    result = result + simword("열대어")
    result = result + simword("구피")
    result = result + simword("어항")
    result = result + simword("질병")
    result = result + simword("프론토사")
    # result = result + simword("열대어 질병")
    result = result + simword("사료")
    result = result + simword("백점병")
    result = result + simword("수질")
    result = result + simword("여과기")
    result = result + simword("바닥재")
    result = result + simword("냉짱")
    result = result + simword("디스커스")
    result = result + simword("환수")
    result = result + simword("관상어")
    result = result + simword("치어")
    result = result + simword("제브라")
    result = result + simword("미니복어")
    result = result + simword("냉짱")
    result = result + simword("이끼")
    result = result + simword("코리도라스")
    # result = result + simword("")
    result = list(set(result))
    for res in result:
        result = result + simword(res)
    result = list(set(result))
    meanv = model.wv.get_mean_vector(result)
    # print(meanv)
    if True:
        vocab = list(model.wv.key_to_index.keys())

        vectors = []
        keys = []
        for k in vocab:
            keys.append(k)
            vectors.append(model.wv.get_vector(k))
        
        cossim = model.wv.cosine_similarities(meanv,vectors)
        
        for cos,k in zip(cossim,keys):
            # print(cos,k)
            if cos > minprob:
                if k not in chk:
                    # print(k)
                    # continue
                    if df.query(f"token==\"{k}\"").shape[0] > 0 and df.query(f"token==\"{k}\"").values[0][1] > mincount:
                        # for _ in range( df.query(f"token==\"{k}\"").values[0][1]):
                        print(k)
                        chk[k] = 1
    import sys           
    def sents(raw,sent,meanvec,word2vec):
        stmp = []
        for rawt in sent.split(" [SEP] "):
            stmp_ = []
            # ht = analysis(rawt,istext=True,mps=[])[0]
            # exit()
            # print(ht)
            rawt = rawt.strip().replace(" + ","+").replace(" +","+").replace("+ ","+").strip()
            # print(rawt)
            for ht_ in rawt.split():
                ht_ = ht_.strip("+")
                head = ht_.split("+")[0]
                
                if "/NOUN" in head:
                    head = head.split("/")[0]
                    if head in word2vec.wv:
                        stmp_.append(head)
            stmp.append(stmp_)
            # print(rawt)
        # print(stmp)
        # exit()
        # for w in sent:
        #     if w in word2vec.wv:
        #         stmp.append(w)
    
        # print(stmp)
        if len(stmp) > 0:
            for stmp_ in stmp:
                if len(stmp_) == 0:
                    continue
                svector = word2vec.wv.get_mean_vector(stmp_)
                cossim = word2vec.wv.cosine_similarities(meanvec,[svector])
                if cossim > minprob:
                    return raw#str(cossim[0])+"\t"+raw+"\t"+" ".join(sent)
        
        return None
        # else:
        #     print("None\t"+raw+"\t"+" ".join(sent))
    
    # import sys
    import sys
    import io
    from ht_utils import normalize,removesym

    rawtexts = []

    notropocal = open("qadataset_other_domain.txt","w",encoding="utf-8")
    with open("qadataset_sents_sim.txt","w",encoding="utf-8") as f:
        # print(len(sentences),len(words),len(rawfilename))
        length = len(sentences[8000:])
        pbar = tqdm(total=length)
        for raw,word_,rawfn in zip(sentences[8000:],wordssep[8000:],rawfilename[8000:]):
            
            rest = sents(raw.strip(),word_,meanv,model)
            
            if rest:
                f.write(rawfn + "\t" + rest+"\n")
                # f.write("<<EOD>>\n")
            else:
                notropocal.write(rawfn + "\t" + raw.strip()+"\n")
                # notropocal.write("<<EOD>>\n")
            pbar.update(1)
        pbar.close()
    notropocal.close()
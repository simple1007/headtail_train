from collections import defaultdict
counts = defaultdict(int)
lines_cnt = 11878186
from tqdm import tqdm


def pos_extract(pos,arr=["NNG","NNP"]):
    for i in arr:
        if i in pos:
            return True
    return False          

pbar = tqdm(total=lines_cnt)
note = open("eomi_error.txt","w",encoding="utf-8")
with open("kiwimorphs.txt",encoding="utf-8") as mp , open("kiwitags.txt",encoding="utf-8") as tg:
    for m,t in zip(mp,tg):
        for mm,tt in zip(m.strip().split(),t.strip().split()):
            
            tt = tt.split("+")
            # print(mm,tt)
            
            if "+" in mm and (pos_extract(tt[0])):
                # print(mm,tt)
                
                if tt[1][0] == "V" or tt[1][0] == "J" or tt[1][0] == "X":
                    counts[mm.split("+")[1]] += 1#+"/"+tt[1])
                
                tmp = [ts[0] for ts in tt[1].split("_")]
                tmp = "_".join(tmp)
                
                if "N_J" in tmp:
                    note.write(mm+ " " + tt[0]+"+"+tt[1] +"\n")
        pbar.update(1)
note.close()
pbar.close()
# print(len(counts))
# counts = Counter(nng_pos)
nng_pos = set()
dict_ = defaultdict(list)

for k,v in counts.items():
    if v >= 40:
        nng_pos.add(k)
        dict_["token"].append(k)
        dict_["count"].append(v)
    # else:
    #     note.write(k+"\n")
# # exit()
# note.close() 
import pandas as pd
import os
dataset = os.path.join(os.environ["Desktop"],"dataset")
df = pd.DataFrame(dict_)
df.to_csv(f"{dataset}\\eomi_count_ht.csv",encoding="utf-8-sig")
print(df.shape)

import pickle
with open("eomi.pkl","wb") as f:
    nng_ = []
    for i in nng_pos:
        nng_.append(i.split("/")[0])
    pickle.dump(nng_,f)
import pandas as pd
df = pd.read_csv(f"{dataset}\\homedari_autos_2_ht_count.csv")
df = df.loc[:,["token"]].values
import numpy as np
df = np.reshape(df,(df.shape[0])).tolist()

with open("nng_eomi.txt","w",encoding="utf-8") as f,open("nng_eomi_prev.txt","w",encoding="utf-8") as fp:
    def ngram(txt):
        txt = " " + txt + " "
        tks = []
        tksbi = []
        for i in range(len(txt)-2):
            bi = txt[i:i+3]
            tks.append(bi)
            tksbi.append(txt[i:i+2])
        return tks,tksbi
    
    def ngram_prev(txt):
        txt = " " + txt + " "
        txt = list(txt)
        txt.reverse()
        txt = "".join(txt)
        tks = []
        tksbi = []
        for i in range(len(txt)-2):
            bi = txt[i:i+3]
            tks.append(bi)
            tksbi.append(txt[i:i+2])
        return tks,tksbi
    
    # def space(txt):
    #     txt = " " + txt + " "
    #     tks = []
    #     for i in range(len(txt)-2):
    #         bi = txt[i:i+3]
            
    from collections import defaultdict
            
    dict_ = defaultdict(int)
    dict_p = defaultdict(int)
    dictuni_ = defaultdict(int)
    dictuni_p = defaultdict(int)
    print(list(nng_pos)[0])
    pbar = tqdm(total=len(nng_pos))
    # df = list(df) + ["먹고"] + ["학교"]
    for eomi in nng_pos:
        # print(eomi)
        # eomit = eomi.split("/")[0]
        # if eomit.startswith("X") or eomit.startswith("V"):
        #     continue
        # eomi = eomi#.split("/")[0]
        
        for noun in df:
            word = noun + eomi
            if len(noun) <= 1:
                print(noun)
                continue
            # f.write()
            bi_, uni_ = ngram(word)
            for bi,uni in zip(bi_,uni_):
                dict_[bi] += 1
                dictuni_[uni] += 1
                # f.write(bi+"|")
            # f.write("\n")
            bip_, unip_ = ngram_prev(word)
            for bip,unip in zip(bip_,unip_):
                dict_p[bip] += 1
                dictuni_p[unip] += 1
                # f.write(bip+"|")
            # f.write("\n")
        # f.write(noun.strip()+"\n")
        pbar.update(1)
        
    for k,v in dict_.items():
        value = v / dictuni_[k[:2]]
        f.write(k+"\t"+str(value)+"\n")

    for k,v in dict_p.items():
        value = v / dictuni_p[k[:2]]
        fp.write(k+"\t"+str(value)+"\n")
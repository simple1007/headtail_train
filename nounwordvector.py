import os
import numpy as np
import pandas as pd

from sklearn.metrics.pairwise import cosine_similarity
from customutils import get_mp_tags
from typing import List

datas = [
    os.path.join(os.environ["DATASET"],"headtail_250205_ht5.txt"),
    os.path.join(os.environ["DATASET"],"headtail_250213_ht5.txt"),
    os.path.join(os.environ["DATASET"],"headtail_250307_ht5.txt"),
    os.path.join(os.environ["DATASET"],"kccword2vec_data_ht5.txt")
]
from collections import defaultdict
counts = defaultdict(int)
all_noun_count = defaultdict(int)
# dataframe = defaultdict(int)
def get_nouns(line:str) -> List[str]:
    global counts
    sents = []
    cnt = 0
    for word in line.split():
        word = word.split("+")
        
        for w in word:
            wh,_ = get_mp_tags(w)
            if wh[1] in ["NOUN"]:
                for wh_ in nouns_split(wh[0]):
                    sents.append(wh_)
                    all_noun_count[wh_] += 1

    return sents

enc = "utf-8"

def get_mp_and_rawtxt(htpath,rawpath):
    ht = []
    raw = []
    with open(htpath,encoding=enc) as htf, open(rawpath,encoding=enc) as rawf:
        tmpht = []
        tmpraw = []
        for httk,rawtk in zip(htf,rawf):
            httk = httk.strip()
            rawtk = rawtk.strip()
            if httk.strip() == "<<EOD>>":
                ht.append(tmpht)
                raw.append(tmpraw)
                tmpht, tmpraw = [], []
            else:
                nouns = get_nouns(httk)
                tmpht.append(nouns)
                tmpraw.append(rawtk)
    return ht, raw

def get_tok(paths):
    heads, raws = [],[]
    for path in paths:
        htpath = path
        rawpath = path.replace("_ht5.txt",".txt").replace("headtail_","naverblog_autospace")
        ht_, raw_ = get_mp_and_rawtxt(htpath,rawpath)
        heads += ht_
        raws += raw_
    return heads, raws

from gensim.models import Word2Vec
    
def word2vec(paths,model_name="word2vec.model",load=False) -> tuple:

    heads, raws = get_tok(paths)
    w2v_doc = []
    for head in heads:
        if len(head) > 0:
            w2v_doc += head
    if load:
        return Word2Vec.load(model_name),heads, raws
    model = Word2Vec(w2v_doc,vector_size=100,window=5,min_count=2,hs=1,sg=1)
    model.save(model_name)
    model = Word2Vec.load(model_name)
    return model,heads,raws

def nouns_split(word):
    return word.split("_")

def simword(model: Word2Vec,word:str,sim_thres:float=0.8) -> List[str]:
    global counts
    sim_words = []
    for w in word.split():
        # for w_ in nouns_split(w):
        if w not in model.wv.key_to_index:
            print("NO",w)
            return []
        # else:
            
        counts[w] += 1
        sim_words.append(w)
    # print(model.wv.most_similar("구피"))
    for w in model.wv.most_similar(word.split(),topn=2000):
        if w[1] >= sim_thres:
            sim_words.append(w[0])
            counts[w[0]] += 1
    return sim_words

def simwords() -> dict:
    model,heads,raws = word2vec(paths=datas,load=True)
    words:List[str] = [
        "구피",
        "고정구피",
        "디스커스",
        "코리도라스",
        "열대어",
        "어항",
        "냉짱",
        "프론토샤",
        "체리새우",
        "치어",
        "수초",
        "흑사",
        "아로와나",
        "테트라",
        "배마름병",
        "바늘꼬리병",
        "열대어 질병",
        "열대어 먹이",
        "대형어",
        "시클리드",
        "백점병",
        "미니 복어",
        "모스 수초",
        "바닥재",
        "여과기",
        "스펀지 여과기",
        "여과재",
        "축약장",
        "안시",
        "코리",
        "물갈이",
        "폴렙테루스",
        "풀레드",
        "소금욕"
    ]
    results = []
    for w in words:
        results += simword(model,w)
    results = list(set(results))
    nouns_df = defaultdict(list)
    for k,v in counts.items():
        for _ in range(v):
            print(k)
    for k,v in all_noun_count.items():
        nouns_df["word"].append(k)
        nouns_df["count"].append(v)
    nouns_df = pd.DataFrame(nouns_df)
    nouns_df.to_csv("all_nouns.csv",encoding="utf-8-sig")
    mean_vec = np.array([model.wv.get_mean_vector(results).tolist()]) 
    return {"result_words":results,"model":model,"mean_vec":mean_vec,"heads":heads,"raws":raws}

def main():
    tropical_cnt = 0
    other_cnt = 0
    
    tropical_f = open("onxx_tropical_domain.txt","w",encoding=enc)
    other_f = open("onnx_other_domain.txt","w",encoding=enc)
    results = simwords()
    for h,r in zip(results["heads"],results["raws"]):
        is_other = True
        for hh in h:
            if len(hh) == 0:
                continue
            sentemb = results["model"].wv.get_mean_vector(hh)
            
            # if results["model"].cos
            
            sentemb = np.array([sentemb.tolist()])
            cossim = cosine_similarity(sentemb,results["mean_vec"])[0][0]
            if cossim > 0.8:
                tropical_f.write("\n".join(r)+"\n<<EOD>>\n")
                is_other = False
                tropical_cnt += 1
                break
                
        if is_other:
            other_f.write("\n".join(r)+"\n<<EOD>>\n")
            other_cnt += 1
    tropical_f.close()
    other_f.close()
    print("tropical:",tropical_cnt,"other:",other_cnt)
    
if __name__ == "__main__":
    main()
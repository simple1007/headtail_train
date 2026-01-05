import os
from collections import defaultdict
from customutils import AC
dataset = os.path.join(os.environ["DATASET"],"ht_dataset","modu","result","kiwimorphs_.txt")

nohtunigram = defaultdict(int)
unigram = defaultdict(int)
bigram = defaultdict(int)
trigram = defaultdict(int)

punigram = defaultdict(int)
pbigram = defaultdict(int)
ptrigram = defaultdict(int)

space = defaultdict(int)
spacelr = defaultdict(int)
htsep = "hththththt"

eomis = defaultdict(int)
def get_prev_unigram(text,unigram):
    prob = 0
    text = " " + text + " "
    for i in range(len(text)-1):
        uni = text[i:i+2]
        if uni in unigram:
            prob += unigram[uni]
    # print(1,len(text))
    return prob / (len(text) - 1)

def get_next_unigram(text,punigram):
    prob = 0
    text = list(text)
    text = list(reversed(text))
    text = "".join(text)
    text = " " + text + " "
    for i in range(len(text)-1):
        uni = text[i:i+2]
        if uni in punigram:
            prob += punigram[uni]
    
    return prob / (len(text) - 1)

def get_prev_bigram(text,bigram):
    prob = 0
    text = " " + text + " "
    for i in range(len(text)-2):
        uni = text[i:i+3]
        if uni in bigram:
            prob += bigram[uni]
    
    return prob / (len(text) - 2)

def get_next_bigram(text,pbigram):
    prob = 0
    text = list(text)
    text = list(reversed(text))
    text = "".join(text)
    text = " " + text + " "
    for i in range(len(text)-2):
        uni = text[i:i+3]
        if uni in pbigram:
            prob += pbigram[uni]
    
    return prob / (len(text) - 1)
aho = AC()
with open(dataset,encoding="utf-8") as f:
    for l in f:
        l = l.replace("+","_")
        l = l.replace("/","_")
        l = l.replace(htsep,"+")
        l = l.strip()
        ltmp = []
        for ll in l.split():
            ll = ll.strip("+").strip()
            if "+" in ll:
                eomi = ll.split("+")[1]
                eomis[eomi] += 1
                eomi = list(reversed(list(eomi)))
                eomi = "".join(eomi)
                aho.maketrie(eomi)
        continue
        l = " ".join(ltmp)
        ll = l
        if True:
        # for ll in l.split():
            ll = ll.strip()
            ll = " " + ll + " " 
            for t in ll:
                nohtunigram[t] += 1
                
            for i in range(len(ll)-1):
                uni = ll[i:i+2]
                unigram[uni] += 1
                
            for i in range(len(ll)-2):
                bi = ll[i:i+3]
                bigram[bi] += 1
        
            # for i in range(len(ll)-3):
            #     tri = ll[i:i+4]
            #     trigram[tri] += 1
            
            for i in range(len(ll)-2):
                tri = ll[i:i+3]
                space[tri] += 1
                spacelr[tri[0]+tri[2]] += 1
                
            ll = ll.strip()
            reverse_eoj = reversed(list(ll))
            reverse_eoj = "".join(list(reverse_eoj))
            reverse_eoj = " " + reverse_eoj + " "
            for i in range(len(reverse_eoj)-1):
                uni = reverse_eoj[i:i+2]
                punigram[uni] += 1
                
            for i in range(len(reverse_eoj)-2):
                bi = reverse_eoj[i:i+3]
                pbigram[bi] += 1
        
            # for i in range(len(reverse_eoj)-3):
            #     tri = reverse_eoj[i:i+4]
            #     ptrigram[tri] += 1

eomis = sorted(eomis.items(),key=lambda x: x[1],reverse=True)

for eomi in eomis:
    eomi = list(eomi)
    eomi[1] = str(eomi[1])
    # print("/".join(eomi))

aho.constructfail()
# print(aho.search("는서에"))
# exit()
import copy
unigram_ = copy.deepcopy(unigram)
bigram_ = copy.deepcopy(bigram)
punigram_ = copy.deepcopy(punigram)
pbigram_ = copy.deepcopy(pbigram)

spaceprob = {}
unigram = {}
bigram = {}
punigram = {}
pbigram = {}
for k,v in space.items():
    spaceprob[k] = v / spacelr[k[0]+k[2]]

for k,v in unigram_.items():
    prev = nohtunigram[k[0]]
    unigram[k] = v / prev

for k,v in bigram_.items():
    prev = unigram_[k[:2]]
    # if v != prev:
    #     print(k[:2])
    #     print(v,unigram_[k[:2]])
    # exit()
    bigram[k] = v / prev
    
    # space[k] =

for k,v in punigram_.items():
    prev = nohtunigram[k[0]]
    punigram[k] = v / prev

for k,v in pbigram_.items():
    prev = punigram_[k[:2]]
    pbigram[k] = v / prev
# print(bigram)
# print(spaceprob)   
# exit()
import pickle
with open("aho.pkl","wb") as f:
    pickle.dump(aho,f)
sent = "피라냐는 식인을 하는 아주 무서운 물고기이다 ."
sents = sent.split()
cnt = 0
# sent = " " + sent + " "
for idx,sent in enumerate(sents):
    # print(sent)
    sent = list(sent)
    sent = list(reversed(sent))
    sent_ = sent
    sent = "".join(sent)
    eomiidx = -1
    # print(sent)
    # if len(sent) > 2:
    prevlen = -1
    for i in range(len(sent)-1):
        eomi_ = sent[:i+1]
        # print(eomi_)
        res = aho.find(eomi_)
        # if len(res) == 0:
        #     break
        # print(res)
        if res:#len(res) > 0:
            # print(eomi_)
            eomiidx = max(prevlen,len(eomi_))
            prevlen = eomiidx
    sent = list(sent)
    if eomiidx > -1:
        # print(eomiidx)
        sent_[eomiidx] = sent_[eomiidx] + "+"
    sent_ = list(reversed(sent_))
    reseoj = "".join(sent_)
    # print(reseoj)
    # if s == " ":
    # if len(sent) < 2:
    #     print(sent)
    # elif len(sent) < 3:
    #     # sent = " " + sent + " "
    #     try:
    #         uni = unigram[sent[0]+"+"]
    #     except KeyError:
    #         uni = 0
            
    #     try:
    #         puni = punigram["+"+sent[1]]
    #     except KeyError:
    #         puni = 0
        
    #     prevp = 0
    #     if idx > 0:
    #         prevp = get_prev_unigram(sents[idx-1],unigram)
        
    #     nextp = 0
    #     if idx < len(sents)-1:
    #         nextp = get_next_unigram(sents[idx+1],punigram)
        
    #     cur = (uni + puni) / 2
    #     lr = prevp * 0.25 + cur * 0.5 + nextp * 0.25
    #     # print(lr,sent)

    # else:
    #     # prev
    #     prevp = 0
    #     if idx > 0:
    #         prevp = get_prev_unigram(sents[idx-1],unigram)
    #     else:
    #         prevp = 1
        
    #     nextp = 0
    #     if idx < len(sents)-1:
    #         nextp = get_next_unigram(sents[idx+1],punigram)
    #     else:
    #         nextp = 1
            
    #     biprevp = 0
    #     if idx > 0:
    #         biprevp = get_prev_bigram(sents[idx-1],bigram)
    #     else:
    #         biprevp = 1
        
    #     binextp = 0
    #     if idx < len(sents)-1:
    #         binextp = get_next_bigram(sents[idx+1],pbigram)
    #     else:
    #         binextp = 1
    #     sent = " " + sent + " "
    #     # psent = list(sent)
    #     # psent = list(reversed(sent))
    #     # sents = sent.split()
    #     for i in range(len(sent)):
    #         try:
    #             # print(sent[i])
    #             uni = unigram[sent[i]+"+"]
    #             # print(uni)
    #         except KeyError:
    #             # print(sent[i]+"+")
    #             uni = 0
    #         except IndexError:
    #             uni = 0
                
    #         try:
    #             # print(sent[i+1])
    #             puni = punigram["+"+sent[i+1]]
    #             # print(puni)
    #         except KeyError:
    #             puni = 0
    #         except IndexError:
    #             puni = 0
            
    #         if i+1 < len(sent)-1:
    #             pbi = sent[i+2] + sent[i+1]
    #         else:
    #             # print(sent[i],sent)
    #             pbi = ""
    #         bicur = 0
    #         bipcur = 0
            
    #         try:
    #             bi = sent[i-1] + sent[i]
    #         except IndexError:
    #             bi = ""
    #         try:
                
    #             bicur = bigram[bi+"+"]
    #         except KeyError:
    #             bicur = 0
    #         except IndexError:
    #             bicur = 0
    #         try:
    #             bipcur = bigram["+"+pbi]
    #         except KeyError:
    #             bipcur = 0
    #         except IndexError:
    #             bipcur = 0
    #         unicur = (uni + puni) / 2
    #         if pbi != "":
    #             bicur = (bicur + bipcur) / 2
    #         else:
    #             bicur = bipcur
            
    #         # spaceprob = 0
    #         htp = 0
    #         try:
    #             if sent[i] + "+" + sent[i+1] in spaceprob:
    #                 htp = spaceprob[sent[i] + "+" + sent[i+1]]
    #         except IndexError:
    #             htp = 0
    #         except IndexError:
    #             htp = 0
    #         cur = unicur * 0.25 + htp * 0.5 + bicur * 0.25
    #         # print(cur,)
    #         print(unicur,htp,bicur)
    #         # lr = (prevp*0.4+biprevp*0.6) * 0.25 + cur * 0.5 + (nextp * 0.4 + binextp * 0.6) * 0.25
    #         print(2,cur,sent[i],sent)

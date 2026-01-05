enc = "utf-8"
from collections import defaultdict
#from lib2to3.pgen2.parse import ParseError
import re
from tqdm import tqdm

# with open("morphs.txt",encoding=enc) as f:
#     wdict = defaultdict(int)

#     for l in f:
#         l = l.strip().split()#.replace("+"," ")
        
#         # l = l.split()
#         for ll in l:
#             if "+" in ll:
#                 # print(ll)
                
#                 tk = ll.split("+")[1]
#                 if not re.findall(r"\d+",tk) and not re.findall(r"[a-zA-Z]",tk):
#                     # if tk not in wdict:
#                     wdict[tk] += 1

# with open("tails.txt",'w',encoding=enc) as f:
#     for w,c in wdict.items():
#         if c > 9:
#             f.write(w+"\t"+str(c)+"\n")

# exit()
if True:
    with open("kcc150.txt",encoding=enc) as f:
        count = 0
        for l in f:
            count += 1 
        wdict = defaultdict(int)
        f.seek(0)
        pbar = tqdm(total=count) 
        c = 0
        for l in f:
            # print(l)
            pbar.update(1) 
            c += 1
            # l = l.strip().replace("+"," ")
            # l = l.strip()         
            l = l.split()
            # print(l)
            for ll in l:
                ll = ll.strip()
                #if ll.count("+") > 1:
                    #print(ll)
                if '+' not in ll:
                    tk = ll.split("/")[0]
                    #if tk not in wdict:
                        # print(tk)
                    wdict[tk] += 1
                else:
                    tk = ll.split("+")#[:-1]
                    for tt in tk:
                        tk = tt.split("/")[0]
                        wdict[tk]+=1
                     
                    #continue
                    #tk = ll.split("+")[-1].split("/")[0]
                    # # print(tk)
                    # if tk.strip() == "":
                    #    continue
                    #if tk[0] in n_tags:
                    #    wdict[tk[0]] += 1
                    #    wdict[tk[1:]] += 1
                    #elif tk[:2] in n_tags:
                    #    wdict[tk[:2]] += 1
                    #    wdict[tk[2:]] += 1
                    #else:
                    #    wdict[tk] += 1
        # print(c)
    print(len(wdict))
    pbar.close() 
    temp = defaultdict(int)
    with open("tokens.txt",'w',encoding=enc) as f:
        sorts = sorted(wdict.items(),key= lambda x: x[1],reverse=True)
        # counts = set()
        temp2 = defaultdict(int) 
        
        for w,c in tqdm(sorts):#wdict.items():
            # counts.add(c)
            # continue
            # if c > 30000:
            if len(w) != 1:
                temp[w] += c
                # f.write(w+"\t"+str(c)+"\n")
            else:
                temp2[w] += 1
            # else:
                # temp_syl[]
        sorts = sorted(temp.items(),key=lambda x: x[1], reverse=True) 
        for w,c in sorts[:37000]:
            f.write(w+"\t"+str(c)+"\n")
        sorts = sorts[37000:] + sorted(temp2.items(),key=lambda x: x[1], reverse=True)
        for w,c in tqdm(sorts): 
            for w_ in w:
                # if w in temp:
                #     temp[w_] += c
                # else:
                temp2[w_] += c        
       # sorts = sorted(list(counts))
        # percent = int(len(sorts) * 0.8)
        # # print(sorts)
        # print(sum(list(counts))/len(counts))
        # print(sum(list(sorts[:percent]))/len(sorts[:percent]))
        # exit()
        # sorts = sorted(temp.items(),key = lambda x: x[1])

        for s, c in temp2.items():
            # if c > 3:    
            f.write(s+"\t"+str(c)+"\n")
print(len(sorts))
worddict_ = []
with open("tokens.txt",encoding=enc) as f:
    for l in f:
        l = l.strip().split("\t")
        # print(l)
        try:
            int(l[0])
        except ValueError:
            if len(l) == 2:
                worddict_.append([l[0],l[1]])

import random
import copy
length = len(worddict_)
mask_tokens_length = int(length * 0.3)

indexs = [i for i in range(length)]
masks = random.sample(copy.deepcopy(indexs),mask_tokens_length)

del indexs

with open("masks_tokens.txt","w",encoding=enc) as f:
    # f.write()
    for i in masks:
        f.write(worddict_[i][0]+"\t"+str(worddict_[i][1])+"\n")
exit()
from konlpy.tag import Komoran
from tqdm import tqdm
k = Komoran()
tags = ['X','J','E']
with open("KCC150_Korean_sentences_UTF8.txt",encoding=enc) as f:
    wdict = {}
    with tqdm(total=len(f.readlines())) as pbar:
        f.seek(0)
        for index,l in enumerate(f):
            # l = l.strip().replace("+"," ")
            # print("\r {}".format(index),end='')
            # l = l.split()
            l = k.pos(l)
            for ll in l:
            
                if ll[1][0] in tags:
                    #continue
                    1==1
                # tk = ll.split("/")[0]
                if ll[0] not in wdict:
                    wdict[ll[0]] = 11
            pbar.update(1)
print()
print(len(wdict))

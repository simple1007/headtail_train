# -*- coding: utf-8 -*-
import pandas as pd
import re
import os
from collections import defaultdict
from ht_utils import removesym,tagged_reduce
from jamo import h2j,j2hcj


tagsep = "@@@"
htsep = "hththththt"
desktop=os.environ["Desktop"] + "/dataset/ht_dataset/modu/train/"
if __name__ == "__main__":
    with open(f"{desktop}tail.dict",encoding="utf-8") as f:
        
        dict_ = defaultdict(int)
        dict_2 = defaultdict(int)
        heads_list = []
        count = 0
        tmpcount = 0
        for l in f:
            # print(l)
            # l = l.replace("+","_").replace("/SP","_/SP")
            # print("abc"+l.strip()+"abc")
            # print()
            l = l.strip()
            if l.split("\t")[0] in ["1","2"]:
                l = re.sub(r" +",r" ",l)
                l = l.split("\t")
                # print(l)
                head = l[1].strip("_")
                tail = l[2].split("+")[0]
                lht = l[2].split("+")[0].split(tagsep)
                lht = lht[0] + "/" + lht[1].strip("_")
                
                head_ = head.split("/")
                headmp_ = head_[0]
                # print(head_)
                head_ = head_[1].split("_")
                if len(head_) >= 2:
                    # print(head_)
                    headt = tagged_reduce("_".join(head_))#head_[0]+"_"+head_[-1]
                    # print(headt)
                else:
                    headt = head_[0]
                l[2] = [k for k in l[2].split(htsep)]
                l[2] = " ".join(l[2])
                # print(l)
                # exit()
                l[2] = l[2].replace(f"/{tagsep}",f"_{tagsep}")
                l[2] = l[2].replace(htsep,"+").replace(tagsep,"/")
                # print(6,l)
                # print(l[2])
                # print(l[2])
                # exit()
                # if l[2].strip() == "":
                #     # print(l)
                #     exit()
                dict_[headmp_+"/"+headt+"="+l[2]]+=1
                # print(7,l,tail)
                
                # print(l)
                # exit()
                # if headt.startswith("V") and headmp_ == "갔":
                #     print(headmp_+"/"+headt+"="+l[2])
                # if 
                # print(dict_)
                # exit()
            elif l.split("\t")[0] in ["3","4"]:
                # if "잘랐" in l:
                #     print(l)
                l = re.sub(" +"," ",l)
                l = l.split("\t")
                # print(l[1])
                
                head = l[1].strip("_").split("/")
                # print(head)
                if "NN" in head[1]:
                    continue
                # print(l)
                tail = l[2].strip("_").split(tagsep)
                tmp = ""
                # if "ㄱ" <= tail[0][0] <= "ㅎ":
                #     tmp = tail[0][-1]
                if tmp!="":
                    # tail_tmp = tmp + tail[0]
                    print(head[0]+" "+tail[0])
                # print(removesym(head[0]),head,tail)
                if head[0] == "|":
                    continue
                # print(head,tail)
                # print(l)
                try:
                    l[3] = l[3].replace(htsep,"+").replace(tagsep,"/").split("+")
                    l[3][0] = l[3][0].split("/")[0][-1] + "/" + l[3][0].split("/")[1]
                    # print(l)
                    # exit()
                    # if "/V" in l[0]:
                    
                        # print(l[3])
                    # else:
                        # l[3] = l[3][0]
                except:
                    # print(l)
                    # exit()
                    tmpcount += 1
                    continue
                # exit()
                # print(l[3])
                # exit()
                # print(l)
                # exit()
                if head[1].startswith("V") or head[1].startswith("XSV") or head[1].startswith("XSA"):
                    # print(l)
                    # exit()
                    length = 0
                    for mptmp in l[3][:2]:
                        length += len(mptmp.split("/")[0])
                        
                    # print(l[3],length)
                    # print(l[3])
                    if len(l[3]) > 1:
                        firsteum = j2hcj(h2j(l[3][1]))
                    if length == 2 and "/E" in l[3][1] and len(l[3]) > 1 and (firsteum[0] == "ㅇ" or "ㄱ" <= l[3][1] <= "ㅎ"):
                        # print(l)
                        # exit()
                        l[3] = " ".join(l[3][:2])
                    else:
                        l[3] = " ".join(l[3][:1])
                    # all = l[3] = " ".join(l[3])
                    dict_2[tmp+head[0][-1]+"/"+head[1]+"="+l[3]] += 1 #+tail[0][-1]+"/"+tail[1]]+=1
                elif len(l[3]) >= 2:
                    # print(3,l)
                    # exit()
                    l[3] = " ".join(l[3][0])
                    # all = l[3] = " ".join(l[3])
                    dict_2[tmp+head[0][-1]+"/"+head[1]+"="+l[3]] += 1 #+tail[0][-1]+"/"+tail[1]]+=1
                
                # print(head)
                # print(l[3])
                # print(dict_2)
                # exit()
                heads_list.append(tail[0])
            count += 1
# print(tmpcount)    
# exit()  
res = defaultdict(list)
ktmp = defaultdict(int)
dict_ = [[k,v] for k,v in dict_.items()]
dict_ = sorted(dict_,key=lambda x: x[1],reverse=True)
for k,v in dict_:
    k = k.split("=")
    if k[1] == "":
        print(k)
        exit()
    # ktmp[]
    # print(k)
    # ktmp = k[0].split("@")
    # print(ktmp)
    # ktmp[1] = tagged_reduce("_"+ktmp[1])
    # # k[0]
    # ktmp[0] = ktmp[0].split("/")
    # ktmp[0][1] = tagged_reduce(ktmp[0][1])
    # ktmp[0] = "/".join(ktmp[0])
    # ktmp = "@".join(ktmp)
    # print(k)
    # k[0] = "@".join(k[0])
    # k[0] = k[0].split("/")
    # k[0] 
    res["token"].append(k[0])
    res["res"].append(k[1])
    res["count"].append(v)
# exit()
df = pd.DataFrame(res)
# df = df.query("count >= 50")
df.to_csv("tail_ori_dict.csv",encoding="utf-8-sig")

from datetime import datetime
start = datetime.now()
print(df.query("token==\"했었다/XSV_EP\""))
print(datetime.now()-start)
res = defaultdict(list)
dict_2 = [[k,v] for k,v in dict_2.items()]
dict_2 = sorted(dict_2,key=lambda x: x[1],reverse=True)

for k,v in dict_2:
    k = k.split("=")
    # print(k)
    # print(k[1])
    # exit()
    # print(k)
    res["token"].append(k[0])
    res["res"].append(k[1])
    res["count"].append(v)
    heads_list.append(k[-1])

import pickle
with open("heads.dict","wb") as f:
    heads_list = set(heads_list)
    print("가" in heads_list)
    pickle.dump(heads_list,f)

df = pd.DataFrame(res)
# df = df.query("count >= 50")
df.to_csv("head_ori_dict.csv",encoding="utf-8-sig")
start = datetime.now()
# print(df)
# exit()
print(df.query("token==\"갔/VV\""))
# t = df.query("token==\"갔/VV\"")
# ser = pd.Series(["그"])
# t = t.query("")
def apply(txt):
    if txt.startswith("그"):
        return True
    return False
# t["flag"] = t["res"].apply(apply)
# print(t.query("flag==True"))
# print(t[t.str.startswith(("그"))])
print(datetime.now() - start)


head_dict = pd.read_csv("head_ori_dict.csv",encoding="utf-8-sig")
tail_dict = pd.read_csv("tail_ori_dict.csv",encoding="utf-8-sig")

def to_dict(dic_,mode="head"):
    dict_ = defaultdict(set)
    # if mode != "tail":   
    dic_ = dic_.query("count > 4")
    dic = dic_.loc[:,["token","res"]].values.tolist()
    for t,r in dic:
        # print(t,r)
        if mode=="tail":
            t = t.split("@")
            # if len(t)
            # print(t,r)
            if len(t) > 1:
                # print(j2hcj(h2j(t[0][0])),j2hcj(h2j(r[0])))
                # exit()
                # print(t)
                t1 = t[1]#.split("_")
                if len(t1.split("_")) > 1:
                    t0 = tagged_reduce(t[0])
                    t1 = tagged_reduce("_"+t1)
                    # if "-" in t1[0]:
                    #     t1[0] = tagged_reducet1[0].split("-")[0]
                    # if "-" in t1[-1]:
                    #     t1[-1] = t1[-1].split("-")[0]
                    # t1 = t1[0]+"_"+t1[-1]
                    t = t0 + "@" + t1
                    # print(t)
                else:  
                    # t[0] = t[0]
                    t[1] = tagged_reduce("_"+t[1])  
                    t = "@".join(t)
            else:
                t[0] = tagged_reduce(t[0])
                if len(t) > 1:
                    t[1] = tagged_reduce("_"+t[1])
                t = "@".join(t)
            # print(t)
        # print(t,r)
        # exit()
        # print(t,r)
        # print(j2hcj(h2j(t[0]))[0], j2hcj(h2j(r[0]))[0])
        if mode == "tail":
            t_ = j2hcj(h2j(t[0]))#[0]
            r_ = j2hcj(h2j(r[0]))#[0]
        else:
            # t__ = t
            t__ = t.split("/")
            r__ = r.split("/")
            # print
            # print(t__,r__)
            t_ = j2hcj(h2j(t__[0][-1]))#[0]
            r_ = j2hcj(h2j(r__[0][-1]))#[0]
            # print(5,t_)
        from joinjamo import join_jamos
        # print(join_jamos(t_),join_jamos(r_))
        # exit()
        ttmp = t
        # if "갔" in ttmp:
        #     print(t_,r_,t,r)
        #     exit()
        # print(t,r)
        if t_[0] == r_[0]:
            # print(t,r)
            # exit()
            if mode != "tail":
                # print(t,r)
                if False:# "@" in t__[1]:
                    tt__ = t__[1].split("@")
                    tt__[1] = tagged_reduce("_"+tt__[1])
                    t__[1] = "@".join(tt__)
                    # print(t__)
                # print(4,t_)
                # print(join_jamos("ㄱㅏ").encode())
                # print(join_jamos(t_[:2]).encode())
                # t = t__[0][:-1] + join_jamos(t_[:2]) + "/" + t__[1]
                # r = r__[0][:-1] + join_jamos(r_[:2]) + "/" + r__[1]
                # print(t,r)
                # if r.strip()=="":
                # print(0,t)
                # print(0,r)
                # print(t.encode().decode("utf-8").encode())
                # print("가/VV@EP_EF".encode("euc-kr").decode("utf-8").encode())
                import copy
                t = copy.deepcopy(t)
                r = copy.deepcopy(r)
                    # exit()
                dict_[t.strip()].add(r.strip())
                
                # if "갔" in ttmp:
                #     # print(t__,r__)
                #     print(dict_["가/VV@EP_EF"])
                #     print("가/VV@EP_EF"==t)
                #     print(dict_[t])
                #     print(1,t)
                #     print(2,r)
                #     exit()
                # if ttmp == "갔":
                # print(t,r)
                
            # elif mode != "tail":
            #     print(t,r)
            #     dict_[t].add(r)
            
        elif mode == "tail":
            # print(1,t,r)
            # print(t)1
            dict_[t].add(r)
        # elif mode == "tail":
            # dict_[t].add(r)
    
    return dict_

head_dict = to_dict(head_dict)
# exit()
tail_dict = to_dict(tail_dict,mode="tail")
# exit()
print("까꿍")
# print(head_dict)
# print(tail_dict)
print(head_dict["갔/VV@EP_EF"])
print(head_dict["했/VV@EP_EF"])
# exit()
import pickle
with open("head.dict","wb") as f:
    pickle.dump(head_dict,f)

with open("tail.dict","wb") as f:
    pickle.dump(tail_dict,f)
# head_dict.to_csv("head.dict",encoding="utf-8-sig")
# tail_dict.to_csv("tail.dict",encoding="utf-8-sig")

def to_csv_dict(dic,filename):
    rescsv = defaultdict(list)
    for k,v in dic.items():
        # for vv in v:
        v = list(v)
        tmpv = []
        tmpvv = []
        for vv in v:
            if vv.strip() == "":
                continue
            if len(vv.split()) == 1:
                tmpv.append(vv)
            elif len(vv.split()) >= 2:
                tmpvv.append(vv)
        if len(tmpv) > 0:
            v = tmpv
        elif len(tmpvv) > 0:
            v = tmpvv
        else:
            continue
        v = "|".join(v)
        rescsv["tk"].append(k)
        rescsv["ori"].append(v)
    
    import pandas as pd
    df = pd.DataFrame(rescsv)
    df.to_csv(filename,encoding="utf-8-sig")
    dict_ = defaultdict(set)
    for d in df.loc[:,["tk","ori"]].values:
        ishan = re.search(r"[가-힣]",d[0])
        # print(d,ishan)
        if d[1] != "" and ishan:
            # print(d)
            
            for dd in d[1].split("|"):
                # print(dd)
                dict_[d[0]].add(dd)
                # print()
    
    return dict_

head_dict = to_csv_dict(head_dict,"head.csv")
tail_dict = to_csv_dict(tail_dict,"tail.csv")
print(head_dict["갔/VV@EP_EF"])
with open("head.dict","wb") as f:
    pickle.dump(head_dict,f)
print(tail_dict["니다/VV@EF"])
# with open("tail.dict","wb") as f:
#     pickle.dump(tail_dict,f)
from collections import defaultdict
from ht_utils import tagged_reduce
res = defaultdict(int)
with open("list.txt",encoding="utf-8") as f:
    for l in f:
        l = l.strip()
        l = l.split("\t")
        if l[0] != "8":
            continue
        # print(l)
        tags = l[1].split("/")[1]
        mp = l[1].split("/")[0]
        l[3] = l[3].replace(" @@@ ","@@@")
        if "_" in tags:
            tmp = ""
            res_ = []
            
            # print(l[3])
            for tk in l[3].split():
                # print(tk)
                # print(tk)
                ttkmp = tk.split("@@@")[0]
                ttktag = tk.split("@@@")[1]
                
                tmp += ttkmp
                
                if len(tmp) > len(mp):
                    res_.append(tk)
                
            
            l[2] = l[2].split("/")
            l[2][1] = tagged_reduce("_"+l[2][1].strip("_"))
            # l[2] = "/".join(l[2])
            
            l[1] = l[1].split("/")
            l[1][1] = tagged_reduce("_"+l[1][1].strip("_"))
            # l[1] = "/".join(l[2])
            # print(l)
            # print(l[2][0]+f"/{l[1][1]}@{l[2[1]]}"+"=="+" ".join(res_))
            # res[l[2][0]+f"/{l[1][1]}@{l[2][1]}"+"=="+" ".join(res_)+"=="+l[-1]] += 1
            res[l[2][0]+f"/{l[2][1]}"+"=="+" ".join(res_)+"=="+l[-1]] += 1
        
        else:
            l[2] = l[2].split("/")
            l[2][1] = tagged_reduce("_"+l[2][1].strip("_"))
            # l[2] = "/".join(l[2])
            
            l[1] = l[1].split("/")
            l[1][1] = tagged_reduce("_"+l[1][1].strip("_"))
            # if l[2][0]
            # res[l[2][0]+f"/{l[1][1]}@{l[2][1]}"+"=="+" ".join(l[3].split()[1:])+"=="+l[-1]] += 1
            res[l[2][0]+f"/{l[2][1]}"+"=="+" ".join(l[3].split()[1:])+"=="+l[-1]] += 1

tail_dict_set = defaultdict(set)
count = defaultdict(int)
numbers = defaultdict(list)
tkori_sep = " SEP "
xss = defaultdict(list)
from jamo import h2j,j2hcj
import re
def pre_xs(k,vv):
    vvtmp = vv.split()
    k_ = k
    vv_ = vv.split()
    if f"@@@XS" in vvtmp[0]:
        
        vvtmp = vvtmp[1:]
        vv = " ".join(vvtmp)
        xstag = vvtmp[0].split("@@@")
        ktmp = k.split("/")
        if len(ktmp[0]) > len(xstag[0]):
            ktmp[0] = ktmp[0][len(vv_[0].split("@@@")[1]):]
            xssm = k_.split("/")[0][:len(vv_[0].split("@@@")[0])]
            xsst = vv_[0]
            # print(vv_)
            if len(vv_) >= 2 and "@@@E" in vv_[1]:

                # print(j2hcj(h2j(vv_[1][0])[0]))
                
                if re.match(r"[ㄱ-ㅎ]",vv_[1].split("@@@")[0]) and len(vv_[1].split("@@@")[0]) == 1:
                    # print(vv_[1].split("@@@")[0],re.match(r"[ㄱ-ㅎ]",vv_[1].split("@@@")[0]))
                    xsst = xsst + "_" + vv_[1]
                elif j2hcj(h2j(vv_[1][0])[0]) == "ㅇ" and len(vv_[1].split("@@@")[0]) == 1:
                    xsst = xsst + "_" + vv_[1]
                    # print(xsst)
                    
            vvtmp_ = list(map(lambda x: x.split("@@@")[1],vvtmp))
            if len(vvtmp) > 1:
                tag = vvtmp_[0] + "_" + vvtmp_[-1]
                ktmp[1] = tag
            else:
                # else:
                tag = vvtmp_[0]
                ktmp[1] = tag
            # ktmp[0] = k
            k = "/".join(ktmp)
            # print("\n",9999,vv_,k_,xstag,k,vv_[1:])
            # print(8888,xssm,xsst)
            return k," ".join(vv_[1:]),xssm,xsst
    return k,vv,None,None

for k,v in res.items(): 
    k = k.split("==")
    # print(k)
    # if 15 > v:
    if "_" in k[0]: 
        k[0],k[1],xssm,xsst = pre_xs(k[0],k[1])
        if xssm != None and xsst != None:
            xss[xssm].append(xsst)
        tail_dict_set[k[0]].add(k[1])
        count[k[0]+tkori_sep+k[1]] += v
        numbers[k[0]+tkori_sep+k[1]].append(k[-1])

import json
delkeys = []

import re
for k,v in xss.items():
    # xss[k] = list(v)
    tmp = defaultdict(int)
    for vv in v:
        # vv = vv.split("_")[0]
        tmp[vv] += 1
        
        
# for k,v in xss.items():    
    oritmp = []
    # v = set(v)
    # for vv in v:
        # key = k+"@"+vv
        # for vvv in tmp[key]:#.items():
            # print(kk,vv)
        # if tmp[key] > 10:
        # if vv > 10:
    for kk,vv in tmp.items():
        if vv > 10:
            oritmp.append(kk)
        elif "_" in kk and re.match(r"[ㄱ-ㅎ]",kk.split("_")[1][0]):
            if j2hcj(h2j(k[0]))[0] == j2hcj(h2j(kk.split("_")[0][0]))[0]: 
                oritmp.append(kk)
    
    if len(oritmp) > 0:
        xss[k] = oritmp
    else:
        # del xss[k]
        delkeys.append(k)
for dkey in delkeys:
    del xss[dkey]
            
print(f"\n{json.dumps(xss,indent=4,ensure_ascii=False)}\n")
taildicttmp = defaultdict(list)

for k,v in tail_dict_set.items():
    for vv in v:
        key = k +tkori_sep + vv
        taildicttmp[k].append(vv)
        if vv.strip() == "":
            continue
        # if "_" not in k:
        # print(90,key)
        # print(91,count[key])
        # print(92,numbers[key])
        # print()
# exit()
def map_fn(x):
    # print(x)
    return x.split("-")[1]
        
resdicttmp = defaultdict(list)
for k,v in taildicttmp.items():
    
    # resdicttmp["res"].append(k)
    # if count[key] > 15:
    for vv in v:
        key = k + tkori_sep + vv
        if count[key] <= 20:
            continue

        resdicttmp["token"].append(k)
        resdicttmp["ori"].append(vv)
        resdicttmp["count"].append(count[key])

        number_tmp = list(map(map_fn,list(numbers[key])))
        resdicttmp["indexing"].append("|".join(number_tmp))

import pandas as pd
df = pd.DataFrame(resdicttmp)
df.to_csv("tailtmpdict2.csv",encoding="utf-8")
dff = df.loc[:,["indexing"]].values
dic = set()
for dff_ in dff:
    # dic.add(dff_[0])
    for num in dff_[0].split("|"):
        dic.add(num)

taildict = defaultdict(set)
for dff_ in df.loc[:,["token","ori"]].values:
    # for num in dff_[0].split("|"):
        # dic.add(num)
    # print(dff_)
    taildict[dff_[0]].add(dff_[1])
    
# print(xss)

import pickle
with open("tail2.dict","wb") as f:
    pickle.dump(taildict,f)
with open("error_line2.pkl","wb") as f:
    pickle.dump(dic,f)
# for num_ in dic:
#     print(num_)
    
exit()
# for k,v in res.items():
#     k = k.split("==")
    
#     tail_dict["token"].append(k[0])
#     tail_dict["res"].append(k[1])
#     tail_dict["count"].append(v)
    
import pandas as pd

# df = pd.DataFrame(tail_dict_set)

# df.to_csv("tail_dict.csv",encoding="utf-8-sig")
import re
def to_csv_dict(dic,filename):
    rescsv = defaultdict(list)
    dict_ = defaultdict(set)
    for k,v in dic.items():
        # for vv in v:
        # v = list(v)
        tmpv = []
        tmpvv = []
        # print(2,k,v)
        lines = []
        for vv in v:
            if vv != "":
                vv = vv.split("_")
                tmpv.append(vv[0])
                lines.append(vv[1])
                # dict_[k].add(vv[0])
        # for vv in v:
        #     if vv.strip() == "":
        #         continue
                # if len(vv.split()) == 1:
                # tmpv.append(vv)
                # elif len(vv.split()) >= 2:
                #     tmpvv.append(vv)
        # if len(tmpv) > 0:
            # v = tmpv
        # elif len(tmpvv) > 0:
            # v = tmpvv
        # else:
        #     continue
        # v = tmpv
        if len(tmpv) > 0:
            for v_ ,li in zip(tmpv,lines):
                # v = "|".join(tmpv)
                # lines = list(map(str,lines))
                rescsv["tk"].append(k)
                rescsv["ori"].append(v_)
                rescsv["lines"].append(li)
        
    import pandas as pd
    df = pd.DataFrame(rescsv)
    df.to_csv(filename,encoding="utf-8-sig")
    # dict_ = defaultdict(set)
    # print(rescsv)
    # for d in df.loc[:,["tk","ori"]].values:
    #     ishan = re.search(r"[가-힣]",d[0])
    #     # print(d,ishan)
    #     if d[1] != "" and ishan:
    #         # print(d)
    #         for dd in d[1].split("|"):
    # #             # print(d)
    #             dict_[d[0]].add(dd)
    return dict_
tail_dict = to_csv_dict(tail_dict_set,"tail_dict.csv")

# print(tail_dict)
print(len(tail_dict))
import pickle
with open("tail.dict","wb") as f:
    pickle.dump(tail_dict,f)
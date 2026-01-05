import pandas as pd
import os
from collections import defaultdict
from ht_utils import tagged_reduce
dictd = defaultdict(int)
dictd2 = defaultdict(list)
datapath = os.path.join(os.environ["DATASET"],"ht_dataset","kiwitags.txt")
datapath2 = os.path.join(os.environ["DATASET"],"ht_dataset","kiwimorphs.txt")

# with open(datapath,encoding="utf-8") as f:
#     ff = open(datapath2,encoding="utf-8")
#     for l in f:
#         l = l.strip().split()
#         l2 = ff.readline().strip().split()
#         for ll in l:
#             ll = ll.split("+")[0]
#             ll2 = l2.pop(0)
#             # if ll.count("_") > 0:
#             #     ll = ll.split("_")[0] +"_"+ ll.split("_")[-1]
#             dictd[tagged_reduce(ll)] += 1
#             if dictd[tagged_reduce(ll)] < 10:
#                 dictd2[tagged_reduce(ll)].append(ll2)
# dictddf = defaultdict(list)
# for k,v in dictd.items():
#     dictddf["tags"].append(k)
#     dictddf["count"].append(v)
#     dictddf["words"].append(",".join(dictd2[k]))
# df = pd.DataFrame(dictddf)
# df.to_csv("tags.csv",encoding="utf-8-sig")
# exit()
with open("homedari_new.txt",encoding="utf-8") as f:
    nouns = defaultdict(int)
    for l in f:
        l = l.strip()
        for ll in l.split():
            ht = ll.split("+")
            
            ht_ = ht[0].split("/")
            
            if ht_[1] == "NNP" or ht_[1] == "NNG":
                nouns[ht[0]] += 1
                
    
    nouns = [[k,v] for k,v in nouns.items()]
    nouns = sorted(nouns,key=lambda x: x[1],reverse=True)
    
    result = {"token":[],"count":[]}
    
    for k,v in nouns:
        if len(k.split("/")[0]) > 1:
            result["token"].append(k)
            result["count"].append(v)
    
    df = pd.DataFrame(result)
    df.to_excel("nouns.xlsx")
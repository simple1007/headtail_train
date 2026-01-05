import re
import os
import copy
import torch
from customutils import *
from collections import deque
from headtail_util import get_mp_tags
import numpy as np
import pandas as pd
import pickle

def xstag_readdic(path):
    vdic = []
    with open(path,"rb") as f:
        vdic.append(pickle.load(f))
        vdic.append(pickle.load(f))
    # NXMP1902008040.json_xsv.pkl
    return vdic

def vstag_readdic(path):
    with open(path,"rb") as f:
        oridic = pickle.load(f)
    
    return oridic

if os.path.exists("NXMP1902008040.json_xsn.csv"):
    xsn_dict = xstag_readdic("NXMP1902008040.json_xsn.pkl")
    xsa_dict = xstag_readdic("NXMP1902008040.json_xsa.pkl")
    xsv_dict = xstag_readdic("NXMP1902008040.json_xsv.pkl")
    nvv_dict = xstag_readdic("NXMP1902008040.json_nvv.pkl")
    nva_dict = xstag_readdic("NXMP1902008040.json_nva.pkl")
    # nvv_dict = remove_eq_mp(nvv_dict)    
    # nva_dict = remove_eq_mp(nva_dict)
    
    ovv_dict = vstag_readdic("NXMP1902008040.json_vv.pkl")
    ova_dict = vstag_readdic("NXMP1902008040.json_va.pkl")

    
def get_dict_value(df):
    return df[0]
    # return df[0].loc[:,["word"]].values.tolist()[0][0].replace(" @@@ ","/")

def get_dict_nvvalue(value):
    valuetmp = value.split("+")
    if len(valuetmp) >= 2:
        if valuetmp[1].startswith("을") or valuetmp[1].startswith("는") or valuetmp[1].startswith("를"):
            return valuetmp[0]
    return value.replace(" @@@ ","/")

def xstag_join(head,tail,lastnum,htres):
    head_ = head
    if len(head_[0]) > lastnum * -1:
        head_[0] = head_[0][:lastnum] + "/NOUN+" + htres
        if len(tail) > 0:
            res = head_[0] + "+" + "/".join(tail)
            return res
        else:
            return head_[0]
    return ""

def nvtag_getlast_search(vdf,hlast):
    # print(2222,hlast)
    flag,res = vdf.find(hlast)
    
    return flag

def nvtag_join(head,tail,lastnum,htres,vdf,tagname):
    head_ = head
    head_[0] = list(head_[0])
    head_[0][lastnum] = htres 
    head_[0] = "".join(head_[0])
    
    honelast = xstag_getlast(head,-1)
    htwolast = xstag_getlast(head,-2)
    hthreelast = xstag_getlast(head,-3)
    
    oneres = nvtag_getlast_search(vdf,honelast)
    twores = nvtag_getlast_search(vdf,htwolast)
    threeres = nvtag_getlast_search(vdf,hthreelast)
    # print(oneres,twores,threeres,htres,head[0])
    res = ""
    # print(head_)
    nvheadtk = head_[0].split("/")[0]
    # print(nvheadtk)
    if threeres and len(nvheadtk[0]) > 3:
        headvnoun = xstag_getvnoun(head_,-3)
        headv = xstag_getlast(head_,-3)
        if "/" in htres:
            res = headvnoun + "/NOUN+"+headv+f"/{tagname}"+"+"+htres.split("+")[-1]
        else:
            res = headvnoun + "/NOUN+"+headv+f"/{tagname}"
    elif twores and len(nvheadtk) > 2:
        headvnoun = xstag_getvnoun(head_,-2)
        headv = xstag_getlast(head_,-2)
        # print(333,htres)
        if "/" in htres:
            res = headvnoun + "/NOUN+"+headv+f"/{tagname}"+"+"+htres.split("+")[-1]
        else:
            res = headvnoun + "/NOUN+"+headv+f"/{tagname}"
    elif oneres and len(nvheadtk) > 1 :
        headvnoun = xstag_getvnoun(head_,-1)
        headv = xstag_getlast(head_,-1)
        # print(2222,headvnoun,headv)
        if "/" in htres:
            res = headvnoun + "/NOUN+"+headv+f"/{tagname}"+"+"+htres.split("+")[-1]
        else:
            res = headvnoun + "/NOUN+"+headv+f"/{tagname}"
        
    if len(tail) > 0 and res.strip() != "":
        # print(2222,res,tail)
        res = res + "+" + "/".join(tail)
        return res
    else:
        return res
    
def vtag_join(head,tail,htres,vdf):
    head_ = head
    # head_[0] = list(head_[0])
    # head_[0][-1] = htres
    # head_[0] = "".join(head_[0])
    # print(head_)
    # headtmp = head_[0].split("+")[0].split("/")
    # print(headtmp)
    vhead = htres.split("+")[0].split("/")[0]
    # print(head_[0][:-1]+vhead)
    dictres = nvtag_getlast_search(vdf,head_[0][:-1]+vhead)
    # print(dictres)
    # print(11,vhead,htres)
    # print(vhead,head_[0],head[0])
    if dictres:
        # htrestmp = htres.split("+")[1:]
        # htrestmp = "+".join(htrestmp)
        # print(htres)
        if len(tail) > 0:
            return head_[0][:-1]+htres + "+" + "/".join(tail)
        else:
            return head_[0][:-1]+htres

    return ""

def xstag_getvnoun(head,lastnum):
    headtmp = head[0].split("/")[0]
    headvnoun = headtmp[:lastnum]
    return headvnoun

def xstag_getlast(head,lastnum):
    headtmp = head[0].split("/")[0]
    headlast = headtmp[lastnum:]
    # print(headlast)
    return headlast

def xstag_getlast_search(df,word):
    flag,res = df[0].find(word)
    if flag:
        res_ = []
        for w in res:
            res_.append(df[1][w].replace(" @@@ ","/"))
        # print(res_)
        return res_
    else:
        return list()

def get_xsn(hteoj):
    head,tail = get_mp_tags(hteoj)
    honeres, htwores = "", "" 
    hlastone = xstag_getlast(head,-1)#head[0][-1]
    hlasttwo = xstag_getlast(head,-2)#head[0][-2:]
    
    onedict = xstag_getlast_search(xsn_dict,hlastone)#xsn_dict.query(f"ori==\"{hlastone}\"")
    if len(onedict) > 0:
        honeres = get_dict_value(onedict)
    
    twodict = xstag_getlast_search(xsn_dict,hlasttwo)#xsn_dict.query(f"ori==\"{hlasttwo}\"")
    if len(twodict) > 0:
        htwores = get_dict_value(twodict)
        
    if htwores != "":
        res = xstag_join(head,tail,-2,htwores)
        if res != "":
            return res
    elif honeres != "":
        res = xstag_join(head,tail,-1,honeres)
        if res != "":
            return res
    return hteoj

def get_xsa(hteoj):
    head,tail = get_mp_tags(hteoj)
    honeres, htwores = "", "" 
    hlastone = xstag_getlast(head,-1)#head[0][-1]
    hlasttwo = xstag_getlast(head,-2)#head[0][-2:]
    
    onedict = xstag_getlast_search(xsa_dict,hlastone)#xsn_dict.query(f"ori==\"{hlastone}\"")
    if len(onedict) > 0:
        honeres = get_dict_value(onedict)
    
    twodict = xstag_getlast_search(xsa_dict,hlasttwo)#xsn_dict.query(f"ori==\"{hlasttwo}\"")
    if len(twodict) > 0:
        htwores = get_dict_value(twodict)
        
    if htwores != "":
        res = xstag_join(head,tail,-2,htwores)
        if res != "":
            return res
    elif honeres != "":
        res = xstag_join(head,tail,-1,honeres)
        if res != "":
            return res
    return hteoj

def get_xsv(hteoj):
    head,tail = get_mp_tags(hteoj)
    honeres, htwores = "", "" 
    hlastone = xstag_getlast(head,-1)#head[0][-1]
    hlasttwo = xstag_getlast(head,-2)#head[0][-2:]
    
    onedict = xstag_getlast_search(xsv_dict,hlastone)#xsn_dict.query(f"ori==\"{hlastone}\"")
    if len(onedict) > 0:
        honeres = get_dict_value(onedict)
    
    twodict = xstag_getlast_search(xsv_dict,hlasttwo)#xsn_dict.query(f"ori==\"{hlasttwo}\"")
    if len(twodict) > 0:
        htwores = get_dict_value(twodict)
        
    if htwores != "":
        res = xstag_join(head,tail,-2,htwores)
        if res != "":
            return res
    elif honeres != "":
        res = xstag_join(head,tail,-1,honeres)
        if res != "":
            return res
    return hteoj

def get_nvv(hteoj):
    head,tail = get_mp_tags(hteoj)
    hlastone = xstag_getlast(head,-1)
    
    onedict = xstag_getlast_search(nvv_dict,hlastone)#xsn_dict.query(f"ori==\"{hlastone}\"")
    # print(onedict)
    ovvres = nvtag_join(head,tail,-1,hlastone,ovv_dict,"VV")
    # print(111,ovvres,onedict)
    if ovvres != "":
        # print(ovvres)
        return ovvres
    # print(onedict)
    for one in onedict:
        # print(one)
        # if len(one.split("+")).startswith()
        htres = get_dict_nvvalue(one)
        # print(htres)
        ovvres = nvtag_join(head,tail,-1,htres,ovv_dict,"VV")
        # print(ovvres)
        if ovvres != "":
            hovvtmp = ovvres.split("+")[1].split("/")
            if hovvtmp[0] != head[0][-len(hovvtmp[0]):]: 
                return ovvres
        
    return hteoj
def head_eq(head,htres):
    if head[0] == htres.split(" @@@ ")[0]:
        return True
    return False
def get_nva(hteoj):
    head,tail = get_mp_tags(hteoj)
    hlastone = xstag_getlast(head,-1)
    # print(head,tail)
    onedict = xstag_getlast_search(nva_dict,hlastone)#xsn_dict.query(f"ori==\"{hlastone}\"")
    
    ovvres = nvtag_join(head,tail,-1,hlastone,ova_dict,"VA")
    # print(onedict)
    if ovvres != "":
        # ovvres = ovvres.split("+")[:-1]
        # ovvres = "+".join(ovvres)
        # print(ovvres)
        return ovvres
    for one in onedict:

        htres = get_dict_nvvalue(one)
        # print(1111,htres)
        ovvres = nvtag_join(head,tail,-1,htres,ova_dict,"VA")
        if ovvres != "":
            hovvtmp = ovvres.split("+")[1].split("/")
            if hovvtmp[0] != head[0][-len(hovvtmp[0]):]: 
                return ovvres
        
    return hteoj

def get_vv(hteoj):
    head,tail = get_mp_tags(hteoj)
    hlastone = xstag_getlast(head,-1)
    
    onedict = xstag_getlast_search(nvv_dict,hlastone)#xsn_dict.query(f"ori==\"{hlastone}\"")
    # print(onedict)
    # ovvres = vtag_join(head,tail,hlastone,ovv_dict)
    # # print(ovvres)
    # if ovvres != "":
    #     return ovvres
    # print(onedict)
    for one in onedict:
        htres = get_dict_nvvalue(one)
        ovvres = vtag_join(head,tail,htres,ovv_dict)
        if ovvres != "":
            # print(ovvres)
            hovvtmp = ovvres.split("+")[0].split("/")
            if hovvtmp[0] != head[0][-len(hovvtmp[0]):]: 
                return ovvres
        
    return hteoj

def get_va(hteoj):
    head,tail = get_mp_tags(hteoj)
    hlastone = xstag_getlast(head,-1)
    
    onedict = xstag_getlast_search(nva_dict,hlastone)#xsn_dict.query(f"ori==\"{hlastone}\"")
    
    # ovvres = vtag_join(head,tail,hlastone,ova_dict)
    # if ovvres != "":
    #     return ovvres
    for one in onedict:
        htres = get_dict_nvvalue(one)
        ovvres = vtag_join(head,tail,htres,ova_dict)
        if ovvres != "":
            hovvtmp = ovvres.split("+")[0].split("/")
            if hovvtmp[0] != head[0][-len(hovvtmp[0]):]: 
                return ovvres
        
    return hteoj

def get_xstags(hteoj):
    # print(hteoj)
    if "XSN" in hteoj:# or "NOUN" in hteoj:
        # print(hteoj)
        return get_xsn(hteoj)
    elif "XSV" in hteoj:
        return get_xsv(hteoj)
    elif "XSA" in hteoj:
        return get_xsa(hteoj)
    elif "NVV" in hteoj:
        return get_nvv(hteoj)
    elif "NVA" in hteoj:
        return get_nva(hteoj)
    elif "VV" in hteoj:
        return get_vv(hteoj)
    elif "VA" in hteoj:
        return get_va(hteoj)
    
    return hteoj
# print(get_xstags("노동대통령들/XSN+은/J"))
# print(get_xstags("노동대통령했/XSV+다/EF"))
# print(get_xstags("의심스런/XSA+놈/NNB"))

# print(get_xstags("노동대통령줬/NVV+다/EF"))
# print(get_xstags("뼈아프/NVA+다/EF"))
# print(get_xstags("된/VV+다/EF"))
# print(get_xstags("죄송했/VA+다/EF"))

# exit()
            
def normalize(l):
    # l = removesym(l)
    l = l.strip()
    hanja = "\u2e80-\u2eff\u31c0-\u31ef\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fbf\uf900-\ufaff"
    l = re.sub(r"([^가-힣|a-z|A-Z|\d|\s|.?!])", " ",l)#r" \1 ",l)
    l = re.sub(r"([a-z|A-Z]+)", r" \1 ",l)#r" \1 ",l)
    l = re.sub(r"(\d+)", r" \1 ",l)#r" 1 ",l)
    l = re.sub(r"(\d)+\s.\s(\d)+",r" \1.\2 ",l)
    l = re.sub(r"([.!?]+)",r" \1 ",l)
    l = re.sub(r" +"," ",l).strip()
    return l
    l = re.sub(r"([^\uAC00-\uD7A30-9a-zA-Z\sㄱ-ㅎㅏ-ㅣ"+hanja+r"])", r" \1 ", l)
    l = re.sub(r"([ㄱ-ㅎㅏ-ㅣ]+)",r" \1 ",l)
    hanreg = r"(["+hanja+r"]+)"
    l = re.sub(hanreg,r" \1 ",l)
    # print(l)
    l = re.sub(r" +"," ",l).strip()                
    # l = re.sub(r"([^가-힣|ㄱ-ㅎ|a-z|A-Z|\d+]|\d+\.\d+)", r" \1 ",l)
    # l = removesym(l)
    l = re.sub(r"(\d+)",r" \1 " ,l)
    # l = re.sub(r"(\d+)([월일년분초원])")
    l = re.sub(r"([a-z|A-Z]+)",r" \1 ",l)
    l = re.sub(r" +"," ",l).strip()                
    
    l = re.sub(r"(name)\s(\d+)",r"무명",l)
    l = re.sub(r" +"," ",l).strip()                
    
    l = re.sub(r"(address)\s(\d+)",r"주택",l)
    # l = re.sub(r"(xx)+[가힣]")
    # l = re.sub(r"([가-힣]+).\d+")
    return l.strip()
    # print(re.findall(r"([^가-힣|ㄱ-ㅎ|a-z|A-Z|\d|\s])",l))
    l = re.sub(r"([^가-힣|ㄱ-ㅎ|a-z|A-Z|\d|\s])", r" \1 ",l)
    # print(l)
    # print(l)
    # l = re.sub(r"(\d+\.\d+|\d+)",r" \1 ",l)
    l = re.sub(r"(\d+)",r" \1 " ,l)
    l = re.sub(r"([a-z|A-Z]+)",r" \1 ",l)
    # l = re.sub(r"(\.|\,|\?|\!|\~|\#|\@|\$|\%|\^|\&|\*|\(|\)|\=|\+|\-|\_|\/|\>|\<|\;|\:|\'|\"|\]|\[|\{|\}|\`|\|\\)",r" \1 ",l)
    l = re.sub(r" +",r" ",l)
    # l = re.sub(r"(\d+) \. (\d+)",r" \1.\2 ",l)
    l = re.sub(r" +",r" ",l)
    l = l.strip()
    return l
# print(normalize("ㅜㅜ베타"))
# exit()
# txt = "1 단계 사업기간인 2015 년까지 국토해양부가 해마다 ‘ 1 조원+알파 ( α ) ’ 등 6 조원 , 환경부는 10 조원 , 농림수산식품부는 1 조 ~ 3 조원을 투입할 것으로 추정된다 ."
# txt = re.escape(txt)
# print(normalize("3cm"))
# print(normalize("3.3cm"))
# exit() 


filepath = __file__
filepath = filepath.split(os.sep)[:-1]
filepath = os.sep.join(filepath)

symbollistpath = os.path.join(filepath,"postagger_model","symlist.txt")
symlists = []
with open(symbollistpath,encoding="utf-8") as f:
    for symbol in f:
        if symbol.strip() != "":
            symlists.append(re.escape(symbol.strip()))
symbols = r"[" + r"|".join(symlists) + r"]"
# print(symbols)
# def removesym(l):
#     # l = re.sub(symbols,"○",l)
#     l = re.sub(r"([^가-힣|a-z|A-Z|\d|\s])", r"♥",l)
    
#     return l.strip()

def removesym(l):
    # l = re.sub(symbols,"○",l)
    return l
    hanja = "\u2e80-\u2eff\u31c0-\u31ef\u3200-\u32ff\u3400-\u4dbf\u4e00-\u9fbf\uf900-\ufaff"
    hanreg = r"["+hanja+r"]+" 
    l = re.sub(hanreg, r"♥",l)
    l = re.sub(r"[^가-힣|a-z|A-Z|\d|\s|ㄱ-ㅎ|ㅏ-ㅣ|]", "♥",l)
    l = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s가-힣ㄱ-ㅎㅏ-ㅣ]", "♥", l)
    l = re.sub(r" +"," ",l).strip()
    
    return l.strip()

# print(normalize("요런 새로운 가치를 추x해 나가는 과정에서"))
# print(removesym("요런 새로운 가치를 추 x 해 나가는 과정에서"))
# exit()
# print(re.sub(r"[^가-힣|a-z|A-Z|\d|\s|ㄱ-ㅎ|ㅏ-ㅣ]",r"♥","하이 ㄹㄹ }]"))
# print(removesym("123가나다라死後"))
# print(normalize("123가나다라死後"))
# exit()
# exit()

# print(removesym("하핳ㅎㅎ 좋지 않당( ੭ ･ᴗ･ )੭ 무튼 옷구경도 열심히 하고 , 책구경도 하고 빵구경까지 다 하고‍♀️ 지하 1층으로 갔는데..."))
# exit()
# def removesym(l):
#     with open(symbollistpath,encoding="utf-8") as f:
#         # lists = []
#         # print(l)
#         for sym in f:
#             if sym.strip() != "":
    #             # print("error")
    #             l = l.replace(sym.strip(),"○")
    #         # lists.append(l.strip())
    
    # return l
# print(removesym("{"))
# exit()
def is_sym(txt):
    txt_ = " " + txt + " "
    if re.match(r"\sname\d+\s",txt_):
        return None
    if re.match(r"\saddress\d+\s",txt_):
        return None
    res = re.search(r"[^가-힣|ㄱ-ㅎ]",txt)
    # if re.search(r"")
    # l = re.sub(r"[^가-힣|a-z|A-Z|\d|\s|ㄱ-ㅎ|ㅏ-ㅣ]", r"♥",l)
    # l = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s가-힣ㄱ-ㅎㅏ-ㅣ]", r"♥", l)
    if res == None:
        return None
    return "hi"

def tagged_reduce(tag,word=None):
    # return tag.strip("_")
    # if tag == "NNP_EC_JX_VCP_EP_EF":
    #     print(word)
    # if tag.strip("_") == "VX_EP_EC_VV_EF":
    #     print(tag,word)
    #     exit()
    if tag.startswith("_"):# or tag.startswith("E") or tag.startswith("J") or tag.startswith("VCP") or tag.startswith("VCN"):
        # return tag
        # return "TAIL"

        tag = tag[1:].rstrip("_")
        # return tag
        tag = tag.split("_")
        
        res = []
        for tt in tag:
            if "-" in tt:
                tt = tt.split("-")[0]
            res.append(tt)
            # if tt == "EF":
            #     res.append(tt)
            # elif tt[0] == "X":
            #     res.append("X")
            # else:
            #     res.append(tt[0])
            
        # if res[0] in ["XSA","XSV","XSN"]:
        #     if len(res) > 1:
        #         res[0] = "X"
            # if res[0] == "XSA":
            #     res[0] = "VA"
            # elif res[0] == "XSV":
            #     res[0] == "VV"
            # elif res[0] == "XSN":
            #     res[0] == "NNP"    
        tag = res
        if len(tag) == 1 or tag[0] == tag[-1]:
            return tag[0]
        else:
            tags = [tag[0]] + [tag[-1]]
            # if tag[0] == "XSV" or tag[0] == "XSA":
            #     if len(tag) > 2:
            #         tags = [tag[1]] + [tag[-1]]
            #     else:
            #         tags = [tag[1]]
            return "_".join(tags)
    else:
        
        # if tag == ""

            # exit()
        # return tag
        # if tag == "NNP_JKS":
        #     print(word)
        #     exit()
        # res = []
        # tmp = []
        tmp = tag.strip("_").split("_")
        # tmptmp = []
        # for tmp_ in tmp:
        #     if "-" in tmp_:
        #         tmp_ = tmp_.split("-")[0]
        #     tmptmp.append(tmp_)
        # tmp = tmptmp
        if len(tmp) == 1 or tmp[0] == tmp[-1]:
            return tmp[0]
        # else:
        #     tag = tmp[0] + "_" + tmp[-1]
        #     tag = re.sub("NNG_NNG","NNG",tag)
        #     tag = re.sub("NNG_NNP","NNP",tag)
        #     tag = re.sub("NNP_NNG","NNP",tag)
        #     tag = re.sub("NNP_NNP","NNP",tag)
            # return tag.strip().strip()
        # nountmp = []
        
        # for ntag_ in tag.strip("_").split("_"):
        #     nountmp.append(ntag_[0])
        # if tag == "MAG_XSA_ETM_NNB":
        #     print(nountmp)
        #     exit()
        # tagcount = tmp.count("NNG") + tmp.count("NNP") + tmp.count("NP")+ + tmp.count("NR")+ + tmp.count("NNB")#+ tmp.count("NNB") + tmp.count("NP") + tmp.count("NR")
        # tag = "_".join(tmp)
        # # if (tagcount > 0 and len(tmp) != tagcount):
        # #     print(tmp)
        # lastvtag =""
        # for tmpp in tmp:
        #     if not tmpp.startswith("E") and  tmpp in ["XSV","XSA","XSN","VV","VA","VX"]:
        #         lastvtag = tmpp
        # eomitag = ""
        # if lastvtag != tmp[-1] and lastvtag != "":# and tmp[-1].startswith("E"):
        #     eomitag = "_"+tmp[-1]
        # if lastvtag == "XSN":
        #     return "XNNG"+eomitag
        # elif lastvtag == "XSV": #or t[-1] == "VV":
        #     return "XVV"+eomitag
        # elif lastvtag == "XSA":
        #     return "XVA"+eomitag
        # elif lastvtag == "VV":
        #     return "NVV"+eomitag
        # elif lastvtag == "VA":
        #     return "NVA"+eomitag
        # elif lastvtag == "VX":
        #     return "Vv"+eomitag
        # elif tmp[-2] == "XSV": #or t[-1] == "VV":
        #     return "XVV"+"_"+tmp[-1]
        # elif tmp[-2] == "XSA":
        #     return "XVA"+"_"+tmp[-1]
        # elif tmp[-2] == "VV":
        #     return "NVV"+"_"+tmp[-1]
        # elif tmp[-2] == "VA":
        #     return "NVA"+"_"+tmp[-1]
        # elif tmp[-2] == "VA":
        #     return "NVA"+"_"+tmp[-1]
        # if tagcount >= 2 or (tagcount > 0 and len(tmp) != tagcount):#(tmp[0]!="N" or tmp[-1] != "N")):#tmp.count("N") >= 2 and tmp.count("N") == len(tmp):
        #     # res.append("C")
        #     # if not tmp[0].startswith("N"):
        #         # res.append(tmp[0].split("-")[0])
        #     if tmp.count("NNG") >= 1 and tmp.count("NNP") >= 1:
        #         res.append("NNP")
        #         return "NNP"
        #     elif tmp.count("NNP") >= 1:
        #         return "NNP"
        #     elif tmp.count("NNG") >= 1:
        #         res.append("NNG")
        #         return "NNG"
        #     elif tmp.count("NP") >= 1:
        #         return "NP"
        #     elif tmp.count("NR") >= 1:
        #         return "NR"
        #     elif tmp.count("NNB") >= 1:
        #         return "NNB"
        #     else:
        #         res.append("NNP")
        #         return "NNP"

        #     return "_".join(res)
        # else:
        #     return tmp[0] + "_" + tmp[-1]
        # return tag
        # counttag = defaultdict(int)
        # if "XSA" in tag:
        #     return "VA"
        # if "XSV" in tag:
        #     return "VV"
        # if tag.startswith("MAG_VV"):
        #     return "VV"
        # for t in tag.split("_"):
        #     if "-" in t:
        #         t = t.split("-")[0]
        # #     if t not in counttag:
        # #         counttag[t] += 1                
        #     res.append(t)
        # # if res[0] in ["XSA","XSV","XSN"]:
        # #     if len(res) > 1:
        # #         res = res[1:]
        # # if len(res) > 1:
        # #     return res[0]+"_"+res[-1]
        # # else:
        # #     return res[0]
        # # tag = "_".join([res[0],res[1]])
        # countflag = defaultdict(int) 
        # restmp = [] 
        # for r in res:
        #     if r not in countflag:
        #         restmp.append(r)
        #         countflag[r] += 1
        
        # if len(restmp) >= 2:
        #     restmp = [restmp[0],restmp[-1]]
        # else:
        #     restmp = [restmp[0]]
        # tag = re.sub(r"(NNP_NNG|NNG_NNP)+",r"_NNP_",tag)
        # tag = tag.strip("_")
        # tag = re.sub(r"_+","_",tag)
        tag = tag#"_".join(restmp)
    return tag
from transformers import BertTokenizerFast
from collections import defaultdict

class HTInputTokenizer:
    def __init__(self):
        import os
        path = os.sep.join(__file__.split(os.sep)[:-1])
        self.tokenizer = BertTokenizerFast.from_pretrained(os.path.join(path,"kcctokenizer"))
        self.pos2index = {"[PAD]":0,"[SOS]":1,"[EOS]":2,"+":3,"O":4,"[UNK]":5}
        self.index2pos = {v:k for k,v in self.pos2index.items()}
        self.posindex = 6
        self.pad = self.tokenizer.pad_token_id
        self.cls = self.tokenizer.cls_token_id
        self.sep = self.tokenizer.sep_token_id
        self.space = self.tokenizer.convert_tokens_to_ids('▁')
        self.htsep = "hththththt"
    def __len__(self):
        return len(self.tokenizer)
    
    def make_intoken(self,x):
        # print(len(self.tokenizer))
        # exit()
        X = []
        for s in x:
            # s = s.replace("_","")
            # s = removesym(s)
            stmp = []
            # s = s.replace(" + ","+").replace(" +","+").replace("+ ","+")
            # print(s)
            # exit()
            for st in s.split():
                st = st.strip("+")
                stmp.append(st)
            # print(
                # stmp)
            # exit()
            # print(s)
            # exit()
            s = " ".join(stmp)
            s = s.strip().replace("+"," + ").split()
            # print(s)
            # print(s)
            X.append(s)
        
        tokens = self.tokenizer(X,padding="longest",is_split_into_words=True,return_offsets_mapping=True,return_tensors="pt")
        # print(tokens["input_ids"])
        return tokens    
    def maketoken(self,sents,poss):
        X = []
        Y = []
        # from collections import defaultdict
        maxlen = 0
        xs = []#defaultdict(list)
        sents = copy.deepcopy(sents)
        poss = copy.deepcopy(poss)
        for s,p in zip(sents,poss):
            # s = s.replace("_","")
            
            ltmp = [] 
            for ss in s.split():
                ss = ss.strip(self.htsep)
                ltmp.append(ss)
            s = " ".join(ltmp)
            
            ptmp = []
            for pp in p.split():
                pp = pp.strip(self.htsep)
                ptmp.append(pp)
            # p = " ".join(ptmp)
            
            s = s.replace(self.htsep,"+")
            p = p.replace(self.htsep,"+")
  
            st = copy.deepcopy(s)

            sori = copy.deepcopy(s)

            ptmp = []
            p = ["[SOS]"] + p.strip().replace("+"," + _").split() + ["[EOS]"]
            s = s.replace("+"," + ").split()
            for pp in p:
                pp = tagged_reduce(pp).strip("_")
                ptmp.append(pp)
            # print(s,p)
            # exit()
            Y.append(ptmp)
            xs.append([st,s])
            X.append(s)
        tokens = self.tokenizer(X,truncation=True,padding="longest",max_length=350,is_split_into_words=True,return_offsets_mapping=True)
        
        maxlen = len(tokens["input_ids"][0])
        
        X = copy.deepcopy(X)
        Y = copy.deepcopy(Y)

        X,Y = self.mapping(xs,tokens,Y,maxlen)
        
        return X,Y
    def to_mp_tagdict(self,x):
        Xtmp = []
        # for xx in x:
        line=[]
        for xt in x.split():
            xx = xt.split("+")
            tmp=[]

            for xxx in xx:
                if self.tags and xxx in self.tags:
                    tmp.append(xxx)
                elif self.tags and xxx not in self.tags:
                    tmp.append("Fail")
                elif not self.tags:
                    tmp.append(xxx)
            line.append("+".join(tmp))
        # Xtmp.append(" ".join(line))
        return " ".join(line)
    
    def train_pos(self,poss,tags=None,words=None):
        # print(poss)
        self.tags = tags
        if words:
            words = words.split()
        
        for idx,pos in enumerate(poss.split()):
            # cnt = 0
            # for pos_ in pos.split():
            if words:
                word = words[idx]
                word_ = word.split(self.htsep)
            else:
                word_ = None
            pos_ = pos.replace("+","+_")
            for p in pos_.split("+"):
                p = tagged_reduce(p,word=word_).strip("_")
                # if p not in tags:
                #     print(p,tags[p])
                # exit()
                if tags and p in tags:
                    if "B_"+p not in self.pos2index:
                        self.pos2index["B_"+p] = self.posindex
                    
                        self.index2pos[self.posindex] = "B_"+p
                        self.posindex += 1
                        
                        self.pos2index["I_"+p] = self.posindex
                        self.index2pos[self.posindex] = "I_"+p
                        
                        self.posindex += 1
                elif not tags:
                    if "B_"+p not in self.pos2index:
                        self.pos2index["B_"+p] = self.posindex
                        self.index2pos[self.posindex] = "B_"+p
                        self.posindex += 1
                        
                        self.pos2index["I_"+p] = self.posindex
                        self.index2pos[self.posindex] = "I_"+p
                        
                        self.posindex += 1                   
    def mapping(self,sents,tokens,poss,maxlen):
        # tks = res["input_ids"]
        X = []
        Y = []
        resx = None
        resy = None
        # print(tokens.keys())
        import copy
        for idx, (sent, tks, mapping_) in enumerate(zip(sents,tokens["input_ids"],tokens["offset_mapping"])):
            c = 0
            pos = poss[idx]
            tks = tks
            sent = sents[idx]
            tks = deque(tks)
            pos = deque(pos)
            postmp = copy.deepcopy(pos)
            tkstmp = copy.deepcopy(tks)
            mappingtmp = copy.deepcopy(mapping_)
            # poa = pos
            tmp = []
            x = []
            y = []

            tmp = []
            try:
                xt = []
                yt = []
                for i in mapping_:
                    
                    if i[0] == 0 and len(tmp) != 0:# and c != 0:
                        if tmp[-1] == self.space and len(tmp) == 1:
                            y.append([self.pos2index["O"]])
                            x.append(tmp)
                            xt.append(self.tokenizer.convert_ids_to_tokens(tmp))
                            yt.append("O")
                            tmp = []
                            
                        else:
                            p = pos.popleft()
                            # if p == "+":
                            #     # print(1,p,tmp)
                            #     x.append(tmp)
                            #     y.append([self.pos2index["O"]])
                                
                            #     tmp = []
                            if p in ["[SOS]","[EOS]","[PAD]","+"]:
                                y.append([self.pos2index[p]])
                                x.append(tmp)
                                xt.append(self.tokenizer.convert_ids_to_tokens(tmp))
                                yt.append(p)
                                tmp = []
                            else: 
                                ptmp = []
                                ytmp = []
                                for _ in tmp:
                                    if "I_"+p in self.pos2index:
                                        ytmp.append("I_"+p)
                                        ptmp.append(self.pos2index["I_"+p])
                                    else:
                                        ytmp.append("[UNK]")
                                        ptmp.append(self.pos2index["[UNK]"])
                                if "B_" + p in self.pos2index:
                                    ptmp[0] = self.pos2index["B_" + p]
                                    ytmp[0] = "B_" + p
                                    # ytmp[0] = "B_" + p
                                # else:
                                
                                xt.append(self.tokenizer.convert_ids_to_tokens(tmp))
                                yt.append(ytmp)    
                                y.append(ptmp)
                                x.append(tmp)
                                tmp = []
                    
                    tk = tks.popleft()
                    if tk == self.pad:
                        break
                    tmp.append(tk)
                    c+= 1
                # print(xt,yt)
                # for xxxx, yyyy in zip(xt,yt):
                    # print(xxxx,"/",yyyy,"AT",end=" ")
                # print()
                # exit()         
                resx = []
                resy = []
                for xx, yy in zip(x,y):
                    resx = resx + xx
                    resy = resy + yy
                
                if resx[-1] != self.sep:
                    resx = resx + [self.sep]
                    resy = resy + [self.pos2index["[EOS]"]]

                resx = resx + [self.pad for _ in range(maxlen-len(resx))]
                resy = resy + [self.pos2index["[PAD]"] for _ in range(maxlen-len(resy))]
                # if resx[:600][-1] == 8:
                    # resy[:600][-1] = self.pos2index[""]
                X.append(resx)
                Y.append(resy)
            except Exception as ex:
                print(ex)
                print(sent)
                print(postmp)
                print(x)
                print(y)
                # print(tmp,p)
                # exit()
        return X,Y
        
    def to_tag(self,tag):
        tags = []
        for t in tag:
            tmp = []
            for t_ in t:
                tmp.append(self.index2pos[t_])
            tags.append(tmp)
        
        return tags
    
    def to_mp(self,x):
        # print(x)
        tmp = []
        for xx in x:
            # print(xx)
            tmp.append(self.tokenizer.convert_ids_to_tokens(xx))
        # print(tmp)
        # exit()
        return tmp
    
def infer(subwordvocab:HTInputTokenizer,model,x,device):
    with torch.no_grad():
        xs = []
        for idx,xx in enumerate(x):
    
            xx = xx.strip().strip("+").strip()
            xs.append(xx)
        X = subwordvocab.make_intoken(xs)#subwordvocab.tokenizer(xs,padding="longest",max_length=600,is_split_into_words=True,return_offsets_mapping=True,return_tensors="pt")
        # print(X["input_ids"].shape)
        # exit()
        Y = []
       
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16):
            X["input_ids"]= X["input_ids"].to(device)
            tmp = torch.tensor([[0]]).to(device)
            y = model(X["input_ids"])
            # y = beam_search_decode(model, X["input_ids"],subwordvocab=subwordvocab, beam_size=3,device=device)
            # y = torch.tensor(y).to(device)
            y = torch.argmax(y,dim = -1)
            # print(subwordvocab.to_tag(y))
            # exit()
        for sent,xx,mapping,yy in zip(xs,X["input_ids"],X["offset_mapping"],y):
            xxtmp = copy.deepcopy(xx)
            yytmp = copy.deepcopy(yy)
            res = []
            xy = xx[xx != subwordvocab.pad]
            xx = xx[:xy.shape[0]].cpu().numpy().tolist()
            yy = yy[:xy.shape[0]].cpu().numpy().tolist()
            xx = deque(xx)
            yy = deque(yy)
            tmp = []
            ptmp = []
            c = 0
            for offset in mapping:
                if offset[0] == 0 and c!= 0:
                    if tmp[-1] == subwordvocab.space and len(tmp) == 1:
                        res.append([subwordvocab.to_mp([tmp])[0],subwordvocab.to_tag([ptmp])[0]])
                        tmp = []
                        ptmp = []
                    else:
                        res.append([subwordvocab.to_mp([tmp])[0],subwordvocab.to_tag([ptmp])[0]])
                        tmp = []
                        ptmp = []
                if len(xx) == 0:
                    # print(res)
                    break
                    # exit()
                c += 1
                tk = xx.popleft()
                p = yy.popleft()
                tmp.append(tk)
                ptmp.append(p)
                if tk == subwordvocab.pad:
                    break
            restmp = []
            for r in res[1:]:
                word = subwordvocab.tokenizer.convert_tokens_to_string(r[0])
                word = re.sub(r" +","",word)
                # print(word)
                if word == "[SEP]":
                    break
                if len(r[0]) == len(r[1]):
                    biocheck = defaultdict(int)
                    # print(sent,r)
                    # [['▁'], ['O']]
                    if r[0][0] == "▁":
                        # print(r)
                        # exit()
                        continue
                    elif r[0][0] == "+":
                        # print(r[0])
                        # exit()
                        restmp.append(["+","+"])
                        # sent.popleft()
                        continue
                    
                    failflag = False
                    for idx,(t,p) in enumerate(zip(r[0],r[1])):
                        # if t == "[SEP]":
                            # break 
                        # print(p)
                        if p.startswith("B_") and idx == 0:
                            # biocheck[p[2:]] += 1
                            if p != "+" and p != "O":
                                biocheck[p[2:]] += 1
                            else:
                                biocheck["Fail"] += 1
                        # elif p =="+":
                        #     restmp.append(["+","+"])
                        #     failflag = True
                        #     break
                        elif idx == 0 and not p.startswith("B_"):
                            # restmp.append([word,"Fail"])
                            # biocheck = {}
                            failflag = True
                            # print(444,t,p)
                            if p != "+" and p != "O":
                                biocheck[p[2:]] += 1
                            else:
                                biocheck["Fail"] += 1
                            # print(p)
                            # break
                        elif idx > 0 and not p.startswith("I_"):
                            # restmp.append([word,"Fail"])
                            if p != "+" and p != "O":
                                biocheck[p[2:]] += 1
                            else:
                                biocheck["Fail"] += 1
                            # print(p)
                            failflag = True
                            # break
                        elif idx > 0 and p.startswith("I_"):
                            # biocheck[p[2:]] += 1
                            if p != "+" and p != "O":
                                biocheck[p[2:]] += 1
                            else:
                                biocheck["Fail"] += 1
                    
                    if True:#not failflag:    
                        # print(biocheck)      
                        if len(biocheck) == 1:
                            for k,v in biocheck.items():
                                if v == len(r[0]):
                                    restmp.append([word,k])
                        else:
                            # print(133533,r,biocheck)
                            restmp.append([word,list(biocheck.keys())[0]])
                else:
                    print(2444434,r)
                    restmp.append([word,"Fail"])
            # print(restmp)
            # exit()
            htjoin = []
            # print(restmp)
            for idx,rr in enumerate(restmp):
                # print(rr)
                # print(1,sent)
                # print(2,restmp)
                if rr[0] == "+":
                    htjoin.append("+")
                    # continue
                else:
                    if "[UNK]" in rr[0]:
                        # print(idx,restmp[idx],sent.replace("+"," + ").split()[idx])
                        sent = sent.strip()
                        # sent = normalize(sent)
                        # sent = sent.replace(" + ","+").replace(" +","+").replace("+ ","+")
                        
                        sent_ = sent.split()
                        for i in range(len(sent_)):
                            sent_[i] = sent_[i].strip("+")
                        # print(sent)  
                        sent_ = (" ".join(sent_)).replace("+"," + ").split()
                       
                        try:
                            rr[0] = sent_[idx]
                        except Exception as ex:
                            import traceback
                            traceback.print_exception()
                            exit()
                    
                    if re.search(r"[a-z|A-Z]",rr[0]) and not re.search(r"[^a-z|A-Z]",rr[0]):
                        rr[1] = "SL"
                    # print(rr)
                    ht = "/".join(rr)
                    ht = get_xstags(ht)
                    htjoin.append(ht)
            
            # print(htjoin)
            restmp = " ".join(htjoin)
            restmp = re.sub(r" +"," ",restmp)
            # restmp = re.sub(r"\d+\s\.\d+",r"\d+\.\d+"
            restmp = restmp.replace(" + ","+")

            Y.append(restmp)
        return Y

def infer_onnx(subwordvocab:HTInputTokenizer,onnx,x,device):
    with torch.no_grad():
        xs = []
        for idx,xx in enumerate(x):
            xx = xx.strip().strip("+").strip()
            xs.append(xx)
        X = subwordvocab.make_intoken(xs)#subwordvocab.tokenizer(xs,padding="longest",max_length=600,is_split_into_words=True,return_offsets_mapping=True,return_tensors="pt")
        Y = []
        
        X["input_ids"]= X["input_ids"].numpy().astype(np.float32)#to(device)
        # tmp = torch.tensor([[0]]).numpy
        
        input_name = onnx.get_inputs()[0].name
        output_name = onnx.get_outputs()[0].name
        
        y = onnx.run([output_name], {input_name: X["input_ids"]})
        # print(y[0].shape)
        # y = np.array(y[0])
        # y = model(X["input_ids"])
        y = np.argmax(y[0],axis = -1)
        # print(y.shape)
        for sent,xx,mapping,yy in zip(xs,X["input_ids"],X["offset_mapping"],y):
            xxtmp = copy.deepcopy(xx)
            yytmp = copy.deepcopy(yy)
            res = []
            xy = xx[xx != subwordvocab.pad]
            xx = xx[:xy.shape[0]]#.numpy()#.cpu().numpy().tolist()
            yy = yy[:xy.shape[0]]#.numpy()#.cpu().numpy().tolist()
            xx = deque(xx)
            yy = deque(yy)
            tmp = []
            ptmp = []
            c = 0
            for offset in mapping:
                if offset[0] == 0 and c!= 0:
                    if tmp[-1] == subwordvocab.space and len(tmp) == 1:
                        res.append([subwordvocab.to_mp([tmp])[0],subwordvocab.to_tag([ptmp])[0]])
                        tmp = []
                        ptmp = []
                    else:
                        res.append([subwordvocab.to_mp([tmp])[0],subwordvocab.to_tag([ptmp])[0]])
                        tmp = []
                        ptmp = []
                if len(xx) == 0:
                    # print(res)
                    break
                    # exit()
                c += 1
                tk = xx.popleft()
                p = yy.popleft()
                tmp.append(tk)
                ptmp.append(p)
                if tk == subwordvocab.pad:
                    break
            restmp = []
            for r in res[1:]:
                word = subwordvocab.tokenizer.convert_tokens_to_string(r[0])
                word = re.sub(r" +","",word)
                if word == "[SEP]":
                    break
                if len(r[0]) == len(r[1]):
                    biocheck = defaultdict(int)
                    if r[0][0] == "▁":
                        continue
                    elif r[0][0] == "+":
                        restmp.append(["+","+"])
                        continue
                    
                    failflag = False
                    for idx,(t,p) in enumerate(zip(r[0],r[1])):
                        if p.startswith("B_") and idx == 0:
                            biocheck[p[2:]] += 1
                        elif idx == 0 and not p.startswith("B_"):
                            failflag = True
                            biocheck[p[2:]] += 1
                        elif idx > 0 and not p.startswith("I_"):
                            biocheck[p[2:]] += 1
                            failflag = True
                        elif idx > 0 and p.startswith("I_"):
                            biocheck[p[2:]] += 1
                    
                    if True:#not failflag:    
                        if len(biocheck) == 1:
                            for k,v in biocheck.items():
                                if v == len(r[0]):
                                    restmp.append([word,k])
                        else:
                            # print(133533,r,biocheck))
                            restmp.append([word,list(biocheck.keys())[0]])
                else:
                    print(2444434,r)
                    restmp.append([word,"Fail"])

            htjoin = []
            for idx,rr in enumerate(restmp):
                if rr[0] == "+":
                    htjoin.append("+")
                    # continue
                else:
                    if "[UNK]" in rr[0]:
                        sent = sent.strip()
                        
                        sent_ = sent.split()
                        for i in range(len(sent_)):
                            sent_[i] = sent_[i].strip("+")
                        # print(sent)  
                        sent_ = (" ".join(sent_)).replace("+"," + ").split()
                        try:
                            rr[0] = sent_[idx]
                        except Exception as ex:
                            import traceback
                            traceback.print_exception()
                            exit()
                   
                    htjoin.append("/".join(rr))
   
            restmp = " ".join(htjoin)
            restmp = re.sub(r" +"," ",restmp)
            restmp = restmp.replace(" + ","+")

            Y.append(restmp)
        return Y

if __name__ == "__main__":
    import pickle

    with open("postagger_model/subwordvocab.pkl","rb") as f:
        subwordvocab = pickle.load(f)
        text = "나는 (주) 통하보건교육실 소속이다 ."
        text = normalize(text)
        # print(text)
        text = text.replace("+"," + ").split()
        toks = subwordvocab.tokenizer([text],padding="longest",max_length=600,is_split_into_words=True,return_offsets_mapping=True,return_tensors="pt")
        
def beam_search_decode(model, input_seq, beam_size=3,subwordvocab=None,device=None):
    """
    모델과 입력 시퀀스를 받아서 beam search를 이용해 라벨 시퀀스 후보들을 반환합니다.
    
    매개변수:
      model      : 학습된 sequence labeling 모델
      input_seq  : 입력 시퀀스 tensor, shape (1, seq_len)
      beam_size  : 각 타임스텝에서 유지할 후보 수

    반환:
      beam: 최종 라벨 시퀀스 후보들의 리스트.
            각 요소는 (라벨 시퀀스, 누적 로그 확률) 형태의 튜플입니다.
    """
    tmp = torch.tensor([[0]]).to(device)
    # 모델 예측 (배치 사이즈=1)
    # logits = model(input_seq)            # shape: (1, seq_len, num_labels)
    logits = model(tmp,input_seq,tmp,tmp,subwordvocab=subwordvocab)
    # logits = logits.squeeze(0)           # shape: (seq_len, num_labels)
    
    # 안정적인 연산을 위해 log softmax 적용 (각 타임스텝에서 라벨 선택 확률)
    # log_probs = torch.log_softmax(logits, dim=-1)
    beams = []
    for log_probs in logits:
        # beam 초기 상태: 빈 시퀀스와 누적 로그 확률 0
        # print(log_probs.shape)
        beam = [ ([], 0.0) ]

        # 입력 시퀀스 길이만큼 반복하면서 beam search 진행
        for t in range(log_probs.size(0)):
            new_beam = []
            # 현재 beam의 각 후보에 대해 다음 단계 확률 확장
            for seq, cumulative_log_prob in beam:
                # t번째 타임스텝의 log probability (num_labels 길이 벡터)
                current_log_probs = log_probs[t]
                # 해당 타임스텝에서 상위 beam_size 개의 라벨 선택
                topk_log_probs, topk_indices = current_log_probs.topk(beam_size)
                for i in range(beam_size):
                    # 현재 후보 시퀀스에 선택된 라벨을 추가
                    next_seq = seq + [topk_indices[i].item()]
                    # 누적 로그 확률 업데이트 (log space에서 덧셈)
                    next_log_prob = cumulative_log_prob + topk_log_probs[i].item()
                    new_beam.append((next_seq, next_log_prob))
            # beam 확률이 높은 순서대로 정렬한 후 beam_size만큼 유지
            beam = sorted(new_beam, key=lambda x: x[1], reverse=True)[:beam_size]
        # print(beam[0][0])
        # print(subwordvocab.to_tag([beam[0][0]]))
        # exit()
        beams.append(beam[0][0])
    
    # print(subwordvocab.to_tag(beams))
    # exit()
    return beams

# 3. 예제 실행
if __name__ == '__main__':
    # 하이퍼파라미터 설정
    vocab_size = 1000     # 단어 집합 크기 (예시)
    embedding_dim = 64    # 임베딩 벡터 차원
    hidden_dim = 128      # LSTM 은닉 상태 차원
    num_labels = 10       # 가능한 라벨의 수 (예시)
    seq_len = 10          # 입력 시퀀스 길이
    
    # 모델 초기화
    model = SequenceLabelingModel(vocab_size, embedding_dim, hidden_dim, num_labels)
    
    # 임의의 입력 시퀀스 생성 (정수 인덱스 시퀀스)
    input_seq = torch.randint(0, vocab_size, (1, seq_len))  # shape: (1, seq_len)
    
    # Beam search 디코딩 수행
    beam_results = beam_search_decode(model, input_seq, beam_size=3)
    
    # 결과 출력
    print("입력 시퀀스:", input_seq)
    print("Beam Search 결과 (라벨 시퀀스, 누적 로그 확률):")
    for seq, log_prob in beam_results:
        print(seq, log_prob)
import random
import os
import sys

from tqdm import tqdm
from collections import OrderedDict

def duplicate_tag_remove(t):
    prev = -1
    res = []
    for tt in t.split("_"):
        if prev != tt:
            tt = tt.split("-")[0]
            res.append(tt)
        prev = tt
    return "_".join(res)


    
    # exit()
datapath = os.path.join(os.environ["DATASET"],"ht_dataset","modu")
    
def make_pair(start,end,seeks,num,htsep,tagsep):
    datapath = os.path.join(os.environ["DATASET"],"ht_dataset","modu","result")
    
    shuffle = open(os.path.join(datapath,"headtail_train_shuffle.txt"),encoding="utf-8")
    shuffle.seek(seeks[start])
    with open(os.path.join(datapath,f"kiwimorphs_{num}.txt"),"w",encoding="utf-8") as kmp, open(os.path.join(datapath,f"kiwitags_{num}.txt"),"w",encoding="utf-8") as ktg:
        cnt = 0
        for i in range(start,end+1):
            cnt += 1
            # if cnt % 50000 == 0:
            sys.stdout.write("\r"+str(cnt)+"/"+str(end-start))
            l = shuffle.readline()
            l = l.strip()
            if l == "":
                # print(11111,l,22222)
                # break
                continue
            mp = []
            tg = []
            for ll in l.split():
                if htsep in ll:
                    mp_ = []
                    tg_ = []
                    # ll = ll.replace("+/SW","``/SW")
                    for lll in ll.split(htsep):
                        # print(lll)
                        
                        try:
                            # lll = lll.replace("``/SW","+/SW")
                            m = lll.split(tagsep)[0]
                            t = lll.split(tagsep)[1]
                        except:
                            # if re.search
                            print(ll)
                            print("'"+l+"'")
                            print(lll)
                            # exit()
                        if "_" in t:
                            t = duplicate_tag_remove(t)
                            # print(t)
                        mp_.append(m)
                        tg_.append(t)
                    
                    mp.append(htsep.join(mp_))
                    tg.append(htsep.join(tg_))
                
                else:
                    mp.append(ll.split(tagsep)[0])
                    t = ll.split(tagsep)[1]
                    
                    t = duplicate_tag_remove(t)
                
                    tg.append(t)

            mps = " ".join(mp)
            tgs = " ".join(tg)
            
            kmp.write(mps+"\n")
            ktg.write(tgs+"\n")

        shuffle.close()
        
import multiprocessing
if __name__ == "__main__":
    tagsep = "@@@"
    htsep = "hththththt"
    
    htdataset = os.path.join(os.environ["DATASET"],"ht_dataset","modu","train","headtail.txt")#headtail_train_mp.txt"
    # htdataset = "noise.txt"
    with open(htdataset,encoding="utf-8",buffering=1) as f:
        # sys.stdin = f
        # f = sys.stdin
        count = 0#11878186
        seeks = {}
        seeks[count] = 0
        # for l in tqdm():
        # line = f.readline()
        # pbar = tqdm(total=count)
        # tmp = open(f"{os.environ['DATASET']}\\dataset\\ht_dataset\\modu\\tmp.txt","w",encoding="utf-8")
        # while line:
        for line in f:
            # count += 1
            sys.stdout.write(f"\r{count}")
            # seeks[count] = f.tell()
            line = line.strip()
            error=True
            
            for ll in line.split():
                # if ll.count("+") >= 2 or "+/" in ll:
                #     error = True
                #     break
                    # flag = True
                for lll in ll.split(htsep):
                    # if lll == "" and ll.count("+") >= 2:
                    #     continue
                    try:
                        tags = lll.split(tagsep)[1]
                    except Exception as ex:
                        print(ex)
                        print(ll)
                        print(lll)
                        error = True
                        
                    # print(tags)
                    # if ("NNG" in tags or "NNP" in tags) and ("_J" in tags or tags[0] == "J" or "_V" in tags or tags[0] == "V" or "_E" in tags or tags[0] == "E"):
                    #     error = True
                    # if ("NNG" in tags or "NNP" in tags) and ():
                        # print(lll)
                        # error = True
            # if not error:
            #     # print(line)
            count += 1
            #     tmp.write(line+"\n")
            # print(line)
            # exit()
            # pbar.update(1)
        # tmp.close()
        
        f.seek(0)
        lines = f.readlines()
        count = len(lines)
        rindex = [i for i in range(count)]
        random.shuffle(rindex)
        # f.seek(0)
        
        # with open(f"{os.environ['DATASET']}\\tmp.txt",encoding="utf-8") as tmp:
            # lines = tmp.readlines()
        datapath = os.path.join(os.environ["DATASET"],"ht_dataset","modu","result")
    
        shuffle = open(os.path.join(datapath,"headtail_train_shuffle.txt"),"w",encoding="utf-8",buffering=1)
        seeks_tmp = {}
        seeks_tmp[0] = 0
        idx = 0
        for cnt in tqdm(rindex):
            # f.seek(seeks[cnt])
            # f.seek(seeks[cnt])
            # l = f.readline()
            # print(cnt,len(lines))
            l = lines[cnt]
            lines[cnt] = ""
                        
            
            shuffle.write(l.strip()+"\n")
            seeks_tmp[idx] = shuffle.tell()
            idx+=1
            # else:
            #     shuffle.write("error"+l.strip()+"\n")
            #     seeks_tmp[idx] = shuffle.tell()
        seeks = seeks_tmp   
        # pbar = tqdm(total=count)
        shuffle.close()
    split_size = 16
    count_split = count // split_size
    print(count,split_size)
    pss = []
    cnt = 0
    for i in range(0,count,count_split):
        ps = multiprocessing.Process(target=make_pair,args=(i,i+count_split,seeks,cnt,htsep,tagsep))
        ps.start()
        pss.append(ps)
        cnt += 1
    
    for ps in pss:
        ps.join()
    
    resx = open(os.path.join(datapath,"kiwimorphs.txt"),"w",encoding="utf-8")
    resy = open(os.path.join(datapath,"kiwitags.txt"),"w",encoding="utf-8")
    for dir in os.listdir(datapath):
        tmppath = os.path.join(datapath,dir)
        if dir.startswith("kiwimorphs_"):
            with open(tmppath,encoding="utf-8") as f:
                for l in f:
                    resx.write(l)
            os.remove(tmppath)
        elif dir.startswith("kiwitags_"):
            with open(tmppath,encoding="utf-8") as f:
                for l in f:
                    resy.write(l)
            os.remove(tmppath)
    resx.close()
    resy.close()

    from normalize_line import noramlize_line
    
    # exit()
    import pandas as pd
    from customutils import get_mp_tags
    from collections import defaultdict
    
    resx = open(os.path.join(datapath,"headtail_train_shuffle.txt"),encoding="utf-8")
    # resy = open(os.path.join(datapath,"kiwitags.txt"),encoding="utf-8")
    
    counttag = defaultdict(int)
    countwords = defaultdict(set)
    countwordsht = defaultdict(set)
    countlines = defaultdict(set)
    for idx,tks in enumerate(resx):
        for tk in tks.split():
            htsep = "hththththt"
            tagsep = "@@@"
            
            tk = tk.replace("+","_").replace("/","_")
            tk = tk.replace(htsep,"+")
            tk = tk.replace(tagsep,"/")
            
            head,tail = get_mp_tags(tk)
            
            # if len(tail) > 0:
            #     counttag[head[1]] += 1
            #     countwords[head[1]].add((head[0]+tail[0],head[0]+"+"+tail[0]))
            #     # countwordsht[head[1]+"+"+tail[1]].add(head[0]+"+"+tail[0])
            #     countlines[head[1]].add(idx)
            # else:

            
            
            if len(tail) > 0:
                counttag[tail[1]] += 1
                countwords[tail[1]].add((head[0]+tail[0],head[0]+"+"+tail[0],head[1]+"+"+tail[1]))
                countlines[tail[1]].add(idx)
                
                counttag[head[1]] += 1
                countwords[head[1]].add((head[0]+tail[0],head[0]+"+"+tail[0],head[1]+"+"+tail[1]))
                # countwordsht[head[1]].add([head[0])
                countlines[head[1]].add(idx)
            else:
                counttag[head[1]] += 1
                countwords[head[1]].add((head[0],head[0],head[1]))
                # countwordsht[head[1]].add([head[0])
                countlines[head[1]].add(idx)
    resx.close()
    dataframe = defaultdict(list)
    # from autospace import AutospaceTrainer,Auto2
    # auto = Auto2()
    res = set()
    for k,v in counttag.items():
        if v <= 5:
            words = list(countwords[k])
            lines = list(countlines[k])
            ht = list(countwordsht[k])
            res.update(lines)
            for w in words:
                lines = list(map(str,lines))
                dataframe["ht"].append(w[1])
                dataframe["httag"].append(w[2])
                dataframe["word"].append(w[0])
                autoword = w[0]#auto.autospace(w[0])#.strip()
                dataframe["autospace"].append(autoword)
                # print("dfaf"+autoword+"adfasdf"," " in autoword,w[0])
                if " " in autoword:
                    dataframe["is_auto"].append("Y")
                else:
                    dataframe["is_auto"].append("N")
                dataframe["tag"].append(k)
                dataframe["error_idx"].append(",".join(lines))
                dataframe["count"].append(v)
    
    resx.close()
    resy.close()
    # resx.seek(0)
    # resx = open(os.path.join(datapath,"headtail_train_shuffle.txt"),encoding="utf-8")
    
    # noise = open("noise.txt","w",encoding="utf-8")
    resx = open(os.path.join(datapath,"kiwimorphs.txt"),encoding="utf-8")
    resy = open(os.path.join(datapath,"kiwitags.txt"),encoding="utf-8")
    
    delresx = open(os.path.join(datapath,"delkiwimorphs.txt"),"w",encoding="utf-8")
    delresy = open(os.path.join(datapath,"delkiwitags.txt"),"w",encoding="utf-8")
    
    for idx,(x,y) in enumerate(zip(resx,resy)):
        if idx not in res:
            if x.strip() != "":
                if len(x.split()) > 3:
                    delresx.write(x.strip()+"\n")
                    delresy.write(y.strip()+"\n")
                # noise.write(l.strip()+"\n")
    resx.close()
    resy.close()
    delresx.close()
    delresy.close()
    noramlize_line(datapath)
    # noise.close()
    df = pd.DataFrame(dataframe)
    df.to_csv("errortag.csv",encoding="utf-8-sig")
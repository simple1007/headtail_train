from ht_utils import normalize, removesym, is_sym
from collections import defaultdict
import re
import os
import sys
from jamo import h2j, j2hcj
from kiwipiepy import Kiwi
from tqdm import tqdm
import re
import pandas as pd
from collections import defaultdict
# from autospace import AutospaceTrainer
# from autospace import Auto2
from kiwipiepy import Kiwi
from customutils import get_mp_tags,duple_tag,make_head_tail
    
head = defaultdict(int)
tail = defaultdict(int)
tagsep = "@@@"
htsep = "hththththt"
tokenslist = ["했","하","해","함","됨","됌","한","할","된","되","될","됐","됬","돼","됄"]
def duplicate_tag_remove(t):
    prev = -1
    res = []
    for tt in t.split("_"):
        if prev != tt:
            tt = tt.split("-")[0]
            res.append(tt)
        prev = tt
    return "_".join(res)

makemorphs = False
makeheadtail = True
if makemorphs:
    def make_mp(fn,kiwires,auto):
        
        kiwires = open(kiwires,"w",encoding="utf-8")
        kiwi = Kiwi()
        try:
            f = open(fn,encoding="utf-8")
        except:
            import traceback
            import traceback; traceback.print_exc();
            exit()
        # print(cnt)
        if True:
            cnt = 0
            try:
                for l in f:
                    cnt += 1
            except:
                print("error:",fn)
                exit()
            kiwires.write("--------------------------------------------------\n")
            # print(cnt)         
            f.seek(0)
            # with tqdm(total=cnt) as pbar:
            lcount = 0
            tmpcnt = 0
            # pbar = tqdm()
            for l in f:
                
                l = l.strip("#").strip()
                l = auto.autospace(l)
                l = normalize(l)
                l = removesym(l)
                # l = re.sub(r"([^가-힣|ㄱ-ㅎ|a-z|A-Z|\d+\.\d+|\d+]+)", r" \1 ",l)
                
                # l = re.sub(r"(\d+\.\d+|\d+)",r" \1 ",l)
                # l = re.sub(r"([a-z|A-Z]+)",r" \1 ",l)
                # # l = re.sub(r"(\.|\,|\?|\!|\~|\#|\@|\$|\%|\^|\&|\*|\(|\)|\=|\+|\-|\_|\/|\>|\<|\;|\:|\'|\"|\]|\[|\{|\}|\`|\|\\)",r" \1 ",l)
                # l = re.sub(r" +",r" ",l)
                # l = re.sub(r"(\d+) \. (\d+)",r" \1.\2 ",l)
                # l = re.sub(r" +",r" ",l)
                # l = l.strip()
                
                # if "3만원" in l:
                #     print(l)
                #     exit()
                # print(l)
                # break
                # ltmp 
                # kiwires.write(l+"\n")
                
                # tk = kiwi.tokenize(l)
                # print(l)
                
                no_tag = False
                sents = kiwi.split_into_sents(l,return_tokens=True,compatible_jamo=True)
                # print(sents)
                # sents = kiwi.split_into_sents("지금하기엔",return_tokens=True,compatible_jamo=True)
                # print(sents)
                # exit()
                res = []
                flag = False
                for tk in sents:
                    flag = False
                    # print(tk)
                    for ll in tk.tokens:
                        
                        if " " in ll.form:
                            # l = l.replace(ll.form,ll.form.replace(" ","_"))
                            res = []
                            flag = True
                            break
                            # continue  
                        # else:
                        # print(ll)
                        if ll.tag in ["UN","W_URL","W_EMAIL","W_HASHTAG","W_MENTION","W_SERIAL","W_EMOJI","Z_CODA","Z_SIOT","USER0~4"]:
                            no_tag = True
                        res.append(ll.form + f" {tagsep} " + ll.tag.split("-")[0] +"\t"+str(ll.word_position)+'\n')
                        # kiwires.write(ll.form.replace(" ","_") + " / " + ll.tag +"\t"+str(ll.word_position)+'\n')
                    if flag:
                        # kiwires.write("--------------------------------------------------\n")
                        continue
                    if not no_tag:
                        kiwires.write(tk.text+"\n")
                        kiwires.writelines(res)
                        kiwires.write("--------------------------------------------------\n")
                    res = []
                    
                # kiwires.write("--------------------------------------------------\n")
                
                lcount += 1
                # pbar.update(1)
                if flag:
                    res = []
                    continue
                tmpcnt += 1
                if tmpcnt % 5000 == 0:
                    sys.stdout.write(fn+"="+str(tmpcnt)+"/"+str(cnt)+"\n")
                # if lcount == 1000:
                #     break
        f.close()
    import threading
    import multiprocessing
    
    
    # for i in range(13):
    def thread_run(modu,files):
        auto = Auto2()
        for m,f in zip(modu,files):
            
            make_mp(m,f,auto)
            newf = f.split(os.sep)
            newf = os.sep.join(newf[:-1])+"/linefile/"+newf[-1]
            if os.path.exists(newf):
                os.remove(newf)
            os.rename(f,newf)
            # f.close()    
    if __name__ == "__main__":    
        datapath = os.environ["Desktop"] + os.sep + "dataset"# + os.sep + "ht_dataset"
        kcc = f"{datapath}\\KCC150_Korean_sentences_UTF8.txt"
        htpath = os.environ["Desktop"] + os.sep + "dataset" + os.sep + "ht_dataset"#"kjm\\headtail\\split"
        # modu = f"{datapath}\\형태분석_모두의말뭉치\\NIKL_MP_CSV_형태분석\\modu_mp.txt"
        modu = []
        files = []
        for i in range(12):
            modu.append(f"{htpath}\\ht_{i:02d}")
            files.append(f"{htpath}\\ht_{i}.txt")
        
        modu.append(f"{htpath}\\SXMP1902008031_ht.txt")
        files.append(f"{htpath}\\ht_SXMP1902008031_ht.txt")
        
        modu.append(f"{htpath}\\NXMP1902008040_ht.txt")
        files.append(f"{htpath}\\ht_NXMP1902008040_ht.txt")
        # modu.append(f"{datapath}\\형태분석_모두의말뭉치\\NIKL_MP_CSV_형태분석\\modu_mp.txt") 
        # files.append(f"{htpath}\\kiwires_{i+1}.txt")
        
        ths = []
        for m,f in zip(modu,files):
            th1 = multiprocessing.Process(target=thread_run,args=([m],[f]),daemon=True)
            # th1 = threading.Thread(target=thread_run,args=([m],[f]))
            # th1.daemon = True
            # th1.start()
            # th1.join()
            # sys.stdout.write(m+"\n")
            ths.append(th1)
            
        # while True:
        #     cnt = 0
            # if len(ths) == 3:
            #     for th in ths:
            #         # if th.is_alive():
            #             # print(cnt)
            #         th.join()
            #     ths = []
                    
        if len(ths) > 0:
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            ths = []
        # for th in ths:
        #     th.daemon=True
        
        # th1 = threading.Thread(target=thread_run,args=(modu[:3],files[:3]))
        # # th1.daemon = True
        # th1.start()
        
        # th2 = threading.Thread(target=thread_run,args=(modu[3:6],files[3:6]))
        # # th2.daemon = True
        # th2.start()
        
        # th3 = threading.Thread(target=thread_run,args=(modu[6:9],files[6:9]))
        # # th3.daemon = True
        # th3.start()
        
        
        # th4 = threading.Thread(target=thread_run,args=(modu[9:],files[9:]))
        # # th3.daemon = True
        # th4.start()
        
        # for th in [th1,th2,th3,th4]:
        #     th.join()   
        # ths = [th1,th2,th3]
        # for i in range(ths):
        #     th1 = threading.Thread(target=thread_run,args=(modu[6:8],files[6:8]))
        #     # th1.daemon = True
        #     th1.start()
            
        #     th2 = threading.Thread(target=thread_run,args=(modu[8:10],files[8:10]))
        #     # th2.daemon = True
        #     th2.start()
            
        #     th3 = threading.Thread(target=thread_run,args=(modu[10:],files[10:]))
        #     # th3.daemon = True
            #     th3.start()
            # print(modu)
            # # th1 = threading
            # kiwires = open(f"{datapath}\\kiwiresult.txt","w",encoding="utf-8")
            # make_mp(kcc,kiwires)
            # kiwires.close() 
        
            # kiwires = open(f"{datapath}\\kiwiresult2.txt","w",encoding="utf-8")
            # make_mp(modu,kiwires)
            # kiwires.close()
        exit()
# exit()
from tqdm import tqdm
# exit()


def make_ht_train(fn,result_ht_dict,result_ht):
    # global errors
    xtag = defaultdict(list)
    symcount = 0
    symlength = 0
    errors = 0
    # datapath = os.environ["Desktop"] + os.sep + "dataset\\ht_dataset"
    result_ht_dict_f = open(result_ht_dict,"w",encoding="utf-8")
    result_ht = open(result_ht,"w",encoding="utf-8")
    ttt = []
    sam = []
    number = 0
    # if "SX" in fn:
        # errorf = open("SX.txt","w",encoding="utf-8")
    # elif "NX" in fn:
        # errorf = open("NX.txt","w",encoding="utf-8")
    # else:
    errorf = open("error_head_tail.txt","a+",encoding="utf-8")
    head_tail_orif = open("list.txt","w",encoding="utf-8")
    import pickle
    with open("error_line.pkl","rb") as f:
        error = pickle.load(f)
    with open(fn,encoding="utf-8") as kiwires:
        count = 0
        lt = ""
        for i in kiwires:
            count += 1
        # pbar = tqdm(total=count)
        kiwires.seek(0)
        tmp = []
        cnt = 0
        line = ""
        def tail_tag(tail,xsn=None):
            tail_t = ""
            for idx,t in enumerate(tail):
                # print(1,tail,t)
                if xsn != None and idx == 0:
                    continue
                tail_ = t.split(f" {tagsep} ")
                # tail += tail_[0]
                tail_t += tail_[1] + "_"
            # if xsn != None:
                # tail_t = xsn + "_" + tail_t
            return tail_t
        def token_to_str(h):
            return htsep.join(h).replace(f" {tagsep} ",f" {tagsep} ")
        eojid = 0
        prev = -1
        flag = False
        linecnt = 0
        for idx,l in enumerate(kiwires):
            # tmp.append()
            # cnt = 0
            # print(l)
            def head_tail_1chk(head,tailtk,tail):
                
                if 'ㄱ' <= tail[0] <= 'ㅎ':
                    tail_ = tail#[1:]
                else:
                    tail_ = tail
                jamostr = j2hcj(h2j(tailtk[0]))
                tailjamo = j2hcj(h2j(tail_[0]))
                if  jamostr[0] != tailjamo[0]:
                    #print(head,jamostr,tail,tailjamo)
                    return ""#head[-1]
                return ""
            if l.startswith("--------"):
                number += 1
                headtail_tk = ""
                prev = -1
                xsindex = -1
                # if "xx" in lt:
                # if len(tmp) != len(sam)-1:
                #     errorf.write(str(idx)+"\n")
                #     errorf.write(str(tmp)+"\n")
                #     errorf.write("".join(sam)+"\n")
                #     # cnt = 0
                #     # tmp = []
                #     tmp = []
                #     cnt = 0
                #     errors += 1
                #     sam=[]
                #     print(errors)
                #     continue
                # numflag = False
                # for hhh in tmp:
                #     for hhhh in hhh[1]:
                #         if re.search(f"\d+",hhhh) and f" {tagsep} S" not in hhhh:
                #             numflag = True
                #             break
                # if numflag or "xx" in lt: 
                #     errorf.write(str(idx)+"\n")
                #     errorf.write(str(tmp)+"\n")
                #     errorf.write("".join(sam)+"\n")
                #     # cnt = 0
                #     # tmp = []
                #     errors += 1
                #     sam = []
                #     tmp = []
                #     cnt = 0
                    
                #     continue
                # if str(idx) in error:
                #     errorf.write(str(idx)+"\n")
                #     errorf.write(str(tmp)+"\n")
                #     errorf.write("".join(sam)+"\n")
                #     cnt = 0
                #     tmp = []
                #     continue
                sam = [] 
                try:
                    # print(tmp)
                    # print("\n",99,tmp)
                    # print(lt)
                    words, morphs = [],[]
                    for ttmp in tmp:
                        words.append(ttmp[0])
                        morphs.append(ttmp[1])
                        
                    htsentence = make_head_tail(words,morphs)   
                    result_ht.write(htsentence.strip()+"\n")
                    cnt = 0
                    prev = 0
                    flag = False
                    if False:
                        for i,h in enumerate(tmp):
                            head = ""
                            head_t = ""
                            tail = ""
                            tail_t = ""
                
                            char2 = j2hcj(token_to_str(h[1]))
                            char2 = char2.replace(f" {tagsep} ",f"{tagsep}")
                            
                            if False:
                                for idx,hh in enumerate(h[1]):
                                    if f" {tagsep} XSA" in hh or f" {tagsep} XSV" in hh:
                                        xsindex = idx
                            xsindex=-1
                            nounindex = -1
                            
                            nflag = ""
                            nouncount = 0
                            # print()
                            for nidx,hh in enumerate(h[1]):
                                if f" {tagsep} XR" in hh or f" {tagsep} NR" in hh or f" {tagsep} MAG" in hh or f" {tagsep} MM" in hh or f" {tagsep} XSN" in hh or f" {tagsep} XPN" in hh or f" {tagsep} NNB" in hh or f" {tagsep} NNG" in hh or f" {tagsep} NNP" in hh or f" {tagsep} NP" in hh:# or f" {tagsep} XSN" in hh:
                                # if ((f" {tagsep} N" in nflag and f" {tagsep} NA" not in nflag and f"{tagsep} NNB" not in nflag and f"{tagsep} NR" not in nflag) and (f" {tagsep} VV" in hh  or f" {tagsep} VA" in hh )) or f" {tagsep} XS" in hh or (f" {tagsep} N" in hh and f" {tagsep} NA" not in hh and f" {tagsep} NNB" not in hh and f" {tagsep} NR" not in hh):# or f" {tagsep} NNP" in hh or f" {tagsep} NP" in hh:
                                    # print("에츄")1
                                    
                                    
                                    nounindex = nidx
                                    nouncount += 1
                                    # if nidx != len(h[
                                        # 1]) -1:
                                        
                                            
                                    #     if f" {tagsep} XR" not in h[1][nidx+1] and f" {tagsep} NR" not in h[1][nidx+1] and f" {tagsep} MM" not in h[1][nidx+1] and f" {tagsep} MAG" not in h[1][nidx+1] and f" {tagsep} XPN" not in h[1][nidx+1] and f" {tagsep} NNB" not in h[1][nidx+1] and f" {tagsep} XSN" not in h[1][nidx+1] and f" {tagsep} NNG" not in h[1][nidx+1] and f" {tagsep} NNP" not in h[1][nidx+1] and f" {tagsep} NP" not in h[1][nidx+1] and f" {tagsep} XSV" not in h[1][nidx+1] and  f" {tagsep} XSA" not in h[1][nidx+1] and f" {tagsep} VV" not in h[1][nidx+1]  and f" {tagsep} VA" not in h[1][nidx+1]:
                                    # #         # nounind -= 1
                                    #         # print(hh,nounindex,h[:nounindex],end=" ")
                                    #         break
                                        # else:
                                        #     nounindex = nidx
                                        #     nouncount += 1
                                        #     # nounindex = nidx
                                        #     break
                                    # else:
                                    #     nounindex = nidx
                                    #     nouncount += 1
                                    #     break
                                    # else:
                                    # elif nidx == len(h[1])-2:
                                    #     break

                                    # else: 
                                    # else:
                                    #     nounindex = nidx
                                    #     nouncount += 1
                                    # print(hh,end = " ")
                                    # if f" {tagsep} XS" in hh:
                                    #     # print(hh)
                                    #     xsindex = 100
                                    #     # print(h[1][:nounindex+1])
                                    #     # print()
                                    #     # exit()
                                    
                                    # if f" {tagsep} XS" in hh or f" {tagsep} VV" in hh or f" {tagsep} VA":
                                    #     # print(hh)
                                    #     # exit()
                                    #     break
                                elif f" {tagsep} XSA" in hh or f" {tagsep} XSV":
                                    nouncount += 1
                                    nounindex = nidx
                                    # break
                                elif f" {tagsep} XR" in nflag or f" {tagsep} NR" in nflag or f" {tagsep} MM" in nflag or f" {tagsep} MAG" in nflag or f" {tagsep} XPN" in nflag or f" {tagsep} NNB" in nflag or f" {tagsep} NNG" in nflag or f" {tagsep} NNP" in nflag or f" {tagsep} NP" in nflag or f" {tagsep} XSN" in nflag:
                                    if f" {tagsep} VV" in hh or f" {tagsep} VA":
                                        nounindex = nidx
                                        nouncount += 1
                                        # break
                                # else:
                                #     break
                                nflag = hh
                                # elif nounindex > -1 and f" {tagsep} NNB" in hh:
                                #     nounindex = nidx
                                
                            # if nounindex > -1 and nounindex+1 < len(h[1]) and (nounindex > 0 and f" {tagsep} NNB" in h[1][nounindex+1]):
                            #     nounindex += 1
                            # elif nounindex == -1 and nounindex+1 < len(h[1]) and (nounindex > 0 and f" {tagsep} NNB" in h[1][nounindex+1]):
                            #     nounindex += 1
                            #     if nounindex > -1 and nounindex+1 < len(h[1]) and (nounindex > 0 and f" {tagsep} NNB" in h[1][nounindex+1]):
                            #         nounindex += 1
                            # elif len(h[1]) >= 2 and f" {tagsep} NR" and h[1][0] and f" {tagsep} NNB" in h[1][1]:
                            #     nounindex = 1
                            # if nounindex > -1:
                            #     print(2222,nounindex,h[1])
                            # print(h[0])
                            if False and removesym(h[0]).strip() == "♥":
                                # if len(h[1]) > 1:
                                #     print(h)
                                print(9,h)
                                # exit()
                                h[0] = removesym(h[0]).strip()
                                symb = h[1][0]
                                symb = symb.split(f" {tagsep} ")
                                h[1][0] = removesym(symb[0]).strip() + f" {tagsep} " + symb[1]
                                # print(h)
                                # exit()
                                symcount += len(h[1])
                                symlength += 1
                            # #     # continue
                            #     print(tmp)
                            #     print(1111,h)
                            #     h[0] = "○"
                            # print(h)
                            # print(lt)
                            # if nouncount == 1:
                            #     nounindex = -1
                            if nounindex > -1:#xsindex == -1 and (f" {tagsep} MM" in h[1][0] or f" {tagsep} XPN" in h[1][0] or f" {tagsep} NNG" in h[1][0] or f" {tagsep} NNP" in h[1][0] or f" {tagsep} XSN" in h[1][0] or f" {tagsep} N" in h[1][0]):
                                cnt_eojid = 0
                                # print(h)
                                if False:
                                    for hh in h[1]:
                                        # if  f" {tagsep} XPN" not in hh and f" {tagsep} NNG" not in hh and f" {tagsep} NNP" not in hh and f" {tagsep} N" not in hh and f" {tagsep} XSN" not in hh and f" {tagsep} MM" not in hh:# and " / XSN" not in hh:
                                        if  f" {tagsep} XPN" not in hh and f" {tagsep} NNG" not in hh and f" {tagsep} NNP" not in hh and f" {tagsep} N" not in hh and f" {tagsep} MM" not in hh:# and " / XSN" not in hh:
                                            break
                                        # if f" {tagsep} XSN" in hh or f" {tagsep} XPN" in hh or f" {tagsep} NNG" in hh or f" {tagsep} NNP" in hh  or f" {tagsep} N" in hh or f" {tagsep} XSN" in hh or f" {tagsep} MM" in hh:# or " / XSN" in hh:
                                        if f" {tagsep} XPN" in hh or f" {tagsep} NNG" in hh or f" {tagsep} NNP" in hh  or f" {tagsep} N" in hh or f" {tagsep} MM" in hh:# or " / XSN" in hh:
                                            head += hh.split(f" {tagsep} ")[0]
                                            head_t += hh.split(f" {tagsep} ")[1] + "_"
                                            cnt_eojid += 1
                                # head = 
                                # if f h[1][nounindex+1]
                                
                                headtmp = h[1][:nounindex+1]
                                # if headtmp[0].startswith("후보가"):
                                #     print("\n",h,nounindex)
                                #     print(headtmp)
                                # print(2222,headtmp)
                                head = [ht.split(f" {tagsep} ")[0] for ht in headtmp]
                                head_t = [ht.split(f" {tagsep} ")[1] for ht in headtmp]
                                
                                head = "".join(head)
                                head_t = "_".join(head_t)
                                # if xsindex==100:
                                #     print(head,head_t)
                                    # exit()
                                # print(1,head,head_t)
                                # print(cnt_eojid,h[1])
                                if False and len(h[1])-1 > cnt_eojid:
                                    if False:#f" {tagsep} VV" in h[1][cnt_eojid] or f" {tagsep} VA" in h[1][cnt_eojid]:
                                        # print(1,head,head_t,tail,tail_t,cnt_eojid,h[1])
                                        vtag = h[1][cnt_eojid].split(f" {tagsep} ")
                                        # print(vtag)
                                        
                                        head_ = h[0][:len(head)] + "_" + h[0][len(head):len(head)+len(vtag[0])]
                                        # head = h[0][len(head)]
                                        tail = h[0][len(head)+len(vtag[0]):]
                                        head = head_
                                        
                                        head_t = head_t.strip("_") + "_" + vtag[1]
                                        # import time
                                        # print(head,head_t)
                                        # time.sleep(60)
                                        if len(head) != len(h[0]):
                                            tail_t = tail_tag(h[1][cnt_eojid:])
                                        else:
                                            tail_t = ""
                                        cnt_eojid += 1
                                        lh = h[1][cnt_eojid-1].split(f" {tagsep} ")[0]
                                        result_ht_dict_f.write(str(8)+"\t"+head+"/"+head_t.strip("_")+"@"+tail_t.strip("_")+"\t"+head[-len(lh):]+"\t"+h[1][cnt_eojid-1]+"\n")
                                        # if len(tail.strip()) > 0:
                                        tails = [tailtmp.split(f" {tagsep} ")[1] for tailtmp in h[1][cnt_eojid:]]
                                        
                                        tails = "_".join(tails)
                                        tail_ = [tailtmp.split(f" {tagsep} ")[0] for tailtmp in h[1][cnt_eojid:]]
                                        tail_ = "_".join(tail_)
                                        if tail.strip() != "":
                                            result_ht_dict_f.write(str(9)+"\t"+head+"/"+head_t.strip("_")+"@"+tail+"/"+tail_t.strip("_")+"\t"+tail_+"\t"+tails+"\n")
                                        
                                        # cnt_eojid += 1
                                        # print(23,h[1][cnt_eojid+1],cnt_eojid+1)
                                        # print(head,tail)
                                        # print(2,head,head_t,tail,tail_t)
                                        # import time
                                        # time.sleep(60)
                                    else:
                                        head = h[0][:len(head)]
                                        tail = h[0][len(head):]
                                else:
                                    head = h[0][:len(head)]
                                    tail = h[0][len(head):]
                                    # if head == "예정":
                                    #     print(tmp)
                                    #     print(h[0])
                                    #     print(h[1])
                                    #     print(head,tail)
                                    #     import time
                                    #     time.sleep(60)
                                # print(tail)
                                cnt_eojid = nounindex+1#nounindex + 1
                                # if head == "블레이클리가":
                                #     print("\n",h,nounindex,cnt_eojid,head,tail)
                                tail_t = tail_tag(h[1][cnt_eojid:])
                                if tail_t != "" and tail == "":
                                    head_t = head_t +"_"+tail_t
                                    tail_t = ""
                                # print(head,head_t,tail,tail_t,h,nounindex)
                                if head.strip() != "" and (head_t.startswith("XSA") or head_t.startswith("XSV") or head_t.startswith("VV") or head_t.startswith("VA") or head_t.startswith("VCP")):
                                    # if tail_t != "" and tail == "":
                                    #     head_t = head_t +"_"+tail_t
                                    #     print("\n"+str(10)+"\t"+head+"/"+head_t+"\t"+tail+"/"+tail_t+"\t"+" ".join(h[1])+"\t"+fn+"-"+str(idx))
                                    # else:#if tail_t != "" and tail == "":
                                        
                                    head_tail_orif.write("\n"+str(10)+"\t"+head+"/"+head_t+"\t"+tail+"/"+tail_t+"\t"+" ".join(h[1])+"\t"+fn+"-"+str(idx)+"\n")
                                        
                                if tail.strip() != "":
                                    # print(tmp)
                                    if True:
                                        head_tail_orif.write("\n"+str(8)+"\t"+head+"/"+head_t+"\t"+tail+"/"+tail_t+"\t"+" ".join(h[1])+"\t"+fn+"-"+str(idx)+"\n")
                                    elif len(h[1]) > 1 and f" {tagsep} X" in h[1][1]:
                                        tailtop = h[1][1].split(f" {tagsep} ")[0]
                                        xtag[tail[:len(tailtop)]].append(head+"/"+tail+"/"+"_".join(h[1]))
                                        # print("\n"+str(9)+"\t"+tail[:len(tailtop)]+"\t"+h[1][1]) 
                                    # print("2222",number)
                                if tail.strip() != "" and tail[0] != h[1][cnt_eojid][0]:
                                    char = j2hcj(f"{htsep}".join(h[1][cnt_eojid:]))
                                    char = char.replace(f" {tagsep} ",f"{tagsep}")
                                    if cnt_eojid < len(h[1]):
                                        char = head_tail_1chk(head,tail,h[1][cnt_eojid]) + char
                                    
                                    head_t_ = duplicate_tag_remove(head_t)
                                    # print(1,head+"/"+head_t,tail+"/"+tail_t,h[1])
                                    
                                    result_ht_dict_f.write(str(1)+"\t"+tail+"/"+head_t_.strip("_")+"@"+tail_t.strip("_")+"\t"+char+"\t"+char2+"\n")
                                    
                                if head.strip() != "" and head[-1] != h[1][cnt_eojid-1].split(f" {tagsep} ")[0][-1]:
                                    char = j2hcj(h[1][cnt_eojid-1])
                                    char = char.replace(f" {tagsep} ",f"{tagsep}")
                                    # print(3,head+"/"+head_t,tail+"/"+tail_t,h[1])
                                    if tail != "":
                                        head_t_ = duplicate_tag_remove(head_t)
                                        result_ht_dict_f.write(str(3)+"\t"+head+"/"+head_t.strip("_")+"@"+tail_t.strip("_")+"\t"+char+"\t"+char2+"\n")
                                    else:
                                        result_ht_dict_f.write(str(3)+"\t"+head+"/"+head_t+"\t"+char+"\t"+char2+"\n")
                            elif False and xsindex > -1:
                                # hmp = ""
                                # htag = []
                                head = ""
                                tail = ""
                                head_t = ""
                                tail_t = ""
                                
                                for htmp in h[1][:xsindex]:
                                    htmp= htmp.split(f" {tagsep} ")
                                    head += htmp[0]
                                    head_t += "_" + htmp[1]
                                headl = h[1][xsindex].split(f" {tagsep} ")
                                head_ = h[0][:len(head)] +"_" + h[0][len(head):len(head)+len(headl[0])]
                        
                                # if len(h[0]) > len(head):
                                tail = h[0][len(head)+len(headl[0]):]
                                
                                # print(h)
                                # print(head_,tail)
                                # import time
                                # time.sleep(60)
                                # print(h[0],tail,head)
                                # head_ = h[1][:xsindex+1]
                                # tail_ = h[1][xsindex+1:]
                                head = head_
                                # print(h[1][:len(hmp)+1],h[1][len(hmp)+1:])
                                # head_ [:xsindex+1]
                                # print(head_,tail)
                                head_t = duplicate_tag_remove(head_t).strip("_") + "_" + headl[1]
                                tails = [tailtmp.split(f" {tagsep} ")[1] for tailtmp in h[1][xsindex+1:]]
                                # print(head_t)
                                # print(tails)
                                # import time
                                # time.sleep(60)
                                if tail.strip() != "":
                                    tail_t = tail_tag(h[1][xsindex+1:])
                                else:
                                    for ts in tails:
                                        head_t += "_" + ts
                                    head_t = duplicate_tag_remove(head_t)
                                xsindex = -1
                                # print(head_t,tail_t)
                                # if tail.strip() != "":
                                #     print(str(8)+"\t"+head+"/"+head_t+"\t"+tail+"/"+tail_t+"\t"+h[1])
                                
                                lh = h[1][xsindex].split(f" {tagsep} ")[0]
                                
                                result_ht_dict_f.write(str(8)+"\t"+head+"/"+head_t.strip("_")+"@"+tail_t.strip("_")+"\t"+head[-len(lh):]+"\t"+h[1][xsindex]+"\n")
                                # if len(tail.strip()) > 0:
                                tails = "_".join(tails)
                                tail_ = [tailtmp.split(f" {tagsep} ")[0] for tailtmp in h[1][xsindex+1:]]
                                tail_ = "_".join(tail_)
                                if tail.strip() != "":
                                    # if False:
                                    result_ht_dict_f.write(str(9)+"\t"+head+"/"+head_t.strip("_")+"@"+tail+"/"+tail_t.strip("_")+"\t"+tail_+"\t"+tails+"\n")
                                
                                # import time
                                # print()
                                # print(1,head,2,head_t,h[1],h[0],tail,4,tail_t)
                                # print()
                                # time.sleep(60)
                            else:
                                if False:#len(h[1]) >= 2 and f" {tagsep} MAG" in h[1][0] and f" {tagsep} VV" in h[1][1]:
                                    mag = h[1][0].split(f" {tagsep} ")
                                    vv = h[1][1].split(f" {tagsep} ")
                                    head = mag[0] + "_" + vv[0]
                                    head_t = "MAG_VV"
                                    
                                    tail = h[0][len(mag[0])+len(vv[0]):]
                                    tail_t = tail_tag(h[1][2:])
                                else:
                                    head_ = h[1][0].split(f" {tagsep} ")
                                    
                                    # head_xsn_tmp = None
                                    # if len(h[1]) >= 2:
                                    #     head_xsn_tmp = h[1][1].split(f" {tagsep} ")
                                    # head_xsn = ""
                                    # head_xsn_t = ""
                                    # if False:# head_xsn_tmp and head_xsn_tmp[1] == "XSN":
                                    #     head_xsn = head_xsn_tmp[0]
                                    #     head_xsn_t = head_xsn_tmp[1]
                                    
                                    # if head_xsn != "":
                                    #     head = h[0][:len(head_[0]+head_xsn)]
                                        
                                    #     head_t =  head_[1] +"_" + head_xsn_t + "_"
                                        
                                    #     tail = h[0][len(head_[0]+head_xsn):]
                                    #     tail_t = tail_tag(h[1][1:],xsn=head_xsn_t)
                                    # else:
                                    head = h[0][:len(head_[0])]
                                    head_t =  head_[1]
                                    
                                    tail = h[0][len(head_[0]):]
                                    tail_t = tail_tag(h[1][1:],xsn=None)
                                    
                                    if tail_t != "" and tail == "":
                                        head_t = head_t +"_"+tail_t
                                        tail_t = ""
                                    # if (tail_t.startswith("XSV") or tail_t.startswith("XSA")):
                                    # print(h)
                                    # if "스럽" in h[0] or h[1][1] == "스러":
                                    #     print("\n",h[1])
                                    # print(h[1]) 
                                    if head.strip() != "" and (head_t.startswith("XSA") or head_t.startswith("XSV") or head_t.startswith("VV") or head_t.startswith("VA") or head_t.startswith("VCP")):
                                        # if tail_t != "" and tail == "":
                                        #     head_t = head_t +"_"+tail_t
                                        #     print("\n"+str(10)+"\t"+head+"/"+head_t+"\t"+tail+"/"+tail_t+"\t"+" ".join(h[1])+"\t"+fn+"-"+str(idx))
                                        # else:#if tail_t != "" and tail == "":
                                            
                                        head_tail_orif.write("\n"+str(10)+"\t"+head+"/"+head_t+"\t"+tail+"/"+tail_t+"\t"+" ".join(h[1])+"\t"+fn+"-"+str(idx)+"\n")
                                    
                                    if tail.strip() != "":
                                        # print(tmp)
                                        if True:
                                            head_tail_orif.write("\n"+str(8)+"\t"+head+"/"+head_t+"\t"+tail+"/"+tail_t+"\t"+" ".join(h[1])+"\t"+fn+"-"+str(idx)+"\n")
                                        elif len(h[1]) > 1 and f" {tagsep} X" in h[1][1]:
                                            tailtop = h[1][1].split(f" {tagsep} ")[0]
                                            # print("\n"+str(9)+"\t"+tail[0][:len(tailtop)]+"\t"+h[1][1]) 
                                            # xtag[tail[:len(tailtop)]] = h[1][1]
                                            xtag[tail[:len(tailtop)]].append(head+"/"+tail+"/"+"_".join(h[1]))
                                        # print("2222",number)
                                
                                    if tail.strip() != "" and tail[0] != h[1][1][0]:
                                        # print(tail[:2],h)
                                        char = j2hcj(htsep.join(h[1][1:]))
                                        char = char.replace(f" {tagsep} ",f"{tagsep}")
                                        # print(2,head+"/"+head_t,tail+"/"+tail_t,h[1])
                                    
                                        if cnt_eojid < len(h[1]):
                                            char = head_tail_1chk(head,tail,h[1][cnt_eojid]) + char
                                        result_ht_dict_f.write(str(2)+"\t"+tail+"/"+head_t.strip("_")+"@"+tail_t.strip("_")+"\t"+char+"\t"+char2+"\n")
                                        
                                    # elif tail.strip() != "" and len(h[1][1]) == 2 and tail_t.startswith("VV"):
                                    #     print("\n",h)
                                    #     char = j2hcj(htsep.join(h[1][1:]))
                                        
                                    #     char = char.replace(f" {tagsep} ",f"{tagsep}")
                                    #     if cnt_eojid < len(h[1]):
                                    #         char = head_tail_1chk(head,tail,h[1][cnt_eojid]) + char
                                    #     result_ht_dict_f.write(str(2)+"\t"+tail+"/"+head_t.strip("_")+"@"+tail_t.strip("_")+"\t"+char+"\t"+char2+"\n")
                                    # if tail.strip() != "":
                                        # print(str(8)+"\t"+head+"/"+head_t+"\t"+tail+"/"+tail_t+"\t"+h[1])
                                
                                    if head.strip() != "" and head[-1] != h[1][0].split(f" {tagsep} ")[0][-1]:
                                        char = j2hcj(h[1][0])
                                        char = char.replace(f" {tagsep} ",f"{tagsep}")
                                        # print(4,head+"/"+head_t,tail+"/"+tail_t,h[1])
                                    
                                        if tail != "":
                                            # if "ㄱ" <= tail[0] <= "ㅎ":
                                            
                                            result_ht_dict_f.write(str(4)+"\t"+head+"/"+head_t.strip("_")+"@"+tail_t.strip("_")+"\t"+char+"\t"+char2+"\n")
                                        else:
                                            result_ht_dict_f.write(str(4)+"\t"+head+"/"+head_t+"\t"+char+"\t"+char2+"\n")
                            headm = head+tagsep+head_t.strip("_")
                            tailm = ""
                            if tail.strip() != "":
                                tailm = tail + tagsep + tail_t.strip("_")
                            if tailm == "":
                                headtail_tk += headm + " "
                            else:
                                headtail_tk += headm + htsep +tailm + " "
                                
                        cnt = 0
                        prev = 0
                        flag = False
                        # print(headtail_tk)
                        if headtail_tk.strip()!="":
                            result_ht.write(headtail_tk.strip()+"\n")
                except Exception as ex:
                    cnt = 0
                    prev = 0
                    flag = False
                    # print(tmp)
                    # print(h[1])
                    import traceback
                    # print(head)
                    # print(tmp)
                    traceback.print_exc()
                    # exit()
                    # # print(ex)
                    # # import time
                    # # time.sleep(60)
                    # print(mptps)
                    # print(lt)
                    # print(tmp)
                    # print(h)
                    # tmp = []
                    # mptps = []
                    # lt = ""
                    # exit()
                
                # break
            elif cnt > 0:
                ltmp = l
                sam.append(ltmp)
                
                l = l.strip().split("\t")
                ttt.append(l)
                mp = l[0]
                if True:
                    eojid = int(l[-1]) - 1 #-symcheck
                else:
                    eojid = int(l[-1]) #-symcheck
                # 
                try:
                    mptmp = mp.split(f" {tagsep} ")[0]
                    # if mptmp.strip() == "31":
                    #     print(tmp)
                    #     exit()
                    number += 1

                    # if "♥" == removesym(mptmp):
                    #     # print("\n",0,lt) 
                    #     # print("\n",10,mptmp)
                        
                    #     symcheck += 1
                    #     # print("\n",9,tmp[eojid],mp)
                    #     continue
                    if True:#lt == "오늘 끝나고 또 어디 가요 언니 ?\n":
                        mp_ = mp.split(f" {tagsep} ")
                        # print(is_sym(eojid)[0])
                        # print()
                        if is_sym(mp_[0]) or mp_[1].split("\t")[0] == "SF" or mp_[1].split("\t")[0] == "SW":
                            if len(tmp) -1 >= eojid and  not is_sym(tmp[eojid][0]):
                                tmp = tmp[:eojid] + [[mp_[0],[]]] + tmp[eojid:]
                            elif len(tmp) - 1 < eojid:
                                tmp = tmp + [[mp_[0],[]]]                        
                        # if is_sym(mp_[0]) or mp_[1].split("\t")[0] == "SF" or mp_[1].split("\t")[0] == "SW":
                        #     if 
                            # print(5555,tmp)
                        # print(55555555,tmp,mp)
                        # exit()
                    # if not is_sym(mp[0])
                    tmp[eojid][1].append(mp)
                    # print("\n",10,tmp[eojid],mp)
                    # print(tmp)
                    mptps.append(mp)
                # print(l)
                except:
                    1 == 1
                    # mp_ = mp.split(f" {tagsep} ")
                    # if not is_sym(tmp[eojid][0]) and (mp_[1].split("\t")[0] == "SF" or mp_[1].split("\t")[0] == "SW"):
                    print(tmp[:6])
                        
                    print(tmp)
                    
                    print(eojid)
                    print(lt)
                    print(len(tmp))
                    print(mp)
                    print(sam)
                    # exit()
                    import traceback
                    traceback.print_exc()
                #     # print(tmp)
                    # if eojid >= len(tmp):
                    #     print(lt)
                    #     print(tmp)
                    #     print(eojid)
                    #     print(len(tmp))
                    #     print(ttt)
                    #     print("--------")
                        # tmp = []
                        # The code is a Python script with a commented-out `exit()` function call.
                        # This means that the `exit()` function will not be executed when the script
                        # is run.
                        # exit()
                # print(l)
                # print(tmp[eojid][1])
                # if len(tmp[eojid][1]) >= 2 and prev == eojid and " / XSV" in tmp[eojid][1][-2]:
                #     print(tmp[eojid][0],tmp[eojid][1])
                #     prev = -1
                # else:
                
                #     prev = -1
                
                # if " / XSV" in ltmp:
                #     print(l)
                #     prev = eojid 
    # print(tmp)
                # The code is checking if the variable `prev` is not equal to the variable `eojid`. If
                # they are not equal, then the code inside the if block (denoted by the `#` symbols)
                # will be executed.
                # if prev != eojid:
                    # cnt = 0
                # prev = eojid
                    # print(3,eojid,len(tmp))
            elif cnt == 0:
                # print(l)
                symcheck = 0
                ttt = []
                lt = l
                line = l.strip().split()
                tmp = []
                number += 1
                for ln in line:
                    # if  "♥" == removesym(ln):
                    #     # print("\n",0,ln)
                    #     continue
                    tmp.append([ln,[]])
                # print(tmp)
                # print(l)0
                # tmp = [[ln,[]] for ln in line]
                mptps = []
                # print(stmp)
                # for idx,ln in line:
                #     if " / XSA" in ln or " / XSV" in ln:
                #         xsindex = idx
                sam =[]
                sam.append(l)
                cnt += 1
            # pbar.update(1)
            linecnt += 1
            if linecnt % 50000 == 0:
                sys.stdout.write("\r"+fn+"="+str(linecnt)+"/"+str(count))
        result_ht.close()
        result_ht_dict_f.close()
    import json 
    # print("\n",json.dumps(xtag,indent=4,ensure_ascii=False))
    # print("\n"+str(99)+"\t"+str(errors))
    # print("\nsymlength",symcount/symlength)
if makeheadtail:
    if __name__ == "__main__":
        # auto = Auto2()
        # exit()
        datapath = os.environ["Desktop"] + os.sep + "dataset"

        dir = os.listdir(datapath+"\\ht_dataset\\modu")

        files = []
        # fn = []
        print(dir)
        import multiprocessing
        for idx,dir_ in enumerate(dir):
            
            if not dir_.endswith(".txt"):# or dir_.startswith("SX"):
            # if not os.path.isfile(fn):
                continue
            else:
                print(dir_)
            fn = datapath+"\\ht_dataset\\modu\\"+dir_
            
            dir_ = dir_.replace(".txt","")
            file_tail_dict = datapath+"\\ht_dataset\\modu\\train\\"+f"tail_dict_{dir_}.txt"
            result_ht = datapath+"\\ht_dataset\\modu\\train\\"+f"headtail_train_mp_{dir_}.txt"
            ps = multiprocessing.Process(target = make_ht_train,args=(fn,file_tail_dict,result_ht), daemon=True)
            # ps.start()
            ps.start()
            ps.join()
            exit()
            files.append(ps)
        for ps in files:
            # ps.daemon
            ps.start()
        for ps in files:
            ps.join()
            
        respath = os.path.join(datapath,"ht_dataset","modu","train")
        reshtpath = open(os.path.join(respath,"headtail_train_mp.txt"),"w",encoding="utf-8")
        restaildictf = open(os.path.join(respath,"tail.dict"),"w",encoding="utf-8")
        for dir in os.listdir(respath):
            tmppath = os.path.join(respath,dir)
            if dir.startswith("headtail_train_mp_"):
                with open(tmppath,encoding="utf-8") as f:
                    for l in f:
                        l = l.strip()
                        l = l.split()
                        ltmp = []
                        flags = False
                        if l == "♥ ♥ 뉴스룸 ♥ 한우신 ♥ 대선 테마주 기업들":
                            flags = True
                        for ll in l:
                            httmp = []
                            symflag = False
                            for lll in ll.split(htsep):
                                lll = lll.split(tagsep)
                                # eojtmp = []
                                # for lll in lll.split()
                                if lll[1] == "♥":
                                    print(lll)
                                if lll[0] == "♥":
                                    symflag = True
                                    # if flags:
                                    #     print(lll)
                                    # if lll[1] != "SF" and lll[1] != "SN" and lll[1] != "SL":
                                    #     lll[1] = "S"
                                if not symflag:
                                    # if flags:
                                        # print(lll)
                                    httmp.append(tagsep.join(lll))
                            ltmp.append(htsep.join(httmp))
                        
                        l = " ".join(ltmp)
                        # if flags:
                        #     print(l)
                        # print(l)
                        # exit()
                        l = re.sub(" +"," ",l)
                        reshtpath.write(l.strip()+"\n")
                os.remove(tmppath)
            elif dir.startswith("tail_dict_"):
                with open(tmppath,encoding="utf-8") as f:
                    for l in f:
                        restaildictf.write(l)
                os.remove(tmppath)
            
            
        reshtpath.close()
        restaildictf.close()
        
        with open(os.path.join(respath,"headtail_train_mp.txt"),encoding="utf-8") as f:#, open("noise_dict.txt","w",encoding="utf-8") as ndf:

            count = defaultdict(int)
            countline = defaultdict(set)
            
            counttag = defaultdict(int)
            counttagline = defaultdict(set)
            counttagword = defaultdict(set)
            
            for idx,l in enumerate(f):
                l = l.strip()
                l = l.split()
                # print(l)
                for ll in l:
                    # if ll.strip() == "":
                    #     continue
                    ll = ll.replace("+","-").replace("/","|")
                    ll = ll.replace(htsep,"+").replace(tagsep,"/")
                    head,tail = get_mp_tags(ll)
                    if len(tail) >= 2: 
                        mp = head[0]+"+"+tail[0]
                        head[1] = duple_tag(head[1],"NNG")
                        head[1] = duple_tag(head[1],"NNP")
                        tag = head[1]+"+"+tail[1]
                        
                    else:
                        mp = head[0]
                        head[1] = duple_tag(head[1],"NNG")
                        head[1] = duple_tag(head[1],"NNP")
                        tag = head[1]
                    
                    # tag_ = re.sub(r"NNG_{2,}","_NNG_",tag)
                    # tag_ = re.sub("_+","_",tag_).strip("_")
                    # tag_ = re.sub(r"NNP_{2,}","_NNP_",tag_)
                    # tag_ = re.sub("_+","_",tag_).strip("_")
                    counttag[tag] += 1
                    counttagline[tag].add(idx)
                    counttagword[tag].update([mp])
            
            tagerror = defaultdict(list)
            for k,v in tqdm(counttag.items()):
                lines = list(counttagline[k])
                words = list(counttagword[k])
                lines = list(map(str,lines))
                tagerror["tag"].append(k)
                tagerror["count"].append(v)
                tagerror["word"].append(", ".join(words))
                tagerror["lines"].append(", ".join(lines))
                        # for llll in lll:
                            # llll = llll.split(htsep)
            df2 = pd.DataFrame(tagerror)
            df2 = df2.query("count <= 100")
            df2.to_csv("tagerror.csv",encoding="utf-8-sig")
            f.seek(0)
            for idx,l in enumerate(f):
                # re.search("VV_.NN.\s")
                l = l.strip()
                l = l.split()
                
                for ll in l:
                    wordtmp = ll
                    # if htsep in ll:
                    #     continue
                        
                    ll = ll.split(htsep)
                    
                    for lll in ll:
                        lll = lll.split(tagsep)[1]
                        if "NNB" in lll or "NN" in lll or "NP" in lll:
                            tmptag = []
                            for tag in lll.split("_"):
                                tmptag.append(tag[0])
                            if ("E" in tmptag or "J" in tmptag or "V" in tmptag):
                                count[wordtmp] += 1
                                countline[wordtmp].add(idx)
            tmpori_f = open("ori_tmp.txt","w",encoding="utf-8")
            df = defaultdict(list)

            kiwi = Kiwi()
            auto = Auto2()
            for k,v in tqdm(count.items()):
                lines = list(countline[k])
                lines = list(map(str,lines))
                # k = k.replace("/","_")
                k = k.replace(htsep,"+")
                k = k.replace(tagsep,"/")
                # print(k)
                oriwords = ""
                for wordstmp in k.split("+"):
                    wtmp = wordstmp.split("/")[0]
                    oriwords += wtmp
                # print(oriwords)
                # exit()
                tmpori_f.write(oriwords+"\n")
                auto_oriwords = auto.autospace(oriwords)
                df["word"].append(k)
                df["oriword"].append(oriwords)
                df["autospace"].append(auto_oriwords)
                df["count"].append(v)
                df["lines"].append(", ".join(lines))
                
                tokens = [[] for _ in range(len(auto_oriwords.split()))]
                for m in kiwi.tokenize(auto_oriwords):
                    # print(m.word_position,auto_oriwords)
                    tokens[m.word_position].append(m.form+"/"+m.tag)
                mps = []
                for token in tokens:
                    mps.append("+".join(token))
                df["mps"].append(" ".join(mps))
            tmpori_f.close()
            df = pd.DataFrame(df)
            df = df.query("count <= 100")
            df.to_csv("noise_dict.csv",encoding="utf-8-sig")
            print(df)
            res = set()
            df = df.dropna()
            print(df)
            df = df.loc[:,["lines"]].values
            df2 = df2.loc[:,["lines"]].values
            import numpy as np
            df = np.reshape(df,df.shape[0])
            df2 = np.reshape(df2,df2.shape[0])
            
            df = df.tolist() + df2.tolist()
            for d in df:
                d = d.strip()
                d = d.split(",")
                for dd in d:
                    res.add(int(dd.strip()))
            delnoise_f = open(os.path.join(respath,"headtail_train_mp_delnoise.txt"),"w",encoding="utf-8") 
            with open(os.path.join(respath,"headtail_train_mp.txt"),encoding="utf-8") as f:
                for idx, l in enumerate(f):
                    if idx not in res:
                        l = l.strip()
                        delnoise_f.write(l+"\n")
            
            delnoise_f.close()
        # for ps in files:
        
            
            

    # result_ht_dict_f = open("tail.dict","w",encoding="utf-8")
    # result_ht = open("headtail_train_mp.txt","w",encoding="utf-8")
            
    # exit()

    # with open("postagger/kcc150_morphs.txt",encoding="utf-8") as mp, open("postagger/kcc150_tag.txt",encoding="utf-8") as tag:
            
    #     mfile = open("noise_remove_morphs.txt","w",encoding="utf-8")
    #     tfile = open("noise_remove_tags.txt","w",encoding="utf-8")
    #     for m,t in zip(mp,tag):
    #         m_ = m
    #         t_ = t
    #         # print(m)
    #         # if re.match(r"(\d+.\d+|\d+)",m):
    #         #     m = re.sub(r"(\d+.\d+|\d+)",r" \1 ",m)
    #         #     m = re.sub(" +"," ",m)
                
    #         # print(t)
    #         #     t = t.replace("+SN"," SN").replace("SN+","SN ").replace("SN_","SN ").replace("NR+","SN NR+").replace("NR_","NR ")#.replace("NR_","NR ")
    #         #     t = re.sub(" +"," ",t)

    #         # print(1,m)
    #         # print(2,t)
    #         resm = []
    #         rest = []
    #         for mm,tt in zip(m.split(),t.split()):
    #             if re.match(r"(\d+.\d+|\d+)",mm):
    #                 mm = re.sub(r"(\d+.\d+|\d+)",r" \1 ",mm)
    #                 mm = re.sub(r" +",r" ",mm)
    #             tags = []
    #             morphs = []
    #             for mmm in mm.split():
    #                 # morphs.append(mmm.strip("+"))
    #                 # for mmm_ 1q0i0in mmm.split("")
    #                 if re.match(r"(\d+.\d+|\d+)",mmm):
    #                     # print(111)
    #                     tags.append("SN")
    #                 else:
    #                     # if "+" not in tt:
    #                     # print(mmm,mm,tt)
    #                     tags.append(tt.strip("+"))
    #                     # else:
                            
                
    #             # if "+" in mm:
                    
    #             #     tags.append("+"+tt.split("+")[-1])
    #             #     print(mm,tags)
    #             tt = " ".join(tags)
    #             # tt = tt.replace(" +","+")
    #             # mm = " ".join(morphs)
    #             # tt = tt.replace("+SN"," SN").replace("SN+","SN ").replace("SN_","SN ").replace("NR+","SN NR+").replace("NR_","NR ")#.replace("NR_","NR ")
    #             tt = re.sub(r" +"," ",tt)
                
    #             resm.append(mm)
    #             rest.append(tt)
    #         m = " ".join(resm)
    #         t = " ".join(rest)
    #         m = re.sub(r" +"," ",m)
    #         t = re.sub(r" +"," ",t)    
    #         resm = []
    #         rest = []
    #         tagtmp = []
    #         tokentmp = []
    #         for mmm,ttt in zip(m.split(),t.split()):
    #             if "+" in mmm:
    #                 # if mmm[-1] in tokenslist:
    #                 mmm_ = mmm.split("+")
    #                 ttt_ = ttt.split("+")
    #                 if ttt_[0].startswith("V") and len(mmm_[0]) > 1 and mmm_[0][-1] in tokenslist:
    #                     # print(mmm_[0],mmm_[1])
    #                     mmm = mmm_[0][:-1]+ "+" + mmm_[0][-1]+mmm_[1]
    #                     ttt = "NNG+"+ttt.replace("+","_")
                        
    #             else:
    #                 if ttt.startswith("V"):
    #                     if len(mmm) > 2 and mmm[-1] in tokenslist:
    #                         mmm = mmm[:-1]+ "+" + mmm[-1:]
    #                         ttt = "NNG+"+ttt
    #                         # print(mmm,ttt)
    #                     elif len(mmm) > 3 and mmm[-2] in tokenslist:
    #                         mmm = mmm[:-2]+ "+" + mmm[-2:]
    #                         ttt = "NNG+"+ttt
    #                         # print(mmm,ttt)
    #                     elif len(mmm) > 5 and mmm[-3] in tokenslist:
    #                         mmm = mmm[:-3]+ "+" + mmm[-3:]
    #                         ttt = "NNG+"+ttt
    #                         # print(mmm,ttt)
    #             tokentmp.append(mmm)
    #             tagtmp.append(ttt)
    #         m = " ".join(tokentmp)
    #         t = " ".join(tagtmp)
    #         flag = True
    #         for mm,tt in zip(m.split(),t.split()):
    #             for mmm,ttt in zip(mm.split("+"),tt.split("+")):
    #                 if len(mmm) > 5 and ttt.startswith("V"):
    #                     print(mmm,ttt)
    #                     flag = False
    #                     break
    #         if flag:
    #             mfile.write(m.strip()+"\n")
    #             tfile.write(t.strip()+"\n")
    #         for mm,tt in zip(m.split(),t.split()):
    #             try:
    #                 # print(mm,tt)
    #                 mm = mm.strip("+")
    #                 tt = tt.strip("+")
                    
    #                 mm = mm.split("+")
    #                 tt = tt.split("+")
    #                 # print(mm,tt)
    #                 if tt[0].startswith("V") and "_" in tt[0] and "E" in tt[0]:
    #                     if not re.match(r"\d+",mm[0]) and not re.match(r"[a-z|A-Z]+",mm[0]):
    #                         head[mm[0]+"/"+tt[0]] += 1
    #                         # print(mm,tt)
    #                 if len(mm) == 2 and "_" in tt[1] and "E" in tt[1] and tt[1].startswith("V"):
    #                     # if mm[1] == "실점했다":
    #                         # print(m,t)
    #                         # exit()
    #                     if not re.match(r"\d+",mm[1]) and not re.match(r"[a-z|A-Z]+",mm[1]):
    #                         tail[mm[1]+"/"+tt[1]] += 1
    #             except Exception as ex:
    #                 print("1:"+m_.strip())
    #                 print("2:"+t_.strip())
    #                 print(mm,tt)
    #                 # exit()      
    #                 True          

    # from kiwipiepy import Kiwi

    # def writefile(filename,val,name=None):
    #     heads = defaultdict(int)
    #     heads2 = defaultdict(int)
    #     kiwi = Kiwi()
    #     val = sorted(val.items(),key=lambda x: x[1],reverse=True)    
    #     with open(filename,"w",encoding="utf-8") as res:        
    #         sum = 0
    #         for k,v in val:
    #             sum += v
    #             # if v >= thread:
    #             # if name  == "head":
    #             #     print(k,v)
    #             if 100 < v:
    #                 res.write(k+"\t"+str(v)+"\n")
    #             # print(k,v)
    #             if k.startswith("/"):
    #                 continue
    #             # if len(j2hcj(h2j(k.split("/")[0][-1]))) == 3 and len(k.split("/")[0]) >= 3:
    #             if name == "head":
    #                 txt = k.split("/")[0]
    #                 pos = k.split("/")[1]
    #                 tokens = kiwi.tokenize(txt)
    #                 res_eomi = [tk.form for tk in tokens]
    #                 res_pos = [tk.tag for tk in tokens]
    #                 # heads[pos+"\t"+"+".join(res_eomi)+"\t"+"+".join(res_pos)] += v
    #                 # heads[txt[-1]] += v
    #                 # if len(txt) >= 3:
    #                 #     heads2[txt[-2:]] += v
    #                 heads[txt+"\t"+pos+"\t"+"+".join(res_eomi)+"\t"+"+".join(res_pos)] += v
    #                 # print(k,res_eomi,res_pos)    
    #             if name == "tail":
    #                 txt = k.split("/")[0]
    #                 pos = k.split("/")[1]
    #                 if pos.endswith("NNG") or pos.endswith("NNP") or pos.endswith("MAG") or pos.endswith("MM"):
    #                     continue
    #                 tokens = kiwi.tokenize(txt)
    #                 res_eomi = [tk.form for tk in tokens]
    #                 # tokens = kiwi.tokenize(txt)
    #                 res_pos = [tk.tag for tk in tokens]
                    
    #                 if len(res_eomi) > 1 and len(pos.split("_")) == 2:
    #                     eogan = res_eomi[0]
    #                     # res_eomi = res_eomi[1:]
    #                     heads[txt+"\t"+pos+"\t"+"+".join(res_eomi)+"\t"+"+".join(res_pos)] += v
    #                 elif len(res_eomi) > 1 and len(pos.split("_")) >= 3:
    #                     eogan = res_eomi[0]
    #                     # res_eomi = res_eomi[1:]
    #                     heads[txt+"\t"+pos+"\t"+"+".join(res_eomi)+"\t"+"+".join(res_pos)] += v
    #                 # else:
    #                     # print(k,res_eomi,res_pos)
    #                 #     heads[k+"\t"+txt+"\t"+pos+"\t"+"+".join(res_eomi)+"\t"+"+".join(res_pos)] += v
    #                     # heads[txt+"\t"+"+".join(res_eomi)] += v
    #         # print(sum / len(val))
    #     if name == "head" or name == "tail":
    #         filename = name+"_eomi.txt"
    #         with open(filename,"w",encoding="utf-8") as res:
    #             heads = sorted(heads.items(),key=lambda x: x[1],reverse=True)
    #             for k,v in heads:
    #                 # print(k)
    #                 if v >= 1:
    #                     res.write(k+"\t"+str(v)+"\n")
    #         if name == "head":
    #             filename = name+"2_eomi.txt"
    #             with open(filename,"w",encoding="utf-8") as res:
    #                 heads2 = sorted(heads2.items(),key=lambda x: x[1],reverse=True)
    #                 for k,v in heads2:
    #                     # print(k)
    #                     if v >= 1:
    #                         res.write(k+"\t"+str(v)+"\n")
        
    # writefile("head.txt",head,"head")
    # writefile("tail.txt",tail,"tail")'''

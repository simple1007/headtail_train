import argparse
import re

parser = argparse.ArgumentParser(description="Dataset Normalize")

parser.add_argument("--line",type=str,default="kcc150_morphs.txt")
parser.add_argument("--pos",type=str,default="kcc150_tag.txt")

args = parser.parse_args()
MAX = -1#100#300000
count = 0

X = []
T = []
f = False
err = 0

enc = "utf-8"
if False:
    with open(args.line,encoding=enc) as line, open(args.pos,encoding=enc) as pos:
        cc = 0
        for l,p in zip(line,pos):
            l = l.split()
            p = p.split()
            count += 1
            x = []
            t = []
            cc +=1
            for ll, pp in zip(l,p):
                
                # ll = re.sub(r"(\d+.\d+.\d+|\d+.\d+|\d+)",r" \1 ",ll)
                # ll = re.sub(" +"," ",ll)
                # # print(ll)
                # tempm = []
                # tempt = []
                # flag = False
                # match = re.findall(r"(\d+.\d+.\d+|\d+.\d+|\d+)", ll)
                # print(match)
                
                # ll = re.sub(r"(\d+)", r" \1 ",ll)
                # pp = pp.replace("XSN","@#@")
                # pp = pp.replace("SN"," SN ")#.replace("NR"," NR ").replace(" + "," ")
                # # pp = re,sub
                # pp = pp.replace("@#@","XSN")
                
                ll = re.sub(" +"," ",ll)
                pp = re.sub(" +"," ",pp)
                
                ll = ll.split()
                pp = pp.split()
                x = x + ll
                t = t + pp
                # continue
                # for x_, p_ in zip(ll,pp):
                #     try:
                #         int(x_)
                #         x_ = re.sub(r"(\d+.\d+.\d+|\d+.\d+|\d+)",r" \1 ",x_)
                # if cc == 2:
                #     print(x)
            
                # break
                
                # exit()
            # t_tmp = []
            # # print(x,t)
            # cmp = 0
            # t_ori = t
            # temp = False
            # prev_tmp_l = 0
            # # print(x,len(x))
            # # print(t,len(t))
            # for mindex, xx in enumerate(x):
            #     cmp_flag = False
            #     try:
            #         int(xx)
            #         if t[mindex] != 'SN':# and t[mindex] == 'NR':
            #             # if t[mindex+1] != 'NR':
            #             print(xx)
            #             t_tmp.append('SN')
            #             cmp += 1
            #             cmp_flag = True
            #     except:
            #         # print(xx)
            #         _ = mindex
            #     # t_tmp.append(t[i])
            #     # print(cmp,mindex,mindex-cmp)
            #     # print(mindex,xx,t[mindex-cmp],cmp)
            #     # print(len(t),t,mindex)
            #     # try:
            #     # print(t,mindex,x[mindex])
            #     t_tmp.append(t[mindex])
            #     # except:
            #     #     # err += 1
            #     #     break
            #         # print(x)
            #         # print(t,mindex)
            #         # print(len(x),len(t))
            #         # print(t_ori)
            #         # exit()
            #     # print(x,t)
            #     if cmp_flag:
            #         # print(t_tmp)
            #         # print(x,t)
            #         # exit()
            #         # print("c",len(x))
            #         # print(t,t_tmp,t_ori[mindex-1])
            #         # if prev_tmp_l < len(t_tmp):
            #         # print("----")
            #         print(t_tmp)
            #         # print(t_ori)
            #         # print(t_ori)
            #         # print(len(t_tmp))
            #         print(t_ori[len(t_tmp):])
            #         t = t_tmp + t_ori[len(t_tmp):]
            #         print(t)
            #         # exit()
            #             # prev_tmp_l = len(t_tmp)
            #         # print(len(t))
            #         # print(x)
            #         temp= True
            # # continue        
            #     # t_tmp.append(t[mindex])
            # if temp:
            #     err += 1
            #     # print(x)
            #     print(x)
            #     print(t)
            #     print(len(x)==len(t))
            #     # print(x)
            #     # print(t)
            #     # if err == 1:
            #     #     exit()
            #     # if cmp_flag:
            #     #     cmp += 1
            #     # t = t_tmp
            # # print(x,t_tmp)
            # # exit()
            # if cc == 1000000:
            #     break
                # if len(match)==1:
                #     # print(ll)
                    
                #     ll = ll.split()
                #     # print(ll)

                #     tempm.append(ll[0].strip("+"))
                #     if len(ll) > 1:
                #         tempm.append(ll[1].strip("+"))
                #     pp = pp.split("_")
                    
                #     tempt.append("SN")
                #     if len(ll) > 1:
                        
                #         if len(pp) != 1:
                #             tempt.append("_".join(pp[1:]))
                #         else:
                #             tempt.append("_".join(pp))
                        
                #     # print(tempm,tempt)
                #     x = x + tempm
                #     t = t + tempt
                #     flag = True
                # elif len(match) > 1:
                #     # print(ll,pp)
                #     # ll = ll.replace("+","")
                #     # for m in match:
                #     #     if '.' in m:
                #     #         print(ll,match)
                #     #         exit()
                #     ll = ll.split()
                #     tempm.append(ll[0].strip("+"))
                #     tempm.append(ll[1].strip("+"))
                #     tempm.append(ll[2].strip("+"))
                #     if len(ll) == 4:
                #         tempm.append(ll[3].strip("+"))
                #     pp = pp.split("+")
                #     tempt.append("SN")
                #     tempt.append(pp[0].replace("SN_",""))
                #     tempt.append("SN")
                #     if len(ll) == 4:
                #         continue
                #         # try:
                #             # tempm.append(ll.replace("SN_",""))
                        
                #         try:
                #             pp[1] = pp[1].replace("SN_","")
                #         # except:
                #         #     print(ll,pp)
                #         #     exit()
                #         # print(pp)
                #         # print(pp)
                #             if len(pp[1].split("_")) > 2:
                #                 # print("2",pp[1])
                #                 tempt.append("_".join(pp[1].split("_")[2:]))
                #                 # print("__" in tempt[-1],tempt[-1])
                #             else:
                #                 tempt.append("_".join(pp[1]))
                            
                            
                #             # if "___" in "_".join(pp[1]) or "___" in "_".join(pp[1].split("_")[2:]):
                #             #     print("dfafdasf","_".join(pp[1].split("_")[2:],"_".join(pp[1])))
                #             #     exit()
                #         except:
                #             flag = True
                #             x = []
                #             t = []
                #             break
                #     # print(tempm,tempt)    
                #     x = x + tempm
                #     t = t + tempt
                #     flag=True
                # # continue
                # if not flag:
                #     # print(ll,pp)
                #     x = x + [ll]
                #     t = t + [pp]
                
                # if ll == "P씨+는" or f:
                #     print(x)
                #     # f = True
            if len(x) > 0 and len(t) > 0:
                X.append(x)
                T.append(t)
            f = False
            if count == MAX:
                break
            # if cc == 2:
            #     print(x)
            #     break
            # print(len(x),len(p))
    # print("와우")
n_tags = ["될","됄","한","했","하","해","되","됐","됬",'한','할','될','시키','시켰','시킬','시킨','된','스러',"스럽","스런","드리","드렸"]
resultX = []
resultY = []
count = 0
cnt = 0#len(X)
temp_err = []
cc = 0

with open("morphs.txt","w",encoding=enc) as mf, open("tags.txt","w",encoding=enc) as tf:
    # print(resultX)
    # mf.write("\n".join(resultX))
    # tf.write("\n".join(resultY))
    with open(args.line,encoding=enc) as line, open(args.pos,encoding=enc) as pos:
        for XX, TT in zip(line,pos):
            XX = XX.strip()
            TT = TT.strip()
            
            XX = XX.split()
            TT = TT.split()
            m_ = []
            t_ = []
            # print(XX,TT)
            cc += 1
            # print(cc)
            # print("ddd",len(XX),len(TT))
            # print(XX)
            # print(TT)
            if len(XX) != len(TT):
                # count += 1
                # temp_err.append([XX,TT])
                # print(len(XX),len(TT),XX,TT)
                exit()
                # continue
            cnt += 1
                # print(XX,TT)
                # break
            xt = []
            pt = []

            for xxt, ppt in zip(XX,TT):
                if xxt != '':
                    xt.append(xxt)
                if ppt != '':
                    pt.append(ppt)
            XX = xt
            TT = pt
            for xx,tt in zip(XX,TT):
                # print("왜없어",XX)
                # continue
                xx = xx.strip("+")
                tt = tt.strip("+")
                # print(xx)
                # continue
                # print(XX,xx)
                # print(TT,tt)
                try:

                    if "+" in xx and xx.split("+")[0][-1] in n_tags and tt[0] == "V" and len(xx.split("+")[0]) >= 3:
                        # print("fsfsfsa",xx)
                        tempm = xx.split("+")
                        tempt = ""
                        tempm_ = ["",""]
                        tempm_[0] = tempm[0][:-1] + "+"
                        tempm_[1] = tempm[0][-1] + tempm[1]

                        tempm = "".join(tempm_).strip("+")

                        tt = tt.replace("+","_")
                        if tempm_[0] != "+":
                            tempt = "NNP"+"+"+tt
                            # print(tempt)
                        else:
                            tempt = tt
                        # print("dsfs",tempm)

                        # print(tempm,tempt)
                    # print(xx)
                    elif "+" in xx and xx.split("+")[0][-2:] in n_tags and tt[0] == "V" and len(xx.split("+")[0]) >= 4:
                        tempm = xx.split("+")
                        tempt = ""
                        tempm_ = ["",""]
                        tempm_[0] = tempm[0][:-2] + "+"
                        tempm_[1] = tempm[0][-2:] + tempm[1]

                        tempm = "".join(tempm_).strip("+")

                        tt = tt.replace("+","_")
                        if tempm_[0] != "+":
                            tempt = "NNP"+"+"+tt
                            # print(tempt)
                        else:
                            tempt = tt
                    elif "+" not in xx and xx.split("+")[0][-1] in n_tags and tt[0] == "V" and len(xx.split("+")[0]) >= 3:
                        tempt = "NNP" + "+" + tt#"VV"
                        tempm = xx.split("+")[0][:-1] + "+" + xx.split("+")[0][-1]
                    elif "+" not in xx and xx.split("+")[0][-2:] in n_tags  and tt[0] == "V" and len(xx.split("+")[0]) >= 4:
                        tempt = "NNP" + "+" + tt
                        tempm = xx.split("+")[0][:-2] + "+" + xx.split("+")[0][-2:]
                    elif len(xx.split("+")[0]) >= 2 and xx.split("+")[0][-1] == "들" and tt[0] == "N":
                        # print(xx,tt)
                        xx = xx.split("+")
                        
                        if len(xx) == 2:
                            xx[0] = xx[0][:-1]

                            xx[1] = "들"+xx[1]

                            tempm = "+".join(xx)
                            tempt = tt.replace("+","+XSN")
                        else:
                            tempm = xx[0][:-1] + "+들"
                            tempt = tt + "+XSN"

                    else:
                        tempm = xx
                        tempt = tt      
                        # print(xx)  
                    # print(tempm)
                    m_.append(tempm)
                    t_.append(tempt)
                except:
                    print(xx,tt)
                    # exit()

            m_ = re.sub(" +"," "," ".join(m_))
            t_ = re.sub(" +"," "," ".join(t_))
            m__ = m_
            t__ = t_
            if len(m_.split()) == len(t_.split()):
                count +=1
                resultX.append(m_)
                resultY.append(t_)
                mf.write(m_+"\n")
                tf.write(t_+"\n")
            else:
                print(xt,pt)
        # exit()
        # break
    # if cc == 2:
    #     # print(cc)
    #     break
    # else:
    #     for ii,iii in zip(m_.split(),t_.split()):
    #         print(ii,iii,end=" ")
    #     print()
    # else:
    #     print(m_)
    #     print(t_)

print(count==cnt)
print(count,cnt)
# for i in range(10):
#     # print(temp_err[i])
#     x, t = temp_err[i]


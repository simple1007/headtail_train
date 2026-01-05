import re

def tag_merge(tag_list,token,posvocab):    
    result = []
    length_count = 0
    pos_result = []
    # print(tag_list,token)
    pos = []
    
    for li,(l,tk) in enumerate(zip(tag_list,token)):
        # print(l,"\n",tk)
        
        l = l.replace('\n','').replace(" +","+").replace("+ ","+").replace(" + ","+")
        tk = tk.replace('\n','').replace(" +","+").replace("+ ","+").replace(" + ","+").rstrip("+")
        l = re.sub(" +"," ",l).strip()
        tk = re.sub(" +"," ",tk).strip()
        # print(l,"\n",tk)
        uni,bi = posvocab.tokens2(token[li].replace("+"," "))
        # uni = uni[0]
        if l.startswith('[SOS] '):
            l = l[6:]
        l = l.strip()
        l = re.sub(' +',' ',l)
        temp_l = l
        l = l.split(' ')
        tk = tk.replace('\n','')
        tk = tk.strip()
        flag = False
        
        uni = [uni]
        uni = uni[0]#+ [-1] * (300-len(uni[0]))
        uni = uni[:300]
        l = l[:len(uni)]
        flag_bio_chk = False
        for i,ll in enumerate(l):
            # print(i)
            if i>0 and (uni[i-1] == " " or uni[i-1] == "+"):
                flag = False
                flag_bio_chk = False
            if not flag and (uni[i] != " " or uni[i] != "+"):#ll.startswith('B_'):# and not flag:
                if ll.startswith('B_'):
                    result.append(ll[2:])
                else:
                    if ll != "O":
                        result.append("UNK")
                flag = True
                
            elif ll.startswith("I_") and flag and (uni[i] != " " or uni[i] != "+"):
                if result[-1] != ll[2:] and not flag_bio_chk:
                    result[-1] = "UNK"
                    flag_bio_chk = True
            elif uni[i] == "+":
                result.append("+")
            elif False:
                # print(temp_l)
                if len(result) == 0:
                        # result.append(ll.replace('I_',''))
                    result.append('FAIL')
                    # elif result[-1] != ll.replace('I_',''):
                    #     result[-1] = 'FAIL'
            # print(ll)
            elif ll == "[EOS]":
                break
            # print(posvocab.uni2index[" "],uni[i])
        poss = []
        # print(tk,result)
        # for tk_ in tk.split():
        tmp = []
            # print(tk_,result)
        idx_space = []#re.match()
        # for tt in tk:
        #     if tt == " ":
        #         idx_space.append()
        tk = tk.replace("+"," +")
        tk = re.sub(r" +",r" ",tk)
        # print(tk,result)
        for tk__,tag__ in zip(tk.split(),result):
            # print(tk__,)
            # if tag__ != "+":
            tmp.append(f"{tk__}/{tag__}")
            # else:
            #     tmp.append("+")
        # print(tmp)
        poss = " ".join(tmp)
        # poss = poss.replace(" + ","+")
        poss = re.sub(r" +",r" ",poss)
        poss = poss.replace(" +","+")
        pos_result.append(poss)
        result = []
    #     tk = tk.replace('\n','')
    #     temp_tk = tk
    #     tk = tk.split(' ')
    #     result_tk = []
    #         # print(len(tk),len(result))
    #     for index,tk_ in enumerate(tk):
    #         if '+' in tk_:
    #             result_tk.append(index)
    #             # print(result)
    #             try:
    #                 if (index+1) <= len(result):
    #                     result[index] = result[index] + '+' + result[index+1]
    #                     del result[index+1]
    #             except:
    #                 l = l
    #     line_pos_tk = []
    #     result = ' '.join(result)
    #     result = result.split(' ')
    #     #pos.append(result)
    #     #print(result)
    #     #print(tk)
    #     #print(result,tk)
    # #for p, tk in zip(pos,token):
    #     #p = p.split(' ')
    #     #tk = tk.split(' ')
    #     # print(result)
    #     for tag,tk_ in zip(result,tk):
    #         tag__ = tag.split('+')
    #         tk__ = tk_.split('+')
    #         temp_pos_tk = []
    #         for tagg,tkk in zip(tag__,tk__):
    #             temp_pos_tk.append(tkk+'/'+tagg)
    #         line_pos_tk.append('+'.join(temp_pos_tk))
    #     pos_result.append(' '.join(line_pos_tk))
    #     print(line_pos_tk)
    #     result = []
    #    print(p)
    #    print(tk)
    # print(pos_result)
    return pos_result
        

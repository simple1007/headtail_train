import re

def tagged_reduce(tt):
    tags = []
    # print(tt)
    # print(tt)
    # return tt
    for tag_ in tt:
        temp =[]
        #print(tag_)
        tag_ = tag_.replace("+","+_")
        for tag in tag_.split("+"):
            if not tag.startswith("_"):
                temp.append(tag)
                continue
            tag = tag[1:].split('_')[:2]
            # tag = tag.split('_')
        
            tag = [t if t == 'VV' or t == 'VA' else t[0] for t in tag]
            tag = '_'.join(tag)
            if 't_' in tag:
                tag_spl = tag.split('_')
                if len(tag_spl) <= 2:
                    tag = tag.replace('t_','')
                elif tag_spl[1] == 'SN':
                    tag = tag_spl[-3]+'_'+tag_spl[-2]
                else:
                    tag = tag_spl[1]+'_'+tag_spl[2]
            temp.append(tag)
        tags.append("+".join(temp))
    return tags#' '.join(tags)#,tag_spl

result = []
result_tag = open('bibert_pos_tag_tkbigram_one.txt','w',encoding='utf-8')
err = open('error_tkbigram_one.txt','w',encoding='utf-8')
length_count = 0
c = 0
cccc = 0
with open('bibert_re_tkbigram_one.txt',encoding='utf-8') as f:
    with open('bibert_fail_tkbigram_one.txt',encoding='utf-8') as ff:
        for l,tk in zip(f,ff):
            l = l.replace('\n','')
            l = l.replace(' O ',' ')
            # l = l.strip()
            if l.startswith('O '):
                l = list(l)
                l = l[2:]
                l = ''.join(l)
            l = l.strip()
            l = re.sub(' +',' ',l)
            temp_l = l
            l = l.split(' ')
            tk = tk.replace('\n','')
            tk = tk.strip()
            for ll in l:
                if ll.startswith('B_'):
                    result.append(ll[2:])
                else:
                    # print(temp_l)
                    if len(result) == 0:
                        # result.append(ll.replace('I_',''))
                        result.append('FAIL')
                    # elif result[-1] != ll.replace('I_',''):
                    #     result[-1] = 'FAIL'
            tk = tk.replace('\n','')
            temp_tk = tk
            tk = tk.split(' ')
            result_tk = []
            # print(len(tk),len(result))
            cccc += 1
            flag = False
            for index,tk_ in enumerate(tk):
                if '+' in tk_:
                    result_tk.append(index)
                    # print(result)
                    try:
                        if (index+1) <= len(result):
                            result[index] = result[index] + '+' + result[index+1]
                            del result[index+1]
                    except Exception as ex:
                        l = l
                        flag = True
                        # print(l,cccc,ex)
                    # except:
                    #     print(index)
                    #     print('00',result,len(result))
                    #     print('11',tk,len(tk))
            # if len(result) != len(tk):
            #     # print('1',result,len(result))
            #     # print('2',tk,len(tk))
            #     # del result[-2]
            #     err.write('1'+' '.join(result)+'\n')
            #     # err.write('2'+' '.join(tk)+'\n')
            #     err.write('3'+' '.join(l)+'\n')
            #     length_count+=1
            # c += 1
            # if c == 4:
            #     print(' '.join(l),end="\t")
            #     print(' '.join(result),end="\t")
            #     print(' '.join(tagged_reduce(tk)))
            #     exit()
            # print(result)
            #if flag:
            #    print(' '.join(result),end=" ")
            #    print(' '.join(tk))
            result_tag.write(' '.join(result)+'\n')
            result_tag.flush()
            result = []
# import sys
# sys.exit()
result_tag.close()
# print(length_count)

count = 0
length = 0
t = 0
tt = 0
length2 = 0
count2 =0
total_avg = 0
total_avg2 = 0
total_line = 0
eojeal_count = 0
token_count = 0
# from make_one_tag import *
with open('bibert_fail_tkbigram_one.txt',encoding='utf-8') as by:
    with open('bibert_pos_tag_tkbigram_one.txt',encoding='utf-8') as f:
        for l,y in zip(f,by):
            # l = l.replace('<tab>','+')
            total_line += 1
            # l = l[:-1]
            # y = y[:-2]
            # l = l.replace("+"," ")
            l = l.replace('\n','')
            y = y.replace('\n','')
            # l = l.replace('JK ','JKO ')
            y = y.strip()
            l = l.strip()
            l = l.split(' ')
            y = y.split(' ')
            #temp = []
            #for yy in y:
            #    if '+' not in yy:
            #        yy = yy.split('_')
            #        yy = [yyy[0] for yyy in yy]
            #        temp.append('_'.join(yy))
            #    else:
            #        tt = yy.split('+')
            #        y1 = tt[0].split('_')
            #        y1 = [yy1[0] for yy1 in y1]

            #        y1 = '_'.join(y1)

            #        y2 = tt[1].split('_')
            #        y2 = [yy2[0] for yy2 in y2]

            #        y2 = '_'.join(y2)

            #        temp.append(y1+'+'+y2)
            #y = temp
            #print(y)
            # y = y.replace("+","+_")
            y = tagged_reduce(y)
            #print(len(l),len(y))
            #print(y)
            #exit()
            # print(l,y)
            # l = l[:-1]
            # y = y[:-2]
            # if len(l) == len(y):
            #     t+=1
            # if len(l) != len(y):
            #     print('tt',l,len(l))
            #     print('t',y,len(y))
            #     tt+=1
            # print(l==y,"'"+l+"'","'"+y+"'")
            length = len(y)
            for ll,yy in zip(l,y):
                #length+=1
                #if '+' not in yy:
                    # ll.replace('JK ','JKO ')
                if ll == yy:
                        # print(ll,yy)
                    count+=1
                    # else:
                    #     print(ll,yy)
                        # exit()
                #else:
                    # ll.replace('JK ','JKO ')
                    #yy = yy.split('+')
                    #if yy[1].count('_') > 1:
                    #    yy_temp = yy[1].split('_')
                    #    yy[1] = yy_temp[0] + '_' + yy_temp[1]
                    #yy = '+'.join(yy)
                    #if ll == yy:
                    #    count += 1 
                    # else:
                        # print(ll,yy)
                        # exit()
                    # ll = ll.split('+')
                    # if ll[0] == yy[0]:
                    #         # print(l,y)
                    #     count+=1
                    
                    # try:
                    #     if ll[1] == yy[1]:
                    #         count+=1
                    # except:
                    #     l = l
                    #     print(l,y)
            avg = count/length
            total_avg += avg
            #print("1",avg)
            eojeal_count += count
            count = 0
            length = 0
            for yyy in y:
                yyy = yyy.split('+')
                length2 += len(yyy)
            for ll,yy in zip(l,y):
                    
                if '+' not in yy:
                    #length2+=1
                    if ll == yy:
                        # print(l,y)
                        count2+=1
                else:
                    #length2+=2
                    yy = yy.split('+')
                    #if yy[1].count('_') > 1:
                    #    yy_temp = yy[1].split('_')
                    #    yy[1] = yy_temp[0] + '_' + yy_temp[1]
                    # yy = '+'.join(yy)
                    # if ll == yy:
                    #     count += 1 
                    # ll.replace('JK ','JKO ')
                    ll = ll.split('+')
                    # if len(ll) > 1 and ll[1].count('_') >= 1:
                    #     ll_temp = ll[1].split('_')
                    #     ll[1] = ll_temp[0][0] + '_' + ll_temp[1][0]
                        # print(ll[1])
                    # print(ll,yy)
                    if ll[0] == yy[0]:
                            # print(l,y)
                        count2+=1
                    
                    try:
                        if ll[1] == yy[1]:
                            count2+=1
                    except Exception as ex:
                        l = l
                        # print(ex)
            avg2 = count2/length2
            #print("2",avg2)
            total_avg2 += avg2
            token_count += count2
            count2 = 0
            length2 = 0
                        # print(l,y)
# print(t,tt)
print(total_line)
print("eojeal",total_avg/total_line)
print("token",total_avg2/total_line)
print("eojeal_count",eojeal_count)
print("token_count",token_count)

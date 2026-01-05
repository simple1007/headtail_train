import argparse
from turtle import ycor
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Tokenizer Dataset Create")

parser.add_argument("--input", type=str, help="input head-tail dataset",default="C:\\Users\\ty341\\Desktop\\headtail_train\\dataset\\ht_dataset\\modu\\result\\delkiwimorphs.txt")
parser.add_argument("--output", type=str, help="output file name -> output file name:ht -> wauto_x.txt, waut_y.txt",default="wauto")

args = parser.parse_args()

x_file = '_x.txt'
y_file = '_y.txt'
htsep = "hththththt"
input_name = args.input
output_x = args.output + x_file
output_y = args.output + y_file

out_fx = open(output_x,'w',encoding='utf-8')
out_fy = open(output_y,'w',encoding='utf-8')
length = []
from collections import defaultdict
ll = defaultdict(int)
count_line = 0
with open(input_name,'r',encoding='utf-8') as in_f:
    count = 0
    pcount = 0
    for l in in_f:
        if htsep in l:
            pcount += 1
        count += 1
    # print(count,pcount)
    # exit()
    in_f.seek(0)
    print(count)
    for j,_ in enumerate(tqdm(range(count))):
        # if j == 600000:
        #     break
        l = in_f.readline()+' '
        ltt = l 
        # if htsep not in l:
            # continue
        #l = l.strip()
        l = l.replace("+","_")
        import copy
        ltmp = []#copy.deepcopy(l)
        for l_ in l.split():
            l_ = l_.strip(htsep)
            ltmp.append(l_)
        l = " ".join(ltmp)+" "
        l = l.replace(htsep,"+")
        # 그러니까▁그▁집▁구조
        if l.startswith("그러니까") and "그" in l and "집" in l and "구조" in l:
            print(l)
            print(ltmp)
            print(ltt)
        # if "_" in l:
        #     print(l)
        # l = l.replace("_","")
        l = l.replace('\n','')
        # if len(l) <= 5:
        #     continue
        l = l.split(' ')
        
        data = []
        line = ''
        label = []
        for index,ll in enumerate(l):
            if '+' in ll:
                t = ll.split('+')
                for i in range(len(t[0])):
                    if i == 0:
                        label.append('1')
                    else:
                        label.append('0')
                # label[0]
                label.append('2')
                for i in range(len(t[1])-1):
                    label.append('0')    
                for tt in t:    
                    data.append(tt)
                label.append('0') 
                if index == len(l) -1:
                    line += ''.join(data)
                else:
                    line += ''.join(data) + '▁'
                data = []
            else:
                if index == len(l) -1:
                    line += ll
                else:
                    line += ll + '▁'

                for i in range(len(ll)):
                    if i == 0:
                        label.append('1')
                    else:
                        label.append('0')
                if index != len(l) -1:
                    label.append('0')
        # print()
        # print()
        lines = []
        labels = []
        for la,li in zip(label,line):
            if li != "▁":
                lines.append(li)
                labels.append(la)
                # print(la,end="/")
                # print(li)
        # exit()
        # if len(lines) != len(labels):
        #     print(lines)
        #     print(labels)
        #     print(len(lines),len(labels))
        #     exit()
        # else:
        #     print(lines)
        #     print(labels)
        #     print(len(lines),len(labels))
            
        count_line += 1

        # if count_line > 10:
        #     exit()
        out_fy.write(''.join(label)+'\n')
        out_fx.write(line+'\n')
        
        length.append(len(line))

print(max(length),sum(length)/len(length))
import matplotlib.pyplot as plt
# from matpl
import numpy as np
x = sorted(length,reverse=True)
count = defaultdict(int)
for xx in x:
    count[xx] += 1
plt.bar(count.keys(),count.values())
plt.show()
out_fx.close()
out_fy.close()

import sys
sys.exit()

import pickle
fout = open('autospace.pkl','wb')
with open('autospace.txt','r') as ff:
    with open('autospace_line.txt','r') as f:
        for l,ll in zip(f,ff):
            l = l.replace('\n','')
            ll = ll.replace('\n','')
            #if len(l) != len(ll):
            #    print('error',len(l),"e",len(ll))
            data = []
            for l1,l2 in zip(l,ll):
                a = tuple([l1,l2])
                data.append(a)
            if data[-1][0] == ' ':
                data[:-1]
            pickle.dump(data,fout)

fout.close()

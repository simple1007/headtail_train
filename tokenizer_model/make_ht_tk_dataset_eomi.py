import argparse
from turtle import ycor
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Tokenizer Dataset Create")

parser.add_argument("--input", type=str, help="input head-tail dataset",default="tk_train_data.txt")
parser.add_argument("--output", type=str, help="output file name -> output file name:ht -> httk_x.txt, httk_y.txt",default="httk")

args = parser.parse_args()

x_file = '_x.txt'
y_file = '_y.txt'

input_name = args.input
output_x = args.output + x_file
output_y = args.output + y_file

out_fx = open(output_x,'w',encoding='utf-8')
out_fy = open(output_y,'w',encoding='utf-8')
length = []
from collections import defaultdict
ll = defaultdict(int)
with open(input_name,'r',encoding='utf-8') as in_f:
    count = 0
    for l in in_f:
        count += 1
    in_f.seek(0)
    print(count)
    for _ in tqdm(range(count)):
        l = in_f.readline()+' '
        #l = l.strip()
        l = l.replace('\n','')
        l = l.split(' ')
        
        data = []
        line = ''
        label = []
        for index,ll in enumerate(l):
            if '+' in ll:
                t = ll.split('+')
                for i in range(len(t[0].split('/')[0])-1):
                    label.append('0')
                label.append('4')
                label.append('2')
                for i in range(len(t[1].split('/')[0])-1):
                    label.append('0')    
                for tt in t:    
                    data.append(tt.split('/')[0])
                label.append('1') 
                if index == len(l) -1:
                    line += ''.join(data)
                else:
                    line += ''.join(data) + '▁'
                data = []
            else:
                if index == len(l) -1:
                    line += ll.split('/')[0]
                else:
                    line += ll.split('/')[0] + '▁'

                for i in range(len(ll.split('/')[0])):
                    label.append('0')
                if index != len(l) -1:
                    label.append('1')
        out_fy.write(''.join(label[:-1])+'\n')
        out_fx.write(line[:-1]+'\n')
        
        length.append(len(line[:-1]))

print(max(length),sum(length)/len(length))
import matplotlib.pyplot as plt
# from matpl
import numpy as np
x = sorted(length,reverse=True)
plt.plot(np.arange(50000),x[:50000])
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

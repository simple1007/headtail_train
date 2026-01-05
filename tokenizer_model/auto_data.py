resx = open("autospace_x.txt","w",encoding="utf-8")
resy = open("autospace_y.txt","w",encoding="utf-8")

auto_vocab = {}

kcc = [[0,0]]
k = [[0,0]]

writ = [[0,0]]
w = [[0,0]]

count = 0
datapath = "C:\\Users\\ty341\\OneDrive\\Desktop\\dataset\\"
for idx, dataset in enumerate([datapath+"KCC150_Korean_sentences_UTF8.txt",datapath+"written_raw.txt"]):
    with open(dataset,'r',encoding='utf-8') as x_f:
        for l in x_f:
            l = l.strip()
            y = [0 for i in range(len(l))]
            
            for idx,uni in enumerate(l):
                if uni == " ":
                    y[idx] = 1
            
            y = list(map(str,y))
            y = "".join(y)
            
            y = y.replace("01","1")
            l = l.replace(" ","")
            if len(l) != len(y):
                print(l,y)
            for uni,lb in zip(l,y):
                resx.write(f"{uni}")
                resy.write(f"{lb}")        
            resx.write("\n")
            resy.write("\n")
            count += 1
            
resx.close()
resy.close()

resy = open("autospace_y.txt",encoding="utf-8")

seed = [i for i in range(count)]

import random
random.shuffle(seed)

def shuffle_file(s,ls,fx):
    for s_ in s:
        fx.write(ls[s_])
    
    fx.close()

fx = open("autox.txt","w",encoding="utf-8")
resx = open("autospace_x.txt",encoding="utf-8")
x = resx.readlines()
resx.close()
shuffle_file(seed,x,fx)

fy = open("autoy.txt","w",encoding="utf-8")
resy = open("autospace_y.txt",encoding="utf-8")
y = resy.readlines()
resy.close()
shuffle_file(seed,y,fy)
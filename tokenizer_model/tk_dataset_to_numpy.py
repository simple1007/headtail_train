import pickle
import numpy as np
import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description="Tokenizer Dataset to Train numpy dataset Create")
parser.add_argument("--inputX",type=str,help="train X data path",default="httk_x.txt")
parser.add_argument("--inputY",type=str,help="train Y data path",default="httk_y.txt")
parser.add_argument("--BATCH",type=int,help="Train Data BATCH SIZE",default=50)
parser.add_argument("--MAX_LEN",type=int,help="Sequence Data MAX LENGTH",default=300)

args = parser.parse_args()
inputX = args.inputX
inputY = args.inputY
MAX_LEN = args.MAX_LEN
BATCH = args.BATCH
# print(args)
length = []
bigram = {}
bigram['[PAD]'] = 0
bigram['[UNK]'] = 1
bigram["[SOS]"] = 2
bigram['[EOS]'] = 3
bigram["[MASK]"] = 4
bigram_index = 5
lstm_vocab = {}
lstm_vocab['[PAD]'] = 0
lstm_vocab['[UNK]'] = 1
lstm_vocab["[SOS]"] = 2
lstm_vocab['[EOS]'] = 3
lstm_vocab["[MASK]"] = 4
lstm_index = 5

htsep = "hththththt"
from collections import defaultdict
c = defaultdict(int)

# c[bi] += 1
# ll = 0
# with open(inputX,'r',encoding='utf-8') as x_f:
#     for l in x_f:
#         l = l.strip()
#         for i in range(len(l)-1):
#             bi = l[i:i+2]
#             c[bi] += 1
#         ll += 1
#         print(f"\r{ll}",end="")


with open('vocab.txt','r',encoding='utf-8') as f:
    for index,l in enumerate(f):
        l = l.replace('\n','')
        if len(l) == 1:
            lstm_vocab[l] = lstm_index
            lstm_index += 1
# for idx,h in enumerate(range(ord("가"),ord("힣")+1)):
with open("httk_x.txt",encoding="utf-8") as f:
    for l in f:
        l = l.strip()
        for ll in l:
            char = ll
            if char not in lstm_vocab:
                lstm_vocab[char] = lstm_index
                lstm_index += 1
        # char = chr(h)
        # # print(char)
        # if char not in lstm_vocab:
        #     lstm_vocab[char] = lstm_index
        #     # eum_vocab_dict[char] = lstm_index
        #     lstm_index += 1           
with open('lstm_vocab.pkl','wb') as f:
    pickle.dump(lstm_vocab,f)

with open('lstm_vocab.pkl','rb') as f:
    lstm_vocab = pickle.load(f)

with open(inputX,'r',encoding='utf-8') as x_f:
    with open(inputY,'r',encoding='utf-8') as y_ff:
        count = 0
        for l in x_f:
            count += 1
        x_f.seek(0)
        
        X = []
        Y = []
        lstm_X = []
        lstm_Y = []
        BI = []
        file_num = 0
        
        for _ in tqdm(range(count)):#zip(x_f,y_ff):
            x = x_f.readline()
            y = y_ff.readline()
            x = x.replace('\n','')
            y = y.replace('\n','')
            length.append(len(x))
            if True:#len(x) <= MAX_LEN and len(y) <= MAX_LEN:
                bi_npy = []
                for i in range(len(x)-1):
                    bi = x[i:i+2]
                    
                    if bi not in bigram:# and c[bi] > 2:
                        bigram[bi] = bigram_index
                        bigram_index += 1
                    # if c[bi] <= 2:
                    #     bi = "[UNK]"
                    bi_npy.append(bigram[bi])
                bi_npy = bi_npy[:MAX_LEN] + [0] * (MAX_LEN - len(bi_npy))
                lstm_x = [lstm_vocab[i] if i in lstm_vocab else 1 for i in x]
                lstm_x = lstm_x[:MAX_LEN] + [0] * (MAX_LEN - len(lstm_x))
                # print(x,bigram) 
                # exit()
                y = [int(i) for i in y]
                lstm_y = y[:MAX_LEN] + [3] * (MAX_LEN -len(y))

                lstm_x = np.array(lstm_x)
                lstm_y = np.array(lstm_y)
                # print(lstm_x.shape)
                # print(lstm_y.shape)    
                lstm_X.append(lstm_x)
                lstm_Y.append(lstm_y)

                bi_npy = np.array(bi_npy)
                # print(bi_npy.shape)
                # exit()
                BI.append(bi_npy)
            if len(lstm_X) == BATCH:
                lstm_X = np.array(lstm_X)
                lstm_Y = np.array(lstm_Y)
                BI = np.array(BI)
                # print(BI.shape)
                # np.save('token_data/%05d_lstm_x' % file_num,lstm_X)
                # np.save('token_data/%05d_lstm_y' % file_num,lstm_Y)
                # np.save('token_data/%05d_bigram' % file_num, BI)
                
                BI = []
                lstm_X = []
                lstm_Y = []
                file_num += 1

with open('tokenizer_eum_length.pkl','wb') as f:
    pickle.dump(length,f)
with open('bigram_vocab.pkl','wb') as f:
    pickle.dump(bigram,f)
    # print(bigram)

# print(c)
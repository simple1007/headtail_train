from tokenization_kobert import KoBertTokenizer
from tqdm import tqdm

import numpy as np
import argparse

parser = argparse.ArgumentParser("POS Tagger Numpy Data Create")

parser.add_argument("--MAX_LEN",type=int,help="MAX Sequnce Length",default=200)
parser.add_argument("--BATCH",type=int,help="BATCH Size",default=50)
parser.add_argument("--input_morph",type=str,help="Input line morph file path",default="./kcc150_morphs.txt")
parser.add_argument("--input_tag",type=str,help="Input line tag file path",default="./kcc150_tag.txt")
parser.add_argument("--stop_batch_num",type=int,help="Make Stop Batch Num",default=8000)

tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')

tag = ''#c.pop(0)
temp = ''
count = 0
tag_dict = {}
index = 0
tag_dict['[CLS]'] = 0
tag_dict['[SEP]'] = 1
tag_dict['O'] = 2
tag_dict['[PAD]'] = 3
index = 4
tag_re = []

args = parser.parse_args()

max_len = args.MAX_LEN#200
BATCH = args.BATCH
input_tag = args.input_tag
input_morph = args.input_morph
morph_f = open(input_morph,'r',encoding='utf-8')#'/data/KCC150/KCC150_equal_morph/morphs.txt','r')
tag_f = open(input_tag,'r',encoding='utf-8')#'/data/KCC150/KCC150_equal_morph/tag.txt','r')

X = []
Y = []
TK_BI = []
bidata = []
unidata = []
count_data = 0
char_len = 5
char_count = 0
length = []
from collections import defaultdict
count_word = defaultdict(int)
count_tag = defaultdict(int)
bigram = {}
bigram['[UNK]'] = 0
bigram['[PAD]'] = 1
biindex = 2

unigram = {}
unigram['[UNK]'] = 0
unigram['[PAD]'] = 1
uniindex = 2

token_bi_dict = {}
token_bi_dict['[UNK]'] = 0
token_bi_dict['[PAD]'] = 1
tk_index = 2
train_data = open('train_data.txt','w')

count_length = 0

for l in morph_f:
    count_length += 1
morph_f.seek(0)

def tagged_reduce(tag):
    # print(tag__)
    # print("split",tag__)
    tag = tag__.pop(0)
    if not tag.startswith("_"):
        return tag
    tag = tag[1:]

    tag = tag.split('_')[:2]
    
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
    
    return tag#,tag_spl
fail_line = 0
for line in tqdm(range(count_length)):
# for line,(morph_,tag_) in enumerate(zip(morph_f,tag_f)):
    morph_ = morph_f.readline()
    tag_ = tag_f.readline()

    morph_temp = morph_
    
    morph_ = morph_.replace('\n','').replace('+',' ')
    # morph_ = morph_[:-1]
    tag_ = tag_.replace('\n','').replace('+',' _')

    # tag_ = tag_[:-1]
    bi_npy = []
    uni_npy = []
    token_bi = []

    for i in morph_.split(' '):
        count_word[i] += 1
    mm = tokenizer.tokenize('[CLS] '+ morph_+' [SEP]')
    tag__ = tag_.split(' ')
    if len(tag__) != len(morph_.split()):
        fail_line += 1
        continue
    # for t_t,f_f in zip(tag__,morph_.split()):
    #     print(t_t+"*"+f_f,end=" ")
    # print()
    length.append(len(mm))
    mm = mm[1:-1]
    if len(morph_) < char_len:
        char_count += 1
        continue
    if True:
        # try:
            for i in mm:
                # if len(tag__)>0:
                #     print("ori",tag__[0])
                if '▁' in i and i !='▁':
                   
                    tag = tagged_reduce(tag)
                    # tag = tag__.pop(0)
                    # tag = tag.split('_')
                    # tag = [t[0] for t in tag]
                    # tag = '_'.join(tag)
                    # if 't_' in tag:
                    #     tag_spl = tag.split('_')
                    #     if len(tag_spl) <= 2:
                    #         tag = tag.replace('t_','')
                    #     elif tag_spl[1] == 'SN':
                    #         tag = tag_spl[-3]+'_'+tag_spl[-2]
                    #     else:
                    #         tag = tag_spl[1]+'_'+tag_spl[2]

                    count = 0
                    temp = '*'
                    tag_re.append('B_'+tag)
                elif temp!='▁' and '▁' not in i:
                    tag_re.append('I_'+tag)
                if i == '▁':
                    tag = tagged_reduce(tag)
                    # tag = tag__.pop(0)
                    # tag = tag.split('_')
                    # tag = [t[0] for t in tag]
                    # tag = '_'.join(tag)
                    # if 't_' in tag:
                    #     tag_spl = tag.split('_')
                    #     if len(tag_spl) <= 2:
                    #         tag = tag.replace('t_','')
                    #     elif tag_spl[1] == 'SN':
                    #         tag = tag_spl[-3]+'_'+tag_spl[-2]
                    #     else:
                    #         tag = tag_spl[1]+'_'+tag_spl[2]

                    temp = i
                    count = 0
                    tag_re.append('O')
                    continue
                elif temp == '▁' and '▁' not in i:
                    if count == 0:
                        tag_re.append('B_'+tag)
                        count+=1
                    else:
                        tag_re.append('I_'+tag)
                
                if tag != '':
                    if 'B_'+tag not in tag_dict:
                        tag_dict['B_'+tag] = index
                        index+=1
                    if 'I_'+tag not in tag_dict:
                        tag_dict['I_'+tag] = index
                        index+=1
                    count_tag[tag]+=1
            if len(mm)+2 <= max_len:
                mmm = ['[CLS]'] + mm + ['[SEP]']
                mmm = mmm+['[PAD]'] * (max_len - len(mmm))
                x = tokenizer.convert_tokens_to_ids(mmm)

                for token_i in range(len(mmm)-1):
                    bii = mmm[token_i:token_i+2]
                    bii = '+'.join(bii)
                    if bii not in token_bi_dict:
                        token_bi_dict[bii] = tk_index
                        tk_index += 1
                    token_bi.append(token_bi_dict[bii])

                token_bi = token_bi + [1] * (max_len-len(token_bi)) 
                TK_BI.append(token_bi)
                x = np.array(x)
                X.append(x)

                y = [tag_dict[t] for t in tag_re]

                y = [0] + y + [1]
                if len(mm)+2 != len(y):
                    print(mm,len(mm),y,len(y))
                y = y + [3] * (max_len - len(y))
                
                y = np.array(y)
                if y.shape[0] > max_len or y.shape[0] > max_len:
                    print(y.shape,x.shape,len(tag_re),len(['[CLS]'] + mm + ['[SEP]']))
                Y.append(y)
                if len(x) != len(y):
                    print(x,len(x),y,len(y))
                bidata.append(bi_npy)
                unidata.append(uni_npy)
                if (count_data+1) <= 4000:
                    train_data.write(morph_temp)

            if len(X) == BATCH:
                X = np.array(X)
                Y = np.array(Y)
                TK_BI = np.array(TK_BI)
                if len(X.shape) < 2 :
                    print(X.shape,Y.shape)
                np.save('./kcc150_data/%05d_x' % count_data,X)
                np.save('./kcc150_data/%05d_y' % count_data,Y)

                np.save('./kcc150_data/%05d_tk_bigram' % count_data,TK_BI)
                count_data+=1

                X = []
                Y = []
                TK_BI = []
                bidata = []
                unidata = []

                if (count_data+1)%args.stop_batch_num==0:#8000 == 0:
                    print("{0} stop batch num -> stop".format(args.stop_batch_num))
                    break
        # except IndexError as e:
        #     # print(len(morph_.split()),len(tag_.split()))
        #     # exit()
        #     print(e,line,tag_re,mm)

    tag_re = []
    temp = ''
    tag = ''

train_data.close()

import pickle
with open('kcc150_all_length.pkl','wb') as f:
    pickle.dump(l,f)
with open('kcc150_all_count_word.pkl','wb') as f:
    pickle.dump(count_word,f)
with open('kcc150_all_tag_dict.pkl','wb') as f:
    pickle.dump(tag_dict,f)
with open('kcc150_all_bigram.pkl','wb') as f:
    pickle.dump(bigram,f)
with open('kcc150_all_unigram.pkl','wb') as f:
    pickle.dump(unigram,f)
with open('count_tag.pkl','wb') as f:
    pickle.dump(count_tag,f)
with open('kcc150_all_tokenbi.pkl','wb') as f:
    pickle.dump(token_bi_dict,f)

print("fail_line",fail_line)
print(len(tag_dict))
import pickle
import numpy as np
max_len = 260
length = []
bigram = {}
bigram['[PAD]'] = 0
bigram['[UNK]'] = 1
bigram_index = 2
if True:
    eum_vocab_dict = {}
    lstm_vocab = {}
    lstm_vocab['[PAD]'] = 0
    lstm_vocab['[UNK]'] = 1
    lstm_index = 2
    # index = 0
    with open('vocab.txt','r',encoding='utf-8') as f:
        for index,l in enumerate(f):
            l = l.replace('\n','')
            if len(l) == 1:
                eum_vocab_dict[l] = index
                lstm_vocab[l] = lstm_index
                lstm_index += 1
    # index+=1
    eum_vocab_dict['[UNK]'] = 0 
    with open('eum_vocab.pkl','wb') as f:
        pickle.dump(eum_vocab_dict,f)
    with open('lstm_vocab.pkl','wb') as f:
        pickle.dump(lstm_vocab,f)
else:
    with open('eum_vocab.pkl','rb') as f:
        eum_vocab_dict = pickle.load(f)
    with open('lstm_vocab.pkl','rb') as f:
        lstm_vocab = pickle.load(f)

    with open('kcc150_autospace_line.txt','r',encoding='utf-8') as x_f:
        with open('kcc150_autospace.txt','r',encoding='utf-8') as y_ff:
            X = []
            Y = []
            lstm_X = []
            lstm_Y = []
            BI = []
            file_num = 0
            for x,y in zip(x_f,y_ff):
                x = x.replace('\n','')
                y = y.replace('\n','')
                length.append(len(x))
                if len(x) <= 258 and len(y) <= 258:
                    bi_npy = []
                    for i in range(len(x)-1):
                        bi = x[i:i+2]
                        if bi not in bigram:
                            bigram[bi] = bigram_index
                            bigram_index += 1
                        bi_npy.append(bigram[bi])
                            
                    bi_npy = bi_npy + [0] * (max_len - len(bi_npy))
                    lstm_x = [lstm_vocab[i] if i in lstm_vocab else 1 for i in x]
                    lstm_x = lstm_x + [0] * (max_len - len(lstm_x))
                    x = [eum_vocab_dict[i] if i in eum_vocab_dict else 0 for i in x]
                    x = [2] + x + [3]
                    x = x + [1] * (max_len - len(x))
                    
                    y = [int(i) for i in y]
                    lstm_y = y + [3] * (max_len-len(y))
                    y = [4] + y + [5] 
                    y = y + [3] * (max_len-len(y))

                    x = np.array(x)
                    y = np.array(y)
                    bi_npy = np.array(bi_npy)
                    lstm_x = np.array(lstm_x)
                    lstm_y = np.array(lstm_y)
                    X.append(x)
                    Y.append(y)
                    BI.append(bi_npy)
                    lstm_X.append(lstm_x)
                    lstm_Y.append(lstm_y)
                if len(X) == 50:
                    X = np.array(X)
                    Y = np.array(Y)
                    lstm_X = np.array(lstm_X)
                    BI = np.array(BI)
                    np.save('token_data/%05d_x' % file_num,X)
                    np.save('token_data/%05d_y' % file_num,Y)
                    np.save('token_data/%05d_lstm_x' % file_num,lstm_X)
                    np.save('token_data/%05d_lstm_y' % file_num,lstm_Y)
                    np.save('token_data/%05d_bigram' % file_num,BI)
                    BI = []
                    X = []
                    Y = []
                    lstm_X = []
                    lstm_Y = []
                    file_num += 1

with open('tokenizer_eum_length.pkl','wb') as f:
    pickle.dump(length,f)
with open('bigram_vocab.pkl','wb') as f:
    pickle.dump(bigram,f)

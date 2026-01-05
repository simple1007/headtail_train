import pickle
import re
import numpy as np
import os
# import tensorflow as tf
import torch.nn as nn
from tokenization_kobert import KoBertTokenizer
from tag_merge import tag_merge
from transformers import ElectraTokenizer
from transformers import BertModel, DistilBertModel, AdamW

with open("posvocab.pkl","rb") as f:
    posvocab = pickle.load(f)

max_len = 200

root = os.environ['HT']

with open(root + os.sep + 'kcc150_all_tag_dict.pkl','rb') as f:
    tag_dict = pickle.load(f)
    tag_dict = { v:k for k,v in tag_dict.items() }
with open(root + os.sep + 'kcc150_all_tokenbi.pkl','rb') as f:
    bigram = pickle.load(f)

tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
# tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
def make_input(x_t):
    masks = []
    segments = []

    for x in x_t:
		# for d in x:
        d_ = [t for t in x if t != 1]
		# print(len(d_))
        mask = [1]*len(d_) + [0] * (max_len-len(d_))
        masks.append(mask)
        segment = [0] * max_len
        segments.append(segment)
		# masks = masks * 100
		# segments = segments * 100
    masks = np.array(masks)
    segments = np.array(segments)

    return masks,segments

def w2i(x_t):
    res = []
	# x_t = x_t[:200]
    x = tokenizer(x_t,max_length = max_len,padding="max_length",truncation=True, return_token_type_ids=True,return_attention_mask=True)
    # print(len(a),type(a))
    # masks
    # print(x)
    # exit()
    # return np.array(posvocab.make_batch2([x_t])), np.array([[0]]), np.array([[0]])
    return np.array(x['input_ids']),np.array(x['attention_mask']),np.array(x['token_type_ids'])
    # for i in x_t:
	# 	x = i.replace('  ',' ').replace('+',' ')
	# 	#x = list(x)
	# 	#x = [vocab[e] if e in vocab else vocab['[UNK]'] for e in x ] 
	# 	x = tokenizer.tokenize('[CLS] '+ x+' [SEP]') 
	# 	# print(x,len(x))
        
	# 	x = x[:200]
    #     # print(x,len(x))
	# 	x = tokenizer.convert_tokens_to_ids(x)#[2]+x+[3]

	# 	x = x + [1] * (max_len-len(x))
	# 	res.append(x)

	# x = np.array(res)
    return x
import threading

tag_re = [[],[],[],[]]
def postag(data,line_data):
    thread1 = threading.Thread(target=argmax,args=(data,line_data,0))
    thread2 = threading.Thread(target=argmax,args=(data,line_data,1))
    thread3 = threading.Thread(target=argmax,args=(data,line_data,2))
    thread4 = threading.Thread(target=argmax,args=(data,line_data,3))
    
    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    while True:
        if not thread1.is_alive() and not thread2.is_alive() and not thread3.is_alive() and not thread4.is_alive():
            # print("thread end")
            break
    return tag_re[0] + tag_re[1] + tag_re[2] + tag_re[3] 

# from keras import backend as K

# K.clear_session()

# class POSTagger:
#     def __init__(self):
#         self.session = K.get_session()
#         self.graph = tf.Graph()#tf.get_default_graph()#tf.get_default_graph()
#         # tf.get_default_graph()
#         self.pos_model = tf.keras.models.load_model('tkbigram_one_first_alltag_bert_tagger_tagone.model',compile=False)
#         self.graph.finalize()

#     # def preprocessing(self,data):
#     #     return self.session.run(data)
#     def predict(self,x):
#         with self.session.as_default():
#             with self.graph.as_default():
#                 pred = self.pos_model.predict(x)
#         return pred
# pos_model2 = tf.keras.models.load_model('tkbigram_one_first_alltag_bert_tagger_tagone.model',compile=False)


def start(x1,x2,tkline1,tkline2):
    pos_model1 = POSTagger()
    pos_model2 = POSTagger()
    thread1 = threading.Thread(target=predict_th,args=(pos_model1,x1,tkline1,0))
    thread2 = threading.Thread(target=predict_th,args=(pos_model2,x2,tkline2,1))  

    thread1.start()
    thread2.start()

    while True:
        if not thread1.is_alive() and not thread2.is_alive():
            break
    return tag_re[0] + tag_re[1]  

def predict_th(pos_model,x,tk_line,i,lite=False):
        # sess = tf.get_default_session()
    BI = []
    for xx in x:
        bi_npy = []
        temp = tokenizer.tokenize('[CLS] '+ xx +' [SEP]')
        temp = temp[:max_len]
        for token_i in range(len(temp)):
            bii = temp[token_i:token_i+2]
            bii = '+'.join(bii)
            if bii in bigram:
                bi_npy.append(bigram[bii])
            else:
                bi_npy.append(bigram['[UNK]'])
        bi_npy = bi_npy + [1] * (max_len - len(bi_npy))
        bi_npy = np.array(bi_npy)
        BI.append(bi_npy)

    x_temp = x
    x_len = len(x) + 2
    x = w2i(x)
	# print(x)
	# x = [x]
	# x = x * 100
	# print(x)
	# print(x.shape)

    masks,segments = make_input(x)
    x = np.array(x)
    bi_temp = np.array(BI)

	# print(masks.shape,segments.shape,x.shape)
	# import sys
	# sys.exit()
	# return x,x_temp,x_len
    # with graph.as_default():
    #     with sess.as_default():
    # result = model.predict([bi_temp,x,masks,segments])
    if lite:
        result = pos_model.predict([bi_temp,x,masks])
    else:
        result = pos_model.predict([bi_temp,x,masks,segments])
    result_ = tf.argmax(result,axis=2).numpy()

    result_ = result_[:,1:]
	#return tagging
    tagging = []
    for rr in result_:
        re_tag = []
        for rr_index,i in enumerate(rr):
			#if tag_dict[i] == '[PAD]' and rr_index < len(rr) -1 and tag_dict[rr[rr_index+1]]=='[PAD]':
			#	break
            re_tag.append(tag_dict[i])
        tags = ' '.join(re_tag)
        tags = tags.replace(' [SEP]','').replace('[PAD]','').replace(' [PAD]','')
        tags = re.sub(' +',' ',tags)
        tags = tags.strip()
        # tag_re[i].append(tags)
        tagging.append(tags)

    pos_res=tag_merge(tagging,tk_line)
    tag_re[i] = pos_res

def argmax(x,tkline,index):
    x = x[index]
    tkline = tkline[index]
    x = np.array(x)
    # print(x.shape)
    result_ = tf.argmax(x,axis=2).numpy()

    result_ = result_[:,1:]
	#return tagging
    tagging = []
    for rr in result_:
        re_tag = []
        for rr_index,i in enumerate(rr):
			#if tag_dict[i] == '[PAD]' and rr_index < len(rr) -1 and tag_dict[rr[rr_index+1]]=='[PAD]':
			#	break
            re_tag.append(tag_dict[i])
        tags = ' '.join(re_tag)
        tags = tags.replace(' [SEP]','').replace('[PAD]','').replace(' [PAD]','')
        tags = re.sub(' +',' ',tags)
        tags = tags.strip()
        # tag_re[i].append(tags)
        tagging.append(tags)

    pos_res=tag_merge(tagging,tkline)
    tag_re[index] = pos_res

# def mp_handler(data):
#     p = multiprocessing.Pool(len(data)) # number of parallel threads
#     r=p.map(argmax, data)
#     return r
import time
def predictbi(model,x,tk_line,lite=False,verbose=0):
    	# x = x.replace('  ',' ').replace(' ','▁')
	# x = list(x)
    BI = []
    start = time.time()
    x,masks,segments = w2i(x)
    # print(time.time()-start)
    # BI = x
    # x_ = tokenizer.convert_ids_to_tokens(x)

    # print(tokenizer.ids_to_tokens,tokenizer.ids_to_tokens(len(tokenizer.ids_to_tokens)-1),len(tokenizer.ids_to_tokens))
    # exit()
    start = time.time()
    for xx in x:
        bi_npy = []
        # temp = tokenizer.tokenize('[CLS] '+ xx +' [SEP]')
        # temp = xx#temp[:200]
        temp = tokenizer.convert_ids_to_tokens(xx)
        # exit()
        # temp  = []
        # for xxx in xx:
        #     token = tokenizer.ids_to_tokens[xxx]
        #     if token != '[PAD]':
        #         temp.append(token)
        #     else:
        #         break
        # print(temp)
        # exit()
        for token_i in range(len(temp)):
            bii = temp[token_i:token_i+2]
            bii = '+'.join(bii)
            if bii in bigram:
                bi_npy.append(bigram[bii])
            else:
                bi_npy.append(bigram['[UNK]'])
        bi_npy = bi_npy + [1] * (max_len - len(bi_npy))
        bi_npy = np.array(bi_npy)
        BI.append(bi_npy)
    # print(time.time()-start)
    # exit()
    x_temp = x
    x_len = len(x) + 2
    # x = w2i(x)
    
    # print(x)
	# x = [x]
	# x = x * 100
	# print(x)
	# print(x.shape)

    # masks,segments = make_input(x)
    # x = np.array(x)
    bi_temp = np.array(BI)

	# print(masks.shape,segments.shape,x.shape)
	# import sys
	# sys.exit()
	# return x,x_temp,x_len
    # with graph.as_default():
    #     with sess.as_default():
    if lite:
        result = model.predict([bi_temp,x,masks],verbose=verbose,workers=5,use_multiprocessing=True)
    else:
        result = model.predict([bi_temp,x,masks,segments],verbose=verbose,workers=5,use_multiprocessing=True)

    # data = [[],[],[],[]]#range(0,result.shape[0])
    # line_data = [[],[],[],[]]
    # temp = []
    # line_temp = []
    # c = 0
    # for index,(r,tk_l) in enumerate(zip(result,tk_line)):
    #     temp.append(r)
    #     line_temp.append(tk_l)
    #     if ((index+1) % 125) == 0:
    #         # data.append(temp)
    #         # print(data[0].shape,c)
    #         data[c] = temp
    #         line_data[c] = line_temp
    #         temp = []
    #         line_temp = []
    #         c+=1
    # if len(temp) > 0:
    #     data[c] = temp
    #     line_data[c] = line_temp
    # print(np.array(data[0]).shape,np.array(data[1]).shape)
    # return postag(data,line_data)
    # import sys
    # sys.exit()
    tagging = []
	# #return tagging
    result_ = tf.argmax(result,axis=2).numpy()
    # data = []#range(0,result.shape[0])
    # temp = []
    # for index,r in enumerate(result):
    #     temp.append(r)
    #     if index+1 % 50 == 0:
    #         data.append(temp)
    #         temp = []
    # if len(temp) > 0:
    #     data.append(temp)
    # # result_ = mp_handler(data)
	# #return tagging
    result_ = result_[:,1:]
	# # #return tagging
    for rr in result_:
        re_tag = []
        for rr_index,i in enumerate(rr):
			#if tag_dict[i] == '[PAD]' and rr_index < len(rr) -1 and tag_dict[rr[rr_index+1]]=='[PAD]':
			#	break
            re_tag.append(tag_dict[i])
        tags = ' '.join(re_tag)
        tags = tags.replace(' [SEP]','').replace('[PAD]','').replace(' [PAD]','')
        tags = re.sub(' +',' ',tags)
        tags = tags.strip()
        tagging.append(tags)
        # print(tags)
    # return tagging
    pos_res=tag_merge(tagging,tk_line)
    # pos_res = ' '.join(tagging)
    return pos_res

import torch

def predictbi_pt(model,x,tk_line,device,lite=False,verbose=0):
    	# x = x.replace('  ',' ').replace(' ','▁')
	# x = list(x)
    with torch.no_grad():
        BI = []
        start = time.time()
        x_temp = x
        x,masks,segments = w2i(x)
        start = time.time()
        for xx in x:
            bi_npy = []
            temp = tokenizer.convert_ids_to_tokens(xx)
        
            for token_i in range(len(temp)):
                bii = temp[token_i:token_i+2]
                bii = '+'.join(bii)
                if bii in bigram:
                    bi_npy.append(bigram[bii])
                else:
                    bi_npy.append(bigram['[UNK]'])
            bi_npy = bi_npy + [1] * (max_len - len(bi_npy))
            bi_npy = np.array(bi_npy)
            BI.append(bi_npy)
    
        
        x_len = len(x) + 2
        
        bi_temp = np.array(BI)

        
        # if lite:
        #     result = model.predict([bi_temp,x,masks],verbose=verbose,workers=5,use_multiprocessing=True)
        # else:
        #     result = model.predict([bi_temp,x,masks,segments],verbose=verbose,workers=5,use_multiprocessing=True)


        bi_temp = torch.tensor(bi_temp).to(device)
        x = torch.tensor(x).to(device)
        masks = torch.tensor(masks).to(device)
        segments = torch.tensor(segments).to(device) 
        
        result = model(bi_temp,x,masks,segments)

        tagging = []
        # print(result.shape)
        _,result_ = torch.topk(result,k=1,dim=-1)#tf.argmax(result,axis=2).numpy()
        # print(result_.shape)
        result_ = result_.view(result_.shape[0],result.shape[1]).cpu().numpy()
        result_ = result_[:,1:]

        # re_tag = [
        
        for index,xt in enumerate(result_):
            # print(x_temp[index])

            # print(x_temp[index])
            x_temp[index] = re.sub(' +',' ',x_temp[index])
            x_temp[index] = x_temp[index].replace("+"," ")
            xtt = tokenizer(x_temp[index])["input_ids"]
            # print(xt,len(xt))
            re_tag_ = xt[:len(xtt)-2]
            re_tag = []
            # print(re_tag_.shape)
            # print()
            for rr_index in re_tag_:
                # print(rr_index)
                re_tag.append(tag_dict[rr_index])
            tagging.append(' '.join(re_tag))
        
        pos_res=tag_merge(tagging,tk_line)
        # pos_res = ' '.join(tagging)
    return pos_res
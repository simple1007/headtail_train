# import tensorflow as tf
import numpy as np
import pickle
import torch
from tokenization_kobert import KoBertTokenizer
from train_tkbigram_pos_tagger_pytorch_peft_bi import PosTaggerModelPeftTorch
from transformers import BertModel, DistilBertModel, AdamW
import argparse
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
import os
from torch.quantization import QuantStub, DeQuantStub, float_qparams_weight_only_qconfig, default_qconfig
# print(torch.__version__)
# exit()
# torch.set_num_threads(30)

#w2v = Word2Vec.load("word2vec.model")
#k2i = w2v.wv.key_to_index

#mk = max(k2i.values()) + 1
#k2i["[UNK]"] = mk

#i2k = w2v.wv.index_to_key
# print(i2k[-1])
#i2k.append("[UNK]")

#weight = w2v.wv.vectors
#zerovec = np.array([np.zeros(weight[0].shape[0],dtype=np.float64)])
# exit()
# print(weight.shape)
#weight = np.concatenate([weight,zerovec],axis=0)
# print(weight.shape)
# print(model.wv.most_similar(unk))
# print(type(model.wv.vectors[0]))
# print(k2i["[UNK]"])
# print(i2k)
# print(weight[mk])
#w2v.wv.index_to_key = i2k
#w2v.wv.key_to_index = k2i
#w2v.wv.vectors = weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
# torch.set_num_threads(16)
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
with open('kcc150_all_tag_dict.pkl','rb') as f:
    tag_dict = pickle.load(f)
#with open("ht_lm_dict.pkl","rb") as f:
#    map_dict = pickle.load(f)
tag_len = len(tag_dict.keys())
#vocab_size = len(map_dict)

# bilm = BiGramLM(0,vocab_size,7)
# bilm.load_state_dict(torch.load("lm/lm_5"))
# bilm = bilm.to(device)

parser = argparse.ArgumentParser(description="Postagger")

parser.add_argument("--MAX_LEN",type=int,help="MAX Sequnce Length",default=200)
parser.add_argument("--BATCH",type=int,help="BATCH Size",default=50)
parser.add_argument("--EPOCH",type=int,help="EPOCH Size",default=5)
parser.add_argument("--epoch_step",type=int,help="Train Data Epoch Step",default=4000)
parser.add_argument("--validation_step",type=int,help="Validation Data Epoch Step",default=240)
parser.add_argument("--hidden_state",type=int,help="BiLstm Hidden State",default=tag_len*2)
parser.add_argument("--GPU_NUM",type=str,help="Train GPU NUM",default="0")
parser.add_argument("--model_name",type=str,help="Tokenizer Model Name",default="tkbigram_one_first_alltag_bert_tagger.model")

args = parser.parse_args()
EPOCH = args.EPOCH
max_len = args.MAX_LEN
model = DistilBertModel.from_pretrained('monologg/distilkobert')
# model = BertModel.from_pretrained('monologg/kobert')
# peft_config = LoraConfig(
#     task_type=TaskType.FEATURE_EXTRACTION ,inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
# )
# model = get_peft_model(model, peft_config)

pmodel = PosTaggerModelPeftTorch(model,max_len,args.hidden_state,tag_len)
pmodel.load_state_dict(torch.load("pos_model_bi/model"))
pmodel = pmodel.to(device)
# model.eval()
tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
# print(count_parameters(model))
# model.emb.qconfig = float_qparams_weight_only_qconfig
# model.eval()
# model = torch.quantization.quantize_dynamic(
#  	pmodel,{torch.nn.Linear,torch.nn.LSTM}, dtype=torch.qint8
# )
# print(model)
model = pmodel
# exit()
def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

# print_size_of_model(pmodel)
# print_size_of_model(model)
# print(count_parameters(model))
# print(model)
# exit()

with open('kcc150_all_tag_dict.pkl','rb') as f:
    tag_dict = pickle.load(f)
    tag_dict = { v:k for k,v in tag_dict.items() }
    #print(tag_dict.keys())
#with open('eum_vocab.pkl','rb') as f:
#    vocab = pickle.load(f)
max_len = 200
import time
start = time.time()
# print(vocab.keys())
# x = '안녕하세요 국민대학▁김정민이다▁.'
# x = '내▁눈을▁본다면▁밤하늘의▁별이▁되는▁기분을▁느낄▁수▁있을▁거야'
# x = '나는▁밥을▁먹고▁집에▁갔다.'

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
	bi = []
	for i in x_t:
		x = i.replace('  ',' ').replace('+',' ')
		#x = list(x)
		#x = [vocab[e] if e in vocab else vocab['[UNK]'] for e in x ] 
		x = tokenizer.tokenize('[CLS] '+ x+' [SEP]') 
		x = tokenizer.convert_tokens_to_ids(x)#[2]+x+[3]

		x = x + [1] * (max_len-len(x))
		res.append(x)
		
	x = np.array(res)

	return x

def predict(x):
	# x = x.replace('  ',' ').replace(' ','▁')
	# x = list(x)
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

	# print(masks.shape,segments.shape,x.shape)
	# import sys
	# sys.exit()
	# return x,x_temp,x_len
	x = torch.tensor(x).to(device)
	masks = torch.tensor(masks).to(device)
	segments = torch.tensor(segments).to(device) 
	result = model(x,masks,segments)

	tagging = []
	# print(result.shape)
	for index,i in enumerate(result):
		array = []
		stop = -1
		for re_ in range(len(i)-1,0,-1):
			# print(result[re])
			tag = tf.argmax(i[re_])
			# array.append(tag)

			if tag != 3:
				stop = re_
				break
		temp = i[:stop+1]
		temp = temp[1:]
		#x_te = list(x_temp[index])
		re_tag = []
		for index_temp,te in enumerate(temp):
			tag = tf.argmax(te)
			
			re_tag.append(tag_dict[tag.numpy()])
			#if tag.numpy() == 2:
				#array.append(index_temp)
			#	x_te[index_temp] = x_te[index_temp] + '<tab>'
				# print(index,tag.numpy())
				# print(x_te)
			#elif tag.numpy() == 1:
			#    x_temp[index_temp] = x_temp[index_temp] + '▁'
		
		tagging.append(' '.join(re_tag))
		# print(tagging)
	return tagging

def predictbi(x,bi_temp,y):
    	# x = x.replace('  ',' ').replace(' ','▁')
	# x = list(x)
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
	bi_temp = np.array(bi_temp)

	# print(masks.shape,segments.shape,x.shape)
	# import sys
	# sys.exit()
	# return x,x_temp,x_len
	bi_temp = torch.tensor(bi_temp).to(device)
	x = torch.tensor(x).to(device)
	masks = torch.tensor(masks).to(device)
	segments = torch.tensor(segments).to(device) 
	#lmx = torch.tensor(lmx).to(device) 
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
	# for rr in result_:
	# 	re_tag = []
	# 	for rr_index,i in enumerate(rr):
	# 		if tag_dict[i] == '[PAD]' and rr_index < len(rr) -1 and tag_dict[rr[rr_index+1]]=='[PAD]':
	# 			break
	# 		re_tag.append(tag_dict[i])
	# 	tagging.append(' '.join(re_tag))
	# print(result.shape)
	#for index,i in enumerate(result):
	#	array = []
	#	stop = -1
	#	for re_ in range(len(i)-1,0,-1):
			# print(result[re])
	#		tag = tf.argmax(i[re_])
			# array.append(tag)

	#		if tag != 3:
	#			stop = re_
	#			break
	#	temp = i[:stop+1]
	#	temp = temp[1:]
		#x_te = list(x_temp[index])
	#	re_tag = []
	#	for index_temp,te in enumerate(temp):
	#		tag = tf.argmax(te)
			
	#		re_tag.append(tag_dict[tag.numpy()])
			#if tag.numpy() == 2:
				#array.append(index_temp)
			#	x_te[index_temp] = x_te[index_temp] + '<tab>'
				# print(index,tag.numpy())
				# print(x_te)
			#elif tag.numpy() == 1:
			#    x_temp[index_temp] = x_temp[index_temp] + '▁'
		
	#	tagging.append(' '.join(re_tag))
		# print(tagging)
	return tagging
#x = x[0][:x_len]

#array = array[1:-1]
#for i in array:#range(len(x_temp)):
    #x_te = x_temp[i]
    #tag_te = array[i]

    #if i == 2:
#    x_temp[i] = x_temp[i]+'+'

# print(''.join(x_temp))
# print(array)

with open('kcc150_all_tokenbi.pkl','rb') as f:
    bigram = pickle.load(f)

result_file = open('bibert_re_tkbigram_one.txt','w',encoding='utf-8')
result_file_y = open('bibert_fail_tkbigram_one.txt','w',encoding='utf-8')
result_file_tk = open('bibert_tk_tkbigram_one.txt','w',encoding='utf-8')
from tqdm import tqdm
import re
with open('test_data_x.txt','r',encoding='utf-8') as f:
	with open('test_data_y.txt','r',encoding='utf-8') as ff:
		count = 0
		X = []
		Y = []
		BI = []
		LMX = []
		from collections import deque
		with torch.no_grad():
			for _ in tqdm(range(200000)):
				l = f.readline()
				l = l.replace('\n','')
				temp_l = l
				l = re.sub(' +',' ',l)
				#l = re.sub('\.+','..',l)
				#l = re.sub('ㅋ+','ㅋㅋ',l)
				#l = re.sub('ㅎ+','ㅎㅎ',l)
				#l = re.sub('!+','!',l)
				#l = re.sub('\?+','?',l)
				#l = re.sub(',+',',',l)
				#l = re.sub('~+','~',l)
				#l = re.sub(';+',';',l)
				#l = re.sub('ㅇ+','ㅇㅇ',l)
				#l = re.sub('ㅠ+','ㅠㅠ',l)
				#l = re.sub('ㅜ+','ㅜㅜ',l)
				yy = ff.readline()
				# print(l)
				bi_npy = []
				# uni_npy = []
				l_temp = l.replace('+',' ')
				#for i in range(len(l_temp)-1):
				#	bi = l_temp[i:i+2]
					# if bi not in bigram:
					# 	bigram[bi] = biindex
					# 	biindex += 1
				#	if bi in bigram:
				#		bi_npy.append(bigram[bi])
				#	else:
				#		bi_npy.append(bigram['[UNK]'])
				#bi_npy = bi_npy + [1] * (260 - len(bi_npy))
				#bi_npy = np.array(bi_npy)
				#lmx = []
				#tks = tokenizer.tokenize('[CLS] '+ l_temp +' [SEP]')
				#text = l_temp.strip().replace("+"," ").split()
                # lmx_ = []
				#text = deque(text)
                # print(d)
				# tks = tokenizer.convert_ids_to_tokens(d)
                
				#tempx = []
				#for tk in tks:
				#	if tk.startswith("▁"):
				#		tempx.append(text.popleft())
				#	else:
				#		tempx.append("[UNK]")
                # exit()
                # print(text)
                # print(tks_)
				#text = tempx
				#for text_ in text:
				#	if text_ in w2v.wv.key_to_index:
				#		lmx.append(w2v.wv.key_to_index[text_])
				#	else:
				#		lmx.append(w2v.wv.key_to_index["[UNK]"])
				#lmx = lmx + ([w2v.wv.key_to_index["[UNK]"]] * (max_len - len(lmx)))
				# lmx.append(lmx[:200])
		
				# print(l_temp)
				# for l in ["[SOS]"] + list(temp_l[:198]) + ["[EOS]"]:
				# 	if l in map_dict:
				# 		lmx.append(map_dict[l])
				# 		# print(l)
				# 	else:
				# 		# print(l)
				# 		lmx.append(map_dict["[UNK]"])
				# lmx = lmx + ([map_dict["[PAD]"]] * (max_len - len(lmx)))
 
				temp = tokenizer.tokenize('[CLS] '+ l_temp +' [SEP]')
				for token_i in range(len(temp)):
					bii = temp[token_i:token_i+2]
					bii = '+'.join(bii)
					if bii in bigram:
						bi_npy.append(bigram[bii])
						# print("bi",bii)
					else:
						bi_npy.append(bigram['[UNK]'])
						# print("nbi",bii)
				bi_npy = bi_npy + [1] * (max_len - len(bi_npy))
				bi_npy = np.array(bi_npy)
				if len(temp) <= max_len and bi_npy.shape[0] <= max_len:
					X.append(l)#,l)
					Y.append(yy)
					BI.append(bi_npy)
					#LMX.append(lmx)
					count += 1
					result_file_tk.write(temp_l+'\n')
					if count % 30 ==0:
						# print(Y)
						result = predictbi(X,BI,Y)#.replace('▁',' ')
						result_file.write('\n'.join(result)+'\n')
						result_file_y.write(''.join(Y))
						Y = []
						X = []
						BI = []
						#LMX = []
					# if count == 50000:
					# 	break
			result = predictbi(X,BI,Y)#.replace('▁',' ')
			result_file.write('\n'.join(result)+'\n')
			result_file_y.write(''.join(Y))
		
#if len(X) > 0:	
#	result = predictbi(X,BI)#.replace('▁ ',' ')
#	result_file.write('\n'.join(result)+'\n')   
#	result_file_y.write(''.join(Y))
result_file.close()
result_file_y.close()
result_file_tk.close()
end = time.time()

print(end-start)

t = {}
import pickle
with open("kcc150_all_tag_dict.pkl","rb") as f:
	ttt = pickle.load(f)

count = 0
with open('test_data_y.txt','r',encoding='utf-8') as ff:
	for l in ff:
		l = l.strip().split()

		for ll in l:
			ll = ll.split("+")
			for lll in ll:
				if lll not in ttt:
					t[lll] = 1

print(len(lll))

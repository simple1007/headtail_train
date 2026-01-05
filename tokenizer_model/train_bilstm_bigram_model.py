from tensorflow.keras.layers import LSTM, Input, Bidirectional, Embedding,TimeDistributed,Dense,Concatenate
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import pickle
import numpy as np
import tensorflow as tf
import os
import argparse

parser = argparse.ArgumentParser(description="Train Head-Tail Tokenizer")

parser.add_argument("--BATCH",type=int,help="Train Data BATCH SIZE",default=50)
parser.add_argument("--MAX_LEN",type=int,help="Sequence Data MAX LENGTH",default=300)
parser.add_argument("--epoch_step",type=int,help="Train Data Epoch Step",default=4000)
parser.add_argument("--validation_step",type=int,help="Validation Data Epoch Step",default=400)
parser.add_argument("--EPOCH",type=int,help="Train Epoch SIZE",default=6)
# parser.add_argument("--BATCH",type=int,help="Train BATCH SIZE",default=50)
parser.add_argument("--model_name",type=str,help="Tokenizer Model Name",default="lstm_bigram_tokenizer.model")
parser.add_argument("--GPU_NUM",type=str,help="Train GPU NUM",default="0")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_NUM
max_len = args.MAX_LEN#300
EPOCH = args.EPOCH#6
BATCH_SIZE = args.BATCH#50
count_data = args.epoch_step#4000
validation_data = args.validation_step#400

def dataset():
    for _ in range(EPOCH):
        for i in range(count_data):
            data = np.load('token_data/%05d_lstm_x.npy' % i)
            data_bigram = np.load('token_data/%05d_bigram.npy' % i)
            y = np.load('token_data/%05d_lstm_y.npy' % i)

            yield [data,data_bigram],y

def validation():
    for _ in range(EPOCH*2):
        for i in range(count_data,count_data+validation_data):
            data = np.load('token_data/%05d_lstm_x.npy' % i)
            data_bigram = np.load('token_data/%05d_bigram.npy' % i)
            y = np.load('token_data/%05d_lstm_y.npy' % i)

            yield [data,data_bigram],y

with open('lstm_vocab.pkl','rb') as f:
    lstm_vocab = pickle.load(f)
with open('bigram_vocab.pkl','rb') as f:
    bigram = pickle.load(f)

class TK_Model:
    def __init__(self,max_len):
        self.max_len = max_len  
    def build(self):
        in_word = Input(shape=(self.max_len,))
        emb = Embedding(len(lstm_vocab.keys()),100,input_length=self.max_len)(in_word)

        in_bi = Input(shape=(self.max_len,))
        emb_bi = Embedding(len(bigram.keys()),100,input_length=self.max_len)(in_bi)

        bilstm = Bidirectional(LSTM(64,return_sequences=True,dropout=0.1,name="Eumjeal_Embedding_lstm"),name="eumjul_emb")(emb)
        bilstm_bi = Bidirectional(LSTM(64,return_sequences=True,dropout=0.1,name="Eumjeal_bigram_Embedding_lstm"),name="bigram_emb")(emb_bi)

        concat = Concatenate(axis=-1)([bilstm,bilstm_bi])
        output_tag = TimeDistributed(Dense(4,activation='softmax'))(concat)

        model = Model(inputs=[in_word,in_bi],outputs=output_tag)
        model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

        return model

train_data = dataset()
val_data = validation()

callbacks = [EarlyStopping(monitor='val_loss',patience=2), ModelCheckpoint(args.model_name,monitor='val_loss',save_best_only=True)]

model = TK_Model(max_len).build()
model.fit(train_data,epochs=EPOCH,batch_size=BATCH_SIZE,validation_data=val_data,steps_per_epoch=count_data,validation_steps=validation_data,callbacks=callbacks)#,callbacks=[callback])

# model.save(args.model_name)

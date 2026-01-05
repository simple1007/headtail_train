import urllib.request

urllib.request.urlretrieve("https://raw.githubusercontent.com/songys/Chatbot_data/master/ChatbotData.csv", filename="ChatBotData.csv")

import pandas as pd

from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sentencepiece as spm
VOCAB_SIZE = 10000
class TrainDataset(Dataset):
    def __init__(self, src_data, trg_data):
        super().__init__()

        assert len(src_data) == len(trg_data)

        self.src_data = src_data
        self.trg_data = trg_data

    def __len__(self):
        return len(self.src_data)
        
    def __getitem__ (self, idx):
        src = self.src_data[idx]
        trg_input = self.trg_data[idx]
        trg_output = trg_input[1:SEQ_LEN]
        trg_output = np.pad(trg_output, (0,1), 'constant', constant_values =0)
        # (seq_len,)
        return torch.Tensor(src).long(), torch.Tensor(trg_input).long(), torch.Tensor(trg_output).long()

# train_dataset = TrainDataset(src_train, trg_train)
# train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle= True, pin_memory=True)

class ValidDataset(Dataset):
    def __init__(self, src_data, trg_data):
        super().__init__()

        assert len(src_data) == len(trg_data)

        self.src_data = src_data
        self.trg_data = trg_data

    def __len__(self):
        return len(self.src_data)
        
    def __getitem__ (self, idx):
        src = self.src_data[idx]
        trg_input = self.trg_data[idx]
        trg_output = trg_input[1:SEQ_LEN]
        trg_output = np.pad(trg_output, (0,1), 'constant',constant_values= 0)

        return torch.Tensor(src).long(), torch.Tensor(trg_input).long(), torch.Tensor(trg_output).long()

if __name__ == "__main__":
    # valid_dataset = ValidDataset(src_valid, trg_valid)
    # valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle= False, pin_memory=True)
    # en_train = open('bible-all.en.txt',encoding="utf-8")
    # en_train_content = en_train.read()

    # en_train_list = en_train_content.split('\n')


    # ko_train = open('bible-all.kr.txt',encoding="utf-8")
    # ko_train_content = ko_train.read()



    # ko_train_list = ko_train_content.split('\n')




    # en_train_list[:10]




    # data = pd.DataFrame()
    # data['en_raw'] = en_train_list
    # data['ko_raw'] = ko_train_list



    # data = data.reset_index(drop = True)
    # data.head()

    # data['Q'] = data['en_raw'].apply(lambda x: x.split(' ')[1:])
    # data['Q'] = data['Q'].apply(lambda x: (' ').join(x))
    # data['A'] = data['ko_raw'].apply(lambda x: x.split(' ')[1:])
    # data['A'] = data['A'].apply(lambda x: (' ').join(x))

    data = pd.read_csv("ChatBotData.csv")
    data = data[['Q','A']]
    # data = data[['Q','A']]
    # print(data.head())
    # data.to_csv("train.csv")


    with open('src.txt', mode = 'w', encoding='utf8') as f:
        f.write('\n'.join(data['Q']))
    with open('trg.txt', mode= 'w', encoding='utf8') as f:
        f.write('\n'.join(data['A']))
        # f.write('\n'+'\n'.join(data['Q']))
    exit()
    corpus = "src.txt"
    prefix = "src"
    spm.SentencePieceTrainer.train(
        f"--input={corpus} --model_prefix={prefix} --vocab_size={VOCAB_SIZE}" +
        " --model_type=bpe" +
        " --max_sentence_length=999999" +  # ¹®Àå ÃÖ´ë ±æÀÌ
        " --pad_id=0 --pad_piece=[PAD]" +  # pad (0)
        " --unk_id=1 --unk_piece=[UNK]" +  # unknown (1)
        " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence (2)
        " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence (3)
        " --user_defined_symbols=[SEP],[CLS],[MASK]");  # »ç¿ëÀÚ Á¤ÀÇ ÅäÅ«

    corpus = "trg.txt"
    prefix = "trg"
    spm.SentencePieceTrainer.train(
        f"--input={corpus} --model_prefix={prefix} --vocab_size={VOCAB_SIZE}" +
        " --model_type=bpe" +
        " --max_sentence_length=999999" +  # ¹®Àå ÃÖ´ë ±æÀÌ
        " --pad_id=0 --pad_piece=[PAD]" +  # pad (0)
        " --unk_id=1 --unk_piece=[UNK]" +  # unknown (1)
        " --bos_id=2 --bos_piece=[BOS]" +  # begin of sequence (2)
        " --eos_id=3 --eos_piece=[EOS]" +  # end of sequence (3)
        " --user_defined_symbols=[SEP],[CLS],[MASK]");  # »ç¿ëÀÚ Á¤ÀÇ ÅäÅ«



    # sp_src = spm.SentencePieceProcessor()
    # sp_src.Load('src.model')


    # for idx in range(3):
    #     sentence = data['en'][idx]
    #     print(sp_src.EncodeAsPieces(sentence))
    #     print(sp_src.EncodeAsIds(sentence))

    # sp_trg = spm.SentencePieceProcessor()
    # sp_trg.Load('trg.model')


    # for idx in range(3):
    #     sentence = data['ko'][idx]
    #     print(sp_trg.EncodeAsPieces(sentence))
    #     print(sp_trg.EncodeAsIds(sentence))
import os
import argparse
from tokenizers import BertWordPieceTokenizer, ByteLevelBPETokenizer, CharBPETokenizer, SentencePieceUnigramTokenizer
from tokenizers.processors import TemplateProcessing
import sentencepiece as spm
if True:
    parser = argparse.ArgumentParser()
    # parser.add_argument('--corpus_file', type=str, required=True)
    # parser.add_argument('--vocab_size', type=int, default=0, required=True)
    # parser.add_argument('--limit_alphabet', type=int, default=0)
    # parser.add_argument('--tokenizer', type=str, default='sentencepiece')
    # parser.add_argument('--model_type', type=str, default='gpt')
    # parser.add_argument('--save_dir', type=str, default='sp')
    # args = parser.parse_args()
    user_symb = [f"[UNK{i}]" for i in range(100)]
    number = [str(i) for i in range(10)]
    special_token = ["+",'[SOS]', '[EOS]','[CLS]',"[SEP]","[MASK]"] + number
    # control=['[PAD]', '[UNK]']


    # tokenizer = SentencePieceUnigramTokenizer()
    # nb = "C:\\Users\\ty341\\OneDrive\\Desktop\\kjm\\autospace\\blog_autospace.txt"
    # nb2 = "C:\\Users\\ty341\\OneDrive\\Desktop\\dataset\\homedari_autos_2.txt"
    nb = os.path.join(os.environ["DATASET"],"ht_dataset","modu","result","delkiwimorphs.txt")
    # nb = os.path.join(os.environ["DATASET"],"KCC150_Korean_sentences_UTF8.txt")
    from headtail_util import get_mp_tags
    with open(nb,encoding="utf-8") as f:
        # text = f.read()#.split("\n")
        res = open("nbs.txt","w",encoding="utf-8")
        linecnt = 0
        for l in f:
            l = l.replace("hththththt"," + ")
            res.write(l)
        # with open(os.path.join(os.environ["DATASET"],"naverblog_sentsplit250205.txt"),encoding="utf-8") as ff:
        #     for l in ff:
        #         res.write(l)
            # linecnt += 1
            # if linecnt == 7000000:
                # print(linecnt)
                # break
        # import re
        # with open("kccres_httk.txt",encoding="utf-8") as ff:
            
            # for l in ff:
            #     tmpl = []
            #     l = re.sub(r" +"," ",l)
            #     l = l.replace(" + ","+").replace(" +","+").replace("+ ","+")
            #     for ll in l.strip().split():
            #         # print(ll)
            #         try:
            #             head,tail = get_mp_tags(ll)
            #         except:
            #             print(ll)
            #             print(l)
            #             exit()
            #         if len(tail) > 0:
            #             ht = head[0] + "+" + tail[0]
            #         else:
            #             ht = head[0]
            #         tmpl.append(ht)
            #     res.write(" ".join(tmpl)+"\n")
        res.close()
    # exit()
        # for l in text:
        #     l = l.replace("_","").replace("+","_").replace("/","_")
        #     l = l.replace("hththththt","+")
        #     res.write(l+"\n")
    input_file = "./nbs.txt"
    vocab_size = 10000
    prefix = "./spm_mp"
    template = "--input={} --model_prefix={} --vocab_size={} --unk_piece=[UNK] --pad_piece=[PAD] --model_type=unigram --seed_sentencepiece_size=7000000 --normalization_rule_name=identity --user_defined_symbols={}"
    print(",".join(special_token))
    cmd = template.format(input_file,prefix,vocab_size,",".join(special_token))
    # import logging
    # logger = logging.getLogger
    # stream_hendler = logging.StreamHandler()
    # logger = logging.getLogger(__name__)
    # logger.addHandler(stream_hendler)
    spm.SentencePieceTrainer.Train(cmd)
    # tokenizer.train(
    #     files=[nb],
    #     vocab_size=20000,
    #     special_tokens=special_token,
    #     unk_token = "[UNK]",
    #     # pad_token = "[PAD]",
    #     # sep_token = "[SEP]",
    #     # cls_token = "[CLS]",
    #     # mask_token = "[MASK]" 
    # )

    from transformers import BertTokenizerFast
    
    if not os.path.exists("kcctokenizer"):
        os.mkdir("kcctokenizer")
    import tokenizers
    # tokenizer.save("kcctokenizer/tokenizer.json")
    tok=tokenizers.SentencePieceUnigramTokenizer.from_spm("./spm_mp.model")
    # tok = BertTokenizerFast(vocab_file="./spm_mp.vocab",)
    tok.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", tok.token_to_id("[CLS]")),
            ("[SEP]", tok.token_to_id("[SEP]"))
        ],
    )
    # tok.pad_token = []
    tok.save("kcctokenizer/tokenizer.json")
    # tok.save_pretrained("hf_format_tokenizer")
    tokenizer = BertTokenizerFast(tokenizer_file="kcctokenizer/tokenizer.json",cls_token="[CLS]",sep_token="[SEP]",pad_token="[PAD]",unk_token="[UNK]",mask_token="[MASK]")
    tokenizer.pad_token = "[PAD]"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    tokenizer.sep_token = "[SEP]"
    tokenizer.sep_token_id = tokenizer.convert_tokens_to_ids("[SEP]")
    tokenizer.cls_token = "[CLS]"
    tokenizer.cls_token_id = tokenizer.convert_tokens_to_ids("[CLS]")
    tokenizer.unk_token = "[UNK]"
    tokenizer.unk_token_id = tokenizer.convert_tokens_to_ids("[UNK]")
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained("./kcctokenizer")
    
from transformers import BertTokenizerFast
from ht_utils import removesym
tokenizer = BertTokenizerFast.from_pretrained("./kcctokenizer")
print( tokenizer.convert_tokens_to_ids("[SEP]"))
tks = ['추진', '지역', '+', '은', '가평읍', '경반', '·', '승안', '·', '마장리', '일대', '3648', '㏊', ',', '설악면', '설곡', '·', '묵안', '·', '엄소리', '일대', '520', '㏊', ',', '상면', '상동', '·', '행현', '·', '덕현리', '일대', '2317', '㏊', '등', '총', '6485', '㏊', '의', '산림', '+', '이다', '.']
pos = ['NNG', 'NNG', '+', 'JX', 'NNP', 'NNG', 'SP', 'NNG', 'SP', 'NNP', 'NNG', 'SN', 'SW', 'SP', 'NNP', 'NNP', 'SP', 'NNP', 'SP', 'NNP', 'NNG', 'SN', 'SW', 'SP', 'NNG', 'NNP', 'SP', 'NNP', 'SP', 'NNP', 'NNG', 'SN', 'SW', 'NNB', 'MM', 'SN', 'SW', 'JKG', 'NNG', '+', 'VCP_EF', 'SF']
# 
txt = "구피+는 바보+다"#'이들 국무위원들+은 개발 제품+을 체험+하고 사내창업 , 창업분사 현황 및 우수 사례+를 경청+했다 .\n'
# txt = removesym(txt)
print(txt)
txt = txt.replace("+"," + ")
tks = txt.split()
pos = ['[SOS]', 'NP_XSN', 'NNP', '+', 'JX', 'NNG', 'NNG', '+', 'JKO', 'NNG', '+', 'XSV_EC', 'NNG', 'SP', 'NNG', 'NNG', 'MAG', 'NNG', 'NNG', '+', 'JKO', 'NNG', '+', 'XSV_EF', 'SF', '[EOS]']
res = tokenizer(tks,max_length=30,truncation=True,is_split_into_words=True,return_offsets_mapping=True)# 먹+고 학교+에 갔+다.",text_target=)
# print(tokenizer.encode("나+는 밥+을 먹+고 학교+에 갔+다.")[])
print(res.tokens())
print(res["input_ids"])
print(tokenizer.convert_ids_to_tokens(res["input_ids"]))
exit()
# print(res)
# tks = res["input_ids"]
# tmp = []
# c = 0

# for i in res["offset_mapping"]:
#     if i[0] == 0 and c != 0:
#         if tokenizer.decode(tmp).strip() != "":
#             print("---------")
#             print(tmp)
#             print(tokenizer.decode(tmp))
#         tmp = []
#     tk = tks.pop(0)
#     tmp.append(tk)
#     c+= 1
# print(len(tokenizer))

sp = tokenizer.convert_tokens_to_ids('▁')

print(sp)
print(tokenizer.convert_ids_to_tokens(sp))
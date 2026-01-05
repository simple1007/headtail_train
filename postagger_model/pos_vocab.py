import re
def tagged_reduce(tag):
    # print(tag__)
    # print("split",tag__)
    # tag = tag__.pop(0)
    # return tag.strip("_")
    if tag.startswith("_"):
        # return tag
        tag = tag[1:].rstrip("_")
        tag = tag.split("_")
        
        res = []
        for tt in tag:
            if "-" in tt:
                tt = tt.split("-")[0]
            res.append(tt)
        tag = res
        if len(tag) == 1:
            return tag[0] 
        else:
            return tag[0] + "_" + tag[-1]
    else:
        res = []
        # tmp = tag
        # if tag.count("NNG") >= 2:
        tmp = []
        # for tag_ in tag.split("_"):
        #     # print(tag_)
        #     if len(tag_) > 0:
        #         tmp.append(tag_[0])
        # # tmp = "".join(tmp)
        if tag.count("NN") >= 2 or (tag.count("NN") > 0 and tag.count("NN") != len(tag.split("_"))):#(tmp[0]!="N" or tmp[-1] != "N")):#tmp.count("N") >= 2 and tmp.count("N") == len(tmp):
            # if tmp[0] != "N":
                # res.append(tag.split("_")[0])
            res.append("C")
            # if tmp[-1] != "N":
                # res.append(tag.split("_")[-1])
            return "_".join(res)
        #     print(tag)
        #     exit()
        for t in tag.split("_"):
            if "-" in t:
                t = t.split("-")[0]
            res.append(t)
        tag = "_".join(res)
        tag = re.sub(r"(NNP_NNG|NNG_NNP)+",r"_NNP_",tag)
        tag = tag.strip("_")
        tag = re.sub(r"_+","_",tag)
        # if tag.count("_") == 4:
        # if tmp != tag:
            # print(tmp,tag)
    return tag
    # The code is removing the first character from the variable `tag` and assigning the result
    # back to the variable `tag`.
    tag = tag.split('_')#[:3]
    
    tag = [t if t == 'VV' or t == 'VA' or t == "NNP" or t == "NNG" else t[0] for t in tag]
    tag = '_'.join(tag)
    return tag
    if 't_' in tag:
        tag_spl = tag.split('_')
        if len(tag_spl) <= 2:
            tag = tag.replace('t_','')
        elif tag_spl[1] == 'SN':
            tag = tag_spl[-3]+'_'+tag_spl[-2]
        else:
            tag = tag_spl[1]+'_'+tag_spl[2]
    
    return tag#,tag_spl

class PosVocab:
    def __init__(self,maxlen):
        self.word2index = {"[PAD]":0,"[UNK]":1,"[SOS]":2,"[EOS]":3,"+":4," ":5}
        self.index2word = {v:k for k,v in self.word2index.items()}
        self.pos2index = {"[PAD]":0,"[SOS]":1,"[EOS]":2,"+":3,"O":4}
        self.index2pos = {v:k for k,v in self.pos2index.items()}
        self.wordindex = 6
        self.posindex = 5
        self.maxlen = maxlen 
    		
    def make_dict(self,ht_sentence,pos_sentence):
        for p in pos_sentence.split():
            # h = list(h.strip().replace("+", " "))#.split()
            p = p.strip().replace("+", " _").split()
            # p =
            for pp in p:
                pp = tagged_reduce(pp)
                if ("B_" + pp) not in self.pos2index:
                    self.pos2index["B_"+pp] = self.posindex
                    self.index2pos[self.posindex] = "B_"+pp
                    self.posindex += 1
                if ("I_" + pp) not in self.pos2index:
                    self.pos2index["I_"+pp] = self.posindex
                    self.index2pos[self.posindex] = "I_"+pp
                    self.posindex += 1
        
        for hh in list(ht_sentence.strip().replace("+", " ")):        
        # for hh in h:
            if hh not in self.word2index:
                self.word2index[hh] = self.wordindex
                self.index2word[self.wordindex] = hh
                self.wordindex += 1
                    
    
    def make_onehot(self,ht_sentence,pos_sentence):
        resultword = []
        resultpos = []
        tag = []
        h_ = ht_sentence.strip().replace("+", " ").split()
        p_ = pos_sentence.strip().replace("+", " _").split()
        for h, p in zip(h_,p_):

            p = tagged_reduce(p)
            # print(p)
            t1 = ["B"+"_"+p]
            t2 = ["I"+"_"+p for _ in range(len(h)-1)]
            for tt in t1+t2:
                resultpos.append(self.pos2index[tt])
                tag.append(tt)
            resultpos.append(self.pos2index["O"])
            tag.append("O")
           
        for h in list(ht_sentence.replace("+", " ")):    
            if h not in self.word2index:
                resultword.append(self.word2index["[UNK]"])
            else:
                resultword.append(self.word2index[h])
        if resultpos[-1] == " ":
            resultpos = resultpos[:-1]
        if tag[-1] == " ":
            tag = tag[:-1]
            
        return resultword, resultpos
    
    def make_batch(self,batch_data_tk,batch_data_p):
        words = []
        poss = []
        for bdsenttk,bdsentp in zip(batch_data_tk,batch_data_p):
            wordidx, posidx = self.make_onehot(bdsenttk,bdsentp)
            
            wordidx = [self.word2index["[SOS]"]] + wordidx[:self.maxlen-2] + [self.word2index["[EOS]"]] 
            wordidx = wordidx + [self.word2index["[PAD]"]] * (self.maxlen - len(wordidx[:self.maxlen]))
            posidx = [self.pos2index["[SOS]"]] + posidx[:self.maxlen-2] + [self.pos2index["[EOS]"]] 
            posidx = posidx + [self.pos2index["[PAD]"]] * (self.maxlen - len(posidx[:self.maxlen]))
        
            words.append(wordidx)
            poss.append(posidx)
            
        return words,poss
    
    def make_onehot2(self,ht_sentence):
        resultword = []
        resultpos = []
        
        for h in list(ht_sentence.strip().replace("+", " ")):
            if h not in self.word2index:
                resultword.append(self.word2index["[UNK]"])
            else:
                resultword.append(self.word2index[h])
        
        return resultword

    def make_batch2(self,batch_data_tk):
        words = []
        poss = []
        for bdsenttk in batch_data_tk:
            wordidx = self.make_onehot2(bdsenttk)
            
            wordidx = [self.word2index["[SOS]"]] + wordidx[:self.maxlen-2] + [self.word2index["[EOS]"]] 
            wordidx = wordidx + [self.word2index["[PAD]"]] * (self.maxlen - len(wordidx[:self.maxlen]))
            words.append(wordidx)
            
        return words

    def to_tag(self,tag):
        result = []
        temp = []
        for t_ in tag:
            if t_ == 0:
                break
            result.append(self.index2pos[t_])
        
        return result

from collections import defaultdict
class BiPosVocab:
    def __init__(self,maxlen,subword=True):
        self.subword = subword
        self.word2index = {"[PAD]":0,"[UNK]":1,"[SOS]":2,"[EOS]":3,"+":4," ":5}
        self.index2word = {v:k for k,v in self.word2index.items()}
        self.pos2index = {"[PAD]":0,"[SOS]":1,"[EOS]":2,"+":3,"O":4,"[UNK]":5}
        self.index2pos = {v:k for k,v in self.pos2index.items()}
        self.wordindex = 6
        self.posindex = 6
        self.maxlen = maxlen 
        self.uni2index = {"[PAD]":0,"[SOS]":1,"[EOS]":2,"+":3," ":4,"[UNK]":5}
        self.index2uni = {v:k for k,v in self.uni2index.items()}
        self.uniindex = 6
        # self.tag_bio = {"B":0,"I":1,"O":2,"[UNK]":3,"[PAD]":4}
        self.count = defaultdict(int)
        
    def __len__(self):
        return len(self.word2index)
    
    def tokens2(self,text):
        text = text
    
        resultbi = []
        resultuni = []
        flag = True
        temp =[]
        textbi = text + "E"
        for index in range(len(textbi)-1):
            bis = ''.join(textbi[index:index+2])
            temp.append(bis)
            
        resultbi.append(temp)
        resultuni.append(list(text))
        return resultuni[0],resultbi[0]
    
    def tokens(self,text):
        text = text
    
        resultbi = []
        resultuni = []
        flag = True
        txt = text.split()
        for index,tk in enumerate(txt):
            temp =[]
            if flag:
                flag = False
            else:
                if tk != '+' and txt[index-1] != "+":
                    tk = "_"+tk

            for i in range(0,len(tk)-1 if len(tk) > 1 else 1,1 if len(tk) > 1 else 1):
                bis = ''.join(tk[i:i+2])
                temp.append(bis)
            
            if len(tk) > 1:
                unis = list(tk)+["$"]
            else:
                unis = list(tk)
            resultbi.append(temp)
            resultuni.append(unis)
        return resultuni,resultbi
    		
    def make_dict(self,ht_sentence,pos_sentence,tags=None):
        self.tags = tags
        for p in pos_sentence.split():
            p = p.strip().replace("+", " + _").split()
            for pp in p:
                if pp == "+":
                    continue
                pp = tagged_reduce(pp)
                self.count[pp] += 1
                if tags == None:
                    # if pp not in self.pos2index:
                    #     self.pos2index[pp] = self.posindex
                    #     self.index2pos[self.posindex] = pp
                    #     self.posindex += 1
                    if ("B_" + pp) not in self.pos2index:
                        self.pos2index["B_"+pp] = self.posindex
                        self.index2pos[self.posindex] = "B_"+pp
                        self.posindex += 1
                    if "I_" + pp not in self.pos2index:
                        self.pos2index["I_"+pp] = self.posindex
                        self.index2pos[self.posindex] = "I_" + pp
                        self.posindex += 1
                else:
                    if pp in tags:
                        # if pp not in self.pos2index:
                        #     self.pos2index[pp] = self.posindex
                        #     self.index2pos[self.posindex] = pp
                        #     self.posindex += 1
                        if ("B_" + pp) not in self.pos2index:
                            self.pos2index["B_"+pp] = self.posindex
                            self.index2pos[self.posindex] = "B_"+pp
                            self.posindex += 1
                        if "I_" + pp not in self.pos2index:
                            self.pos2index["I_"+pp] = self.posindex
                            self.index2pos[self.posindex] = "I_" + pp
                            self.posindex += 1
                    
                        
        tks = self.tokens2(ht_sentence.strip().replace("+", " "))
        for hh in tks[1]:        
            if hh not in self.word2index:
                self.word2index[hh] = self.wordindex
                self.index2word[self.wordindex] = hh
                self.wordindex += 1
            # for u in uni:
        for u in tks[0]:
            if u not in self.uni2index:
                self.uni2index[u] = self.uniindex
                self.index2uni[self.uniindex] = u
                self.uniindex += 1        
                
    def make_onehot(self,ht_sentence,pos_sentence):
        resultword = []
        resultpos = []
        resultuni = []
        tag = []
        biotag = []
        # print(ht_sentence)
        h_ = ht_sentence.strip().replace("+", " ").split()
        p_ = pos_sentence.strip().replace("+", " _").split()
        for h, p in zip(h_,p_):

            p = tagged_reduce(p)
            # print(p)
            if "I_"+p in self.pos2index or "B_"+p in self.pos2index:
                t1 = ["B"+"_"+p]
                t2 = ["I"+"_"+p for _ in range(len(h)-1)]
                # t1 = [p]
                # t2 = [p for _ in range(len(h)-1)]
                # biotag1 = [self.tag_bio["B"]]
                # biotag2 = [self.tag_bio["I"] for _ in range(len(h)-1)]
            else:
                t1 = ["[UNK]"]
                t2 = ["[UNK]" for _ in range(len(h)-1)]
                # biotag1 = [self.tag_bio["[UNK]"]]
                # biotag2 = [self.tag_bio["[UNK]"] for _ in range(len(h)-1)]
            for tt in t1+t2:
                try:
                    resultpos.append(self.pos2index[tt])
                    tag.append(tt)
                except:
                    print(self.tags)
                    print(self.pos2index)
                    print(ht_sentence)
                    print(pos_sentence)
                    print(h,p)
                    print(tt)
                    exit()
            # for tt in biotag1 + biotag2:
            #     biotag.append(tt)
                    
            resultpos.append(self.pos2index["O"])
            # biotag.append(self.tag_bio["O"])
            tag.append("O")

        words = []  
        unis =  []        
        tks = self.tokens2(ht_sentence.replace("+", " "))
        # print(tks)
        for h_ in tks[1]:
            if h_ not in self.word2index:
                resultword.append(self.word2index["[UNK]"])
            else:
                resultword.append(self.word2index[h_])
            words.append(h_)
        
        for uni in tks[0]:
            if uni not in self.uni2index:
                resultuni.append(self.uni2index["[UNK]"])
            else:
                resultuni.append(self.uni2index[uni])
            unis.append(uni)    
        try:
            if resultword[-1] == " ":
                # resultpos = resultpos[:-1]
                resultword = resultword[:-1]
                resultuni = resultuni[:-1]
                # biotag = biotag[:-1]
        except:
            print(ht_sentence)
            print(pos_sentence)
            print(tks)
            exit()
        if tag[-1] == "O":
            tag = tag[:-1]
            resultpos = resultpos[:-1]
            # biotag = biotag[:-1]
        return resultuni,resultword, resultpos
    
    def make_batch(self,batch_data_tk,batch_data_p):
        words = []
        unis =[]
        poss = []
        bios = []
        ml = 0
        for i in batch_data_tk:
            ml = max(len(i),ml)
        maxlen = min(ml,self.maxlen)
        for bdsenttk,bdsentp in zip(batch_data_tk,batch_data_p):
            uniidx, wordidx, posidx = self.make_onehot(bdsenttk,bdsentp)
            
            uniidx = [self.uni2index["[SOS]"]] + uniidx[:maxlen-2] + [self.uni2index["[EOS]"]] 
            uniidx = uniidx + [self.uni2index["[PAD]"]] * (maxlen - len(uniidx[:maxlen]))
            wordidx = [self.word2index["[SOS]"]] + wordidx[:maxlen-2] + [self.word2index["[EOS]"]] 
            wordidx = wordidx + [self.word2index["[PAD]"]] * (maxlen - len(wordidx[:maxlen]))
            posidx = [self.pos2index["[SOS]"]] + posidx[:maxlen-2] + [self.pos2index["[EOS]"]] 
            posidx = posidx + [self.pos2index["[PAD]"]] * (maxlen - len(posidx[:maxlen]))
            # biotag = [self.tag_bio["O"]] + biotag[:self.maxlen-2] + [self.tag_bio["O"]]
            # biotag = biotag + [self.tag_bio["[PAD]"]] * (self.maxlen - len(biotag[:self.maxlen]))
            unis.append(uniidx)
            words.append(wordidx)
            poss.append(posidx)
            # bios.append(biotag)
        
        return unis,words,poss
    
    def make_onehot2(self,ht_sentence):
        resultword = []
        resultpos = []
        resultuni = []
        words =[]
        unis =[]
        tks = self.tokens2(ht_sentence.strip().replace("+", " "))
        for uni,h_ in zip(tks[0],tks[1]):
            unis.append(uni)
            words.append(h_)
            if h_ not in self.word2index:
                resultword.append(self.word2index["[UNK]"])
            else:
                resultword.append(self.word2index[h_])

            if uni not in self.uni2index:
                resultuni.append(self.uni2index["[UNK]"])
            else:
                resultuni.append(self.uni2index[uni])
                
        return resultuni, resultword
   
    def make_batch2(self,batch_data_tk):
        words = []
        poss = []
        unis = []
        for bdsenttk in batch_data_tk:
            uniidx,wordidx = self.make_onehot2(bdsenttk)
            
            uniidx = [self.uni2index["[SOS]"]] + uniidx[:self.maxlen-2] + [self.uni2index["[EOS]"]] 
            uniidx = uniidx + [self.uni2index["[PAD]"]] * (self.maxlen - len(uniidx[:self.maxlen]))
            wordidx = [self.word2index["[SOS]"]] + wordidx[:self.maxlen-2] + [self.word2index["[EOS]"]] 
            wordidx = wordidx + [self.word2index["[PAD]"]] * (self.maxlen - len(wordidx[:self.maxlen]))
   
            words.append(wordidx)
            unis.append(uniidx)
            
        return unis,words

    def to_tag(self,tag):
        result = []
        temp = []
        # if not self.index2bio:
        # count = defaultdict(int)
        
        # for k,v in self.pos2index.items():
        #     count[k.count("_")+1] += 1
        #     if k.count("_") + 1 >= 3:
        #         print(k)
        # print(count)
        # exit()
        # print(self.index2pos.values())
        # exit()
        self.index2bio = {}
        # for k,v in self.tag_bio.items():
            # self.index2bio[v] = k
        for t_ in tag:
            if t_ == 0:
                break
            # assert t_ in self.index2pos
            # print(self.index2pos[t_])
            # print(t_,bio_)
            if t_ in self.index2pos:#(self.index2bio[bio_] == "B" or self.index2bio == "I") and self.index2pos[t_] != "O":
                result.append(self.index2pos[t_])
            else:
                if self.index2pos[t_] == "O":
                    result.append("O")
                else:
                    result.append("UNK")
        return result


class TokenPosVocab:
    def __init__(self,maxlen):
        self.word2index = {"[PAD]":0,"[UNK]":1,"[SOS]":2,"[EOS]":3,"+":4," ":5}
        self.index2word = {v:k for k,v in self.word2index.items()}
        self.pos2index = {"[PAD]":0,"[SOS]":1,"[EOS]":2,"+":3,"O":4}
        self.index2pos = {v:k for k,v in self.pos2index.items()}
        self.wordindex = 6
        self.posindex = 5
        self.maxlen = maxlen 
    
    def make_dict(self,ht_sentence,pos_sentence):
        for p in pos_sentence.split():
            # h = list(h.strip().replace("+", " "))#.split()
            p = p.strip().replace("+", " _").split()
            # p =
            for pp in p:
                pp = tagged_reduce(pp)
                if ("B_" + pp) not in self.pos2index:
                    self.pos2index["B_"+pp] = self.posindex
                    self.index2pos[self.posindex] = "B_"+pp
                    self.posindex += 1
                if ("I_" + pp) not in self.pos2index:
                    self.pos2index["I_"+pp] = self.posindex
                    self.index2pos[self.posindex] = "I_"+pp
                    self.posindex += 1
        
        for hh in list(ht_sentence.strip().replace("+", " ")):        
        # for hh in h:
            if hh not in self.word2index:
                self.word2index[hh] = self.wordindex
                self.index2word[self.wordindex] = hh
                self.wordindex += 1
    
    def make_onehot(self,ht_sentence,pos_sentence):
        resultword = []
        resultpos = []
        tag = []
        h_ = ht_sentence.strip().replace("+", " ").split()
        p_ = pos_sentence.strip().replace("+", " _").split()
        for h, p in zip(h_,p_):

            flag = p.startswith("_")
            p = tagged_reduce(p)
            # print(p)
            t1 = ["B"+"_"+p]
            t2 = ["I"+"_"+p for _ in range(len(h)-1)]
            for tt in t1+t2:
                resultpos.append(self.pos2index[tt])
                tag.append(tt)
            if not flag:
                resultpos.append(self.pos2index["O"])
                tag.append("O")
           
        for h in list(ht_sentence.replace("+", "")):    
            # for hh in h:
            if h not in self.word2index:
                resultword.append(self.word2index["[UNK]"])
            else:
                resultword.append(self.word2index[h])
        if resultpos[-1] == " ":
            resultpos = resultpos[:-1]
        if tag[-1] == " ":
            tag = tag[:-1]
            
        return resultword, resultpos
    
    def make_batch(self,batch_data_tk,batch_data_p):
        words = []
        poss = []
        for bdsenttk,bdsentp in zip(batch_data_tk,batch_data_p):
            wordidx, posidx = self.make_onehot(bdsenttk,bdsentp)
            
            wordidx = [self.word2index["[SOS]"]] + wordidx[:self.maxlen-2] + [self.word2index["[EOS]"]] 
            wordidx = wordidx + [self.word2index["[PAD]"]] * (self.maxlen - len(wordidx[:self.maxlen]))
            posidx = [self.pos2index["[SOS]"]] + posidx[:self.maxlen-2] + [self.pos2index["[EOS]"]] 
            posidx = posidx + [self.pos2index["[PAD]"]] * (self.maxlen - len(posidx[:self.maxlen]))
            words.append(wordidx)
            poss.append(posidx)
            
        return words,poss
    
    def make_onehot2(self,ht_sentence):
        resultword = []
        resultpos = []
        
        for h in list(ht_sentence.strip().replace("+", "")):
            if h not in self.word2index:
                resultword.append(self.word2index["[UNK]"])
            else:
                resultword.append(self.word2index[h])
                   
        return resultword

    def make_batch2(self,batch_data_tk):
        words = []
        poss = []
        for bdsenttk in batch_data_tk:
            wordidx = self.make_onehot2(bdsenttk)
            
            wordidx = [self.word2index["[SOS]"]] + wordidx[:self.maxlen-2] + [self.word2index["[EOS]"]] 
            wordidx = wordidx + [self.word2index["[PAD]"]] * (self.maxlen - len(wordidx[:self.maxlen]))
            words.append(wordidx)
            
        return words

    def to_tag(self,tag):
        result = []
        temp = []
        for t_ in tag:
            if t_ == 0:
                break
            result.append(self.index2pos[t_])

        return result

if __name__ == "__main__":
    vocab = BiPosVocab(maxlen=230)
    mp = "통합보건교육+은 이 대학+만의 특화+된 프로그램+이다 ."
    tp = "NNG+JX MM NNG+JX_JKG NNP+VV_ETM NNG+VCP_EF SF"

    vocab.make_dict(mp,tp)
    vocab.make_onehot2(mp)
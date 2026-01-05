import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from huggingface_hub import login

token = ''
login(token)
# 1. 모델과 토크나이저 로드
model_name = "google/gemma-2-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name,device_map="cuda")
model = AutoModelForCausalLM.from_pretrained(model_name,device_map="cuda")

# 2. 새로운 토큰 추가
new_tokens = set()
with open("C:\\Users\\ty341\\OneDrive\\Desktop\\kjm\\runpod\\runpod\\transformer\\last_dict.txt",encoding="utf-8") as f:
    for nt in f:
        if nt.strip() == "":
            continue
        new_tokens.add(nt.strip())
candidate_new_tokens = list(new_tokens)
new_tokens = []
for token in candidate_new_tokens:
    # 토큰이 이미 토크나이저에 있는지 확인
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id == tokenizer.unk_token_id:  # 토큰이 없으면 unk_token_id를 반환함
        print(f"new token {token}")
        new_tokens.append(token)
    # else:
    #     print(f"토큰 '{token}'은(는) 이미 토크나이저에 있습니다. (ID: {token_id})")

# num_added_tokens = tokenizer.add_tokens(new_tokens)
# print(f"추가된 토큰 수: {num_added_tokens}")

# 5. 모델의 임베딩 레이어 크기 조정 (새 토큰이 추가된 경우에만)

import os
dataset = os.environ["DATASET"]
d1 = os.path.join(dataset,"naverblog_sentsplit250205.txt")
d2 = os.path.join(dataset,"naverblog_sentsplit250213.txt")
d3 = os.path.join(dataset,"naverblog_sentsplit250307.txt")

train_texts = []
for path in [d1,d2,d3]:
    with open(path,encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if l == "":
                continue
            train_texts.append(l)
# train_texts = [
#     "이것은 새단어1 사용 예시입니다.",
#     "새단어2와 새단어3을 문장에서 사용합니다.",
#     "새단어1은 특정 개념을 나타냅니다.",
#     # 더 많은 예시 추가 (충분한 데이터 필요)
# ]


# # 임베딩 레이어 크기 조정
# model.resize_token_embeddings(len(tokenizer))

# # 모든 파라미터 학습 비활성화
# for param in model.parameters():
#     param.requires_grad = False


# input_embeddings = model.get_input_embeddings()

# # 임베딩 레이어는 전체적으로 requires_grad=True로 설정
# input_embeddings.weight.requires_grad = True
# # 입력 임베딩 레이어 가져오기


import typing
import transformers

# 커스텀 트레이너 클래스 생성
class CustomTrainer(Trainer):
    def __init__(
        self,
        new_tokens,
        tokenizer,
        num_added_tokens,
        model,
        args,
        train_dataset,
        
    ):
        
        super(CustomTrainer,self).__init__(
            model=model,
            args=args,
            train_dataset=train_dataset
        )
        self.new_tokens = new_tokens
        self.tokenizer = tokenizer
        self.num_added_tokens = num_added_tokens
        
        
    def create_optimizer(self):
        """모든 옵티마이저 파라미터 그룹을 커스터마이징"""
        # 기본 옵티마이저 생성
        
        new_token_ids = range(len(self.tokenizer) - self.num_added_tokens, len(self.tokenizer))
        
        opt_model = self.model_wrapped if hasattr(self, "model_wrapped") else self.model
        
        # 기본 no_decay 설정
        decay_parameters = self.get_decay_parameter_names(opt_model) if hasattr(self, "get_decay_parameter_names") else []
        
        # 임베딩 레이어 모든 파라미터 가져오기
        embedding_parameters = []
        other_parameters = []
        
        # 파라미터 분류
        for name, param in opt_model.named_parameters():
            # 학습 불가능한 파라미터는 무시
            if not param.requires_grad:
                continue
                
            # 임베딩 파라미터 식별
            if "embed_tokens" in name or "word_embeddings" in name or "embeddings.word" in name:
                embedding_parameters.append((name, param))
            else:
                other_parameters.append((name, param))
        
        # 새로운 옵티마이저 파라미터 그룹 설정
        optimizer_grouped_parameters = []
        
        # 임베딩 레이어에 대한 그래디언트 후크 설정
        if embedding_parameters:
            for name, param in embedding_parameters:
                # 특별한 학습률과 가중치 감소 설정으로 임베딩 레이어 추가
                # 그래디언트 후크를 사용하여 새 토큰만 업데이트되도록 설정
                self._add_gradient_mask_hook(param, new_token_ids)
                
                # 임베딩 파라미터를 옵티마이저에 추가
                optimizer_grouped_parameters.append({
                    "params": [param],
                    "weight_decay": 0.0,  # 일반적으로 임베딩에는 가중치 감소를 적용하지 않음
                })
                print(f"임베딩 파라미터 추가: {name}")
        
        # 기타 파라미터도 추가 (있는 경우)
        for name, param in other_parameters:
            # no_decay 적용 여부 확인
            weight_decay = 0.0 if any(nd in name for nd in decay_parameters) else self.args.weight_decay
            optimizer_grouped_parameters.append({
                "params": [param],
                "weight_decay": weight_decay,
            })
            print(f"기타 파라미터 추가: {name}")
        
        # 최종 옵티마이저 생성
        optimizer_cls = torch.optim.AdamW
        optimizer_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
            "lr": self.args.learning_rate,
        }
        optimizer = optimizer_cls(params= optimizer_grouped_parameters, **optimizer_kwargs)
        # print(optimizer.pa)
        return optimizer
    
    def _add_gradient_mask_hook(self, tensor, token_ids):
        """새 토큰만 그래디언트를 받도록 하는 후크 추가"""
        def grad_mask_hook(grad):
            # 새 토큰 위치만 1, 나머지는 0으로 된 마스크 생성
            mask = torch.zeros_like(grad)
            for idx in token_ids:
                mask[idx] = 1.0
            return grad * mask
        
        # 텐서에 후크 등록 (역전파 시 호출됨)
        tensor.register_hook(grad_mask_hook)
        print(f"그래디언트 마스크 후크 등록 (토큰 ID: {token_ids})")
    # def create_optimizer(self):
    #     # 새 토큰 인덱스 가져오기
    #     new_token_ids = range(len(self.tokenizer) - self.num_added_tokens, len(self.tokenizer))
    #     # for ids in new_token_ids:
    #     #     print(ids)
    #     # exit()
    #     # 전체 임베딩 레이어에서 학습하고자 하는 특정 임베딩만 파라미터로 지정
    #     embedding_params = []
    #     other_params = []
        
    #     for name, param in self.model.named_parameters():
    #         if param.requires_grad:
    #             if "embed_tokens" in name or "embeddings" in name:  # 모델에 따라 이름이 다를 수 있음
    #                 # 임베딩 레이어 파라미터는 나중에 필터링하기 위해 별도로 저장
    #                 embedding_params.append((name, param))
    #             else:
    #                 other_params.append((name, param))
        
    #     # 옵티마이저에 전달할 파라미터 그룹 준비
    #     optimizer_grouped_parameters = []
                
    #     # 새 토큰 임베딩만 학습하기 위한 마스크 생성 및 적용
    #     if embedding_params:
    #         name, embed_weight = embedding_params[0]  # 임베딩 레이어 가져오기
    #         # 모든 파라미터를 먼저 0 학습률로 설정
    #         optimizer_grouped_parameters.append({
    #             "params": [embed_weight],
    #             "lr": 0.0,  # 기존 토큰은 학습하지 않음
    #         })
            
    #         # 새 토큰만 학습하는 파라미터 그룹 추가
    #         # 새 토큰에 해당하는 임베딩 파라미터만 가져오기
    #         new_token_params = []
            
    #         for i in new_token_ids:
    #             # print(i)
    #             new_token_params.append(embed_weight[i])
            
    #         if new_token_params:
    #             optimizer_grouped_parameters.append({
    #                 "params": new_token_params,
    #                 "lr": self.args.learning_rate,  # 새 토큰은 정상 학습률 적용
    #             })
    #             print(f"새 토큰 학습 파라미터 수: {sum(p.numel() for p in new_token_params)}")
        
    #     # 나머지 파라미터(있다면)도 추가
    #     for name, param in other_params:
    #         optimizer_grouped_parameters.append({
    #             "params": [param],
    #         })
        
    #     # 옵티마이저 생성
    #     optimizer = torch.optim.AdamW(
    #         optimizer_grouped_parameters,
    #         lr=self.args.learning_rate,
    #     )
        
    #     return optimizer
# ??? ??? ??
num_added_tokens = 0
if new_tokens:
    num_added_tokens = tokenizer.add_tokens(new_tokens)
    print(f"??? ?? ?: {num_added_tokens}")
    print(f"??? ?? ??: {new_tokens}")
else:
    print("??? ??? ??? ????.")


# ?? ???? ?? ????
for param in model.parameters():
    param.requires_grad = False

# ??? ??? ???? ? ?? ???
input_embeddings = model.get_input_embeddings()
input_embeddings.weight.requires_grad = True

# ????? ???? ?? ?? ?? ??
def gradient_mask_hook(grad):
    mask = torch.zeros_like(grad)
    for idx in new_token_ids:
        mask[idx] = 1.0
    return grad * mask
import numpy as np
import copy
def tokenize_function(examples):
    tokes = tokenizer(examples["text"], padding="max_length",max_length=100, truncation=True,return_tensors="pt")
    # print(tokes.keys()).
    # print(type(tokes["input_ids"].half()))
    tokes["input_ids"] = tokes["input_ids"].long()#half().numpy()#.type(torch.BFloat16Tensor).to("cuda")
    tokes["attention_mask"] = tokes["attention_mask"].long()#half().numpy()#.to("cuda")#type(torch.bfloat16).to("cuda")
    # tokes = tokes.to("cuda",dtype=torch.bfloat16)
    # exit()
    # tokes = torch.tensor(tokes,dtype=torch.float16).to("cuda")
    tokes["labels"] = tokes["input_ids"].clone()#.clone().half().numpy()#.to("cuda")#.type(torch.BFloat16Tensor).to("cuda")
    
    # numpy ??? Python ???? ??
    # for key in tokes:
    #     if isinstance(tokes[key], np.ndarray):
    #         tokes[key] = torch.tensor(tokes[key].tolist())
    
    # ??? ?? (input_ids ??)
    # tokes["labels"] = tokes["input_ids"].clone()#[ids[:] for ids in tokes["input_ids"]]
    # exit()
    return tokes
# print(train_texts)
train_dataset = Dataset.from_dict({"text": train_texts})

tokenized_dataset = train_dataset.map(tokenize_function,remove_columns=["text"], batched=True)
# print(tokenized_dataset)
# tokenized_dataset = tokenized_dataset.remove_columns("text")
# tokenized_dataset.rename_columns("input_ids")
# print(tokenized_dataset)
# exit()
# print(11111111)
# ??? ??? ??? ?? ??
original_embedding_size = model.get_input_embeddings().weight.size(0)
model.resize_token_embeddings(len(tokenizer))
print(f"?? ??? ??: {original_embedding_size}, ? ??? ??: {len(tokenizer)}")

# ? ??? ID ??
new_token_ids = list(range(len(tokenizer) - num_added_tokens, len(tokenizer)))
print(f"? ?? ID ??: {new_token_ids}")
print(num_added_tokens)
# exit()
# ?? ??
# hook = input_embeddings.weight.register_hook(gradient_mask_hook)

print("????? ??? ??? ???????.")
# 7. 학습 설정 및 시작
# training_args = TrainingArguments(
#     output_dir="./results",
#     per_device_train_batch_size=2,
#     num_train_epochs=5,
#     save_steps=500,
#     logging_dir="./logs",
#     logging_steps=100,
#     # tokenzier=tokenizer,
#     learning_rate=1e-3,  # 임베딩만 학습하므로 더 높은 학습률 사용 가능
# )
ds_config = {
    "train_batch_size": 16,  # 전체 배치 크기 (per_device_train_batch_size * gradient_accumulation_steps * num_gpus)
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-3,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.0
        }
    },
    "zero_optimization": {
        "stage": 2,  # Stage 2: 파라미터와 그래디언트를 여러 GPU에 분산
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "contiguous_gradients": True,
        "overlap_comm": True
    },
    "fp16": {
        "enabled": True,
        "loss_scale": 0,  # 동적 손실 스케일링
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    }
}
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    num_train_epochs=5,  # ??? ??? ??
    learning_rate=1e-3,    # ???? ????? ? ?? ??? ?? ??
    logging_steps=100,
    save_strategy="steps",
    save_steps=300,
    save_total_limit=7,
    # fp16=True,             # ??? ??? ?? ?? ??? ??
    optim="adamw_torch",
    seed=42,
    fp16=True,
    deepspeed=ds_config,
    local_rank=-1,
    # device="cuda",
    remove_unused_columns=True,  # ?? ?? ? ??
    logging_dir="./logs"         # ?? ????
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    processing_class=tokenizer,
)
# trainer = Trainer()
# 커스텀 트레이너 사용
# trainer = CustomTrainer(
#     new_tokens,
#     tokenizer,
#     num_added_tokens,
#     model=model,
#     args=training_args,
#     train_dataset=tokenized_dataset,
# )
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"학습 가능한 파라미터: {trainable_params}")
print("train")
trainer.train()
print("save")
# 8. 학습된 모델 저장
model.save_pretrained("./gemma2-7b-new-tokens")
tokenizer.save_pretrained("./gemma2-7b-new-tokens")
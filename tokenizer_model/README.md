# Tokenizer

## Tokenizer Dataset Create
### Head-Tail 형태소 분석 데이터셋으로 부터 Head-Tail Tokenizer Dataset 생성 (일반 음절일 경우 라벨:0, 띄어쓰기 음절일 경우 라벨:1, Head-Tail 경계 음절일 경우 라벨:2)
* Head-Tail Tokenizer 데이터셋 생성
```c
usage: make_ht_tk_dataset.py [-h] [--input INPUT] [--output OUTPUT]

Tokenizer Dataset Create

optional arguments:
  -h, --help       show this help message and exit
  --input INPUT    input head-tail dataset (Head-Tail 형태소 분석 Tokenizer)
  --output OUTPUT  output file name -> output file name:ht -> httk_x.txt, httk_y.txt (Output File Name 지정 ht_train 지정시 -> ht_train_x.txt, ht_train_y.txt 생성)
```
* Default Arguments
  * --input tk_train_data.txt
  * --output httk

* 학습데이터파일 Numpy 학습데이터로 변환
```c
usage: tk_dataset_to_numpy.py [-h] [--inputX INPUTX] [--inputY INPUTY] [--BATCH BATCH] [--MAX_LEN MAX_LEN]

Tokenizer Dataset to Train numpy dataset Create

optional arguments:
  -h, --help         show this help message and exit
  --inputX INPUTX    train X data path (Head-Tail 형태소 분석 파일 경로)
  --inputY INPUTY    train Y data path (Head-Tail 라벨파일 경로: make_ht_tk_dataset.py의 *_y.txt 파일)
  --BATCH BATCH      Train Data BATCH SIZE
  --MAX_LEN MAX_LEN  Sequence Data MAX LENGTH (입력 음절의 최대길이)
```
* Default Arguments
  * --inputX kcc150_autospace_line.txt
  * --inputY kcc150_autospace.txt
  * --BATCH 50
  *--MAX_LEN 300
  
* 학습에 사용된 Numpy Dataset [다운로드](https://drive.google.com/file/d/1fqdUo11f5k3qPkWx1XI4C6pikIsai9cY/view?usp=sharing)]

## Tokenizer 학습
* python .\train_bilstm_bigram_model.py
```c
usage: train_bilstm_bigram_model.py [-h] [--BATCH BATCH] [--MAX_LEN MAX_LEN] [--epoch_step EPOCH_STEP]
                                    [--validation_step VALIDATION_STEP] [--EPOCH EPOCH] [--model_name MODEL_NAME]
                                    [--GPU_NUM GPU_NUM]

Train Head-Tail Tokenizer

optional arguments:
  -h, --help            show this help message and exit
  --BATCH BATCH         Train Data BATCH SIZE
  --MAX_LEN MAX_LEN     Sequence Data MAX LENGTH (입력 시퀀스의 최대 길이 "학습데이터파일 Numpy 학습데이터로 변환"에서 설정한 MAX_LEN 값)
  --epoch_step EPOCH_STEP
                        Train Data Epoch Step (학습할 배치의 개수)
  --validation_step VALIDATION_STEP
                        Validation Data Epoch Step (검증할 배치의 개수)
  --EPOCH EPOCH         Train Epoch SIZE
  --model_name MODEL_NAME
                        Tokenizer Model Name (저장할 모델명)
  --GPU_NUM GPU_NUM     Train GPU NUM (사용할 GPU Number)
```
* Default Arguments
  * --BATCH 50
  * --MAX_LEN 300
  * --epoch_step 4000
  * --validation_step 400
  * --EPOCH 6
  * --model_name lstm_bigram_tokenizer.model
  * --GPU_NUM 0

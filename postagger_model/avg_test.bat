@echo off
setlocal enabledelayedexpansion

@REM set num=0
for /L %%a in (1,1,20) do (
    @REM set /a realepoch=%%a
    echo %%a
    @REM set /a result=%%a+1
    @REM echo !result!-1
    @REM echo !num!+1

    uv run train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path transformer_encoder3 --BATCH 128 --infer --sepoch %%a
    echo "--------------------------------------"
)
@REM python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path ht_postagger_model --BATCH 50 --infer
@REM pause > nul

@REM echo 10
@REM python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path ht_postagger_model --BATCH 128 --infer --sepoch 10
@REM echo 11
@REM python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path ht_postagger_model --BATCH 128 --infer --sepoch 11
@REM echo 12
@REM python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path ht_postagger_model --BATCH 128 --infer --sepoch 12
@REM echo 13
@REM python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path ht_postagger_model --BATCH 128 --infer --sepoch 13
@REM echo 14
@REM python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path ht_postagger_model --BATCH 128 --infer --sepoch 14
@REM echo 15
@REM python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path ht_postagger_model --BATCH 128 --infer --sepoch 15
@REM echo 16
@REM python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path ht_postagger_model --BATCH 128 --infer --sepoch 16
@REM echo 17
@REM python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path ht_postagger_model --BATCH 128 --infer --sepoch 17
@REM echo 18
@REM python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path ht_postagger_model --BATCH 128 --infer --sepoch 18
@REM echo 19
@REM python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path ht_postagger_model --BATCH 128 --infer --sepoch 19
@REM echo 20
@REM python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path ht_postagger_model --BATCH 128 --infer --sepoch 20
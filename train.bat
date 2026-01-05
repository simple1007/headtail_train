@echo off
@REM echo "args: %1"
set input=%1
set batch_size=128
set mincount=500
set epoch=50
set prefix="xstags_biencoder"
@REM echo "1 %input%"
if "%input%"=="prepro" (
    @REM echo "%input%"
    python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path nosf_enc_postagger_nobi_nogqa_minc3 --lr 3e-4 --BATCH %batch_size% --EPOCH %epoch% --eval_step 1000 --prepro_flag --mincount %mincount%
) else (
    @REM echo "no"
    python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path %prefix% --lr 6e-4 --BATCH %batch_size% --EPOCH %epoch% --eval_step 1500 --mincount %mincount%
)
@REM pause
@REM uv run train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path nosf_lstm_postagger --lr 2e-4 --BATCH 50 --EPOCH 5 --eval_step 1000 

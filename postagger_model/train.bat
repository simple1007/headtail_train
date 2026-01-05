@echo off
@REM echo "args: %1"
set input=%1
set batch_size=128
set mincount=21
set epoch=20
set sepoch=0
set prefix="transformer_encoder3"
set eval_step=500
set lr="1e-3"
@REM echo "1 %input%"
if "%input%"=="prepro" (
    @REM echo "%input%"
    uv run train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path nosf_enc_postagger_nobi_nogqa_minc3 --lr 3e-4 --BATCH %batch_size% --EPOCH %epoch% --eval_step %eval_step% --prepro_flag --mincount %mincount%
) else (
    @REM echo "no"
    uv run train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path %prefix% --lr %lr% --BATCH %batch_size% --EPOCH %epoch% --eval_step %eval_step% --mincount %mincount% --sepoch %sepoch%
)
@REM pause
@REM uv run train_tkbigram_pos_tagger_pytorch_peft_bi.py --sepoch 4 --subword --model_path nosf_lstm_postagger --lr 2e-4 --BATCH 50 --EPOCH 5 --eval_step 1000 

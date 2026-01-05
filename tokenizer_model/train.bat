@echo off
@REM echo "args: %1"
set batch_size=128
set epoch=20
set sepoch=0
@REM set prefix="transformer_pre_encoder"
set prefix="lstm_httokenizer"
set lr="5e-4"
set maxlen=400
set eval_step=500
uv run train_bilstm_bigram_model_pytorch.py --sepoch %sepoch% --model_path %prefix% --lr %lr% --BATCH %batch_size% --EPOCH %epoch% --MAX_LEN %maxlen% --eval_step %eval_step%

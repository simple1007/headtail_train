@echo off
setlocal enabledelayedexpansion

@REM set num=0
for /L %%a in (1,1,15) do (
    @REM set /a realepoch=%%a
    echo %%a
    @REM set /a result=%%a+1
    @REM echo !result!-1
    @REM echo !num!+1

    uv run train_bilstm_bigram_model_pytorch.py --model_path lstm_httokenizer --BATCH 128 --infer --sepoch %%a --MAX_LEN 400
)
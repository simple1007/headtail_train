echo %DATASET%
uv run head_tail2.py --input %DATASET%\naverblog_autospace250205.txt --output %DATASET%\headtail_250205_ht5.txt
uv run head_tail2.py --input %DATASET%\naverblog_autospace250213.txt --output %DATASET%\headtail_250213_ht5.txt
uv run head_tail2.py --input %DATASET%\naverblog_autospace250307.txt --output %DATASET%\headtail_250307_ht5.txt
uv run head_tail2.py --input %DATASET%\kccword2vec_data.txt --output %DATASET%\kccword2vec_data_ht5.txt
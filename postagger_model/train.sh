input=$1
batch_size=128
mincount=500
epoch=50
prefix="nlayer3_dmodel384_adagrad"
if [[ "$input" == "prepro" ]]; then
    echo "input: $input"
    echo "python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path nosf_enc_postagger_nobi_nogqa_minc3 --lr 3e-4 --BATCH $batch_size --EPOCH $epoch --eval_step 1000 --prepro_flag --mincount $mincount"
    python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path nosf_enc_postagger_nobi_nogqa_minc3 --lr 3e-4 --BATCH $batch_size --EPOCH $epoch --eval_step 1000 --prepro_flag --mincount $mincount
else
    echo "input: no"
    echo "python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path $prefix --lr 6e-4 --BATCH $batch_size --EPOCH $epoch --eval_step 1500 --mincount $mincount"
    python train_tkbigram_pos_tagger_pytorch_peft_bi.py --subword --model_path $prefix --lr 2e-4 --BATCH $batch_size --EPOCH $epoch --eval_step 1000 --mincount $mincount
fi

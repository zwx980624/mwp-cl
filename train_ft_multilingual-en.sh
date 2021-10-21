#!/bin/bash

python3 run_ft.py \
    --output_dir output/mwp-ft-multilingual-en \
    --bert_pretrain_path /cfs/cfs-fwcdwxrr/neutrali/workspace/pretrained_models/bert-base-multilingual-uncased \
    --model_reload_path output/mwp-cl-multilingual/epoch_19 \
    --data_dir data \
    --finetune_from_trainset Math_23K-MathQA_mbert_token_train.json \
    --train_file MathQA_mbert_token_train.json \
    --dev_file MathQA_mbert_token_val.json \
    --test_file MathQA_mbert_token_test.json \
    --n_val 1 --n_save_ckpt 10 --schedule linear --batch_size 16 --learning_rate 1e-4 --n_epochs 50 \
    --warmup_steps 4500 --hidden_size 768 --beam_size 3 --dropout 0.5 --seed 17
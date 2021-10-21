#!/bin/bash

python3 run_ft.py \
    --output_dir output/mwp-ft-monolingual-en \
    --bert_pretrain_path pretrained_models/bert-base-uncased \
    --model_reload_path output/mwp-cl-monolingual-en/epoch_29 \
    --data_dir data \
    --train_file MathQA_bert_token_train.json \
    --finetune_from_trainset MathQA_bert_token_train.json \
    --dev_file MathQA_bert_token_val.json \
    --test_file MathQA_bert_token_test.json \
    --schedule linear \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --n_epochs 50 \
    --warmup_steps 3500 \
    --n_save_ckpt 3 \
    --n_val 5 \
    --logging_steps 100 \
    --embedding_size 128 \
    --hidden_size 768 \
    --beam_size 3 \
    --dropout 0.5 \
    --seed 17

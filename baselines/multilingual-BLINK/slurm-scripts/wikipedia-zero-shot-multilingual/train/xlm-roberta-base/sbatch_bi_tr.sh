#!/bin/bash

DATA=$PWD/data/wikipedia-zero-shot-multilingual

cd multilingual-BLINK

# bi-encoder train
PYTHONPATH=. python blink/biencoder/train_biencoder.py \
    --data_path $DATA \
    --output_path models/wikipedia-zero-shot-multilingual/xlm-roberta-base/biencoder \
    --learning_rate 1e-05 \
    --num_train_epochs 5 \
    --max_context_length 128 \
    --max_cand_length 128 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --bert_model xlm-roberta-base \
    --type_optimization all_encoder_layers \
    --shuffle \
    --print_interval 500
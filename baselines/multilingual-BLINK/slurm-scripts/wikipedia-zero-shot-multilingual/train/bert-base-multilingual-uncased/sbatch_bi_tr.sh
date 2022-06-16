#!/bin/bash

DATA=$PWD/data

cd multilingual-BLINK

# bi-encoder train
PYTHONPATH=. python blink/biencoder/train_biencoder.py \
    --data_path $DATA/wikipedia-zero-shot-multilingual/ \
    --output_path models/wikipedia-zero-shot-multilingual/bert-base-multilingual-uncased/biencoder \
    --learning_rate 1e-05 \
    --num_train_epochs 5 \
    --max_context_length 128 \
    --max_cand_length 128 \
    --train_batch_size 128 \
    --eval_batch_size 64 \
    --bert_model bert-base-multilingual-uncased \
    --type_optimization all_encoder_layers \
    --gradient_accumulation_steps 4 \
    --shuffle \
    --print_interval 500
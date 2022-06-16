#!/bin/bash

DATA=$PWD/data/wikipedia-zero-shot-multilingual
cd multilingual-BLINK

BERT=bert-base-multilingual-uncased
MODEL_DIR=models/wikipedia-zero-shot-multilingual/$BERT/

# bi-encoder predict
PYTHONPATH=. python blink/biencoder/eval_biencoder_multilingual.py \
    --path_to_model ${MODEL_DIR}/biencoder/pytorch_model.bin \
    --data_path $DATA \
    --output_path ${MODEL_DIR} \
    --encode_batch_size 32 \
    --eval_batch_size 32 \
    --top_k 16 \
    --save_topk_result \
    --bert_model $BERT \
    --mode dev,train \
    --event_dict_path $DATA/label_dict.jsonl

TOPK=${MODEL_DIR}/top16_candidates/
OUT_CROSS=${MODEL_DIR}/crossencoder

# cross-encoder train
PYTHONPATH=. python blink/crossencoder/train_cross.py \
    --data_path $DATA \
    --topk_path $TOPK \
    --output_path $OUT_CROSS \
    --learning_rate 2e-05 \
    --num_train_epochs 5 \
    --max_context_length 128 \
    --max_cand_length 128 \
    --train_batch_size 6 \
    --eval_batch_size 4 \
    --bert_model $BERT \
    --type_optimization all_encoder_layers \
    --add_linear \
    --print_interval 500 \
    --event_dict_path $DATA/label_dict.jsonl \
    --top_k 8
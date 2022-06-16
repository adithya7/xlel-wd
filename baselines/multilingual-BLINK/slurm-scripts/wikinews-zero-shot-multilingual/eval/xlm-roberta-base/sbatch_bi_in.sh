#!/bin/bash

DATA=$PWD/data/wikinews-zero-shot-multilingual

cd multilingual-BLINK

BERT=xlm-roberta-base
MODEL=models/wikipedia-zero-shot-multilingual/$BERT
OUT=$MODEL/wikinews-zero-shot-multilingual_retrieve_out

# bi-encoder predict
PYTHONPATH=. python blink/biencoder/eval_biencoder_multilingual.py \
    --path_to_model $MODEL/biencoder/pytorch_model.bin \
    --data_path $DATA \
    --output_path $OUT \
    --encode_batch_size 32 \
    --eval_batch_size 32 \
    --top_k 8 \
    --save_topk_result \
    --bert_model $BERT \
    --mode test \
    --event_dict_path $DATA/label_dict.jsonl
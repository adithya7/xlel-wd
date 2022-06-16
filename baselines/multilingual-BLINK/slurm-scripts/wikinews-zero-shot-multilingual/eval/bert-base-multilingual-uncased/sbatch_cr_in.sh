#!/bin/bash

DATA=$PWD/data/wikinews-zero-shot-multilingual

cd multilingual-BLINK

BERT=bert-base-multilingual-uncased
MODEL=models/wikipedia-zero-shot-multilingual/$BERT
TOPK=$MODEL/wikinews-zero-shot-multilingual_retrieve_out/top8_candidates
OUT=$MODEL/wikinews-zero-shot-multilingual_rank_out

# cross-encoder predict
PYTHONPATH=. python blink/crossencoder/eval_cross.py \
    --path_to_model $MODEL/crossencoder/pytorch_model.bin \
    --data_path $DATA \
    --topk_path $TOPK \
    --output_path $OUT \
    --max_context_length 128 \
    --max_cand_length 128 \
    --eval_batch_size 2 \
    --bert_model $BERT \
    --add_linear \
    --mode test \
    --event_dict_path $DATA/label_dict.jsonl \
    --save_preds \
    --top_k 8
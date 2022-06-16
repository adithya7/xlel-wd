#!/bin/bash

DATA=$PWD/data/wikipedia-zero-shot-multilingual

cd multilingual-BLINK

BERT=xlm-roberta-base
MODEL=models/wikipedia-zero-shot-multilingual/$BERT/crossencoder/pytorch_model.bin
TOPK=models/wikipedia-zero-shot-multilingual/$BERT/retrieve_out/top8_candidates
OUT=models/wikipedia-zero-shot-multilingual/$BERT/rank_out
DICT=$DATA/label_dict.jsonl

# cross-encoder predict
PYTHONPATH=. python blink/crossencoder/eval_cross.py \
    --path_to_model $MODEL \
    --data_path $DATA \
    --topk_path $TOPK \
    --output_path $OUT \
    --max_context_length 128 \
    --max_cand_length 128 \
    --eval_batch_size 2 \
    --bert_model $BERT \
    --add_linear \
    --mode dev,test \
    --event_dict_path $DICT \
    --save_preds \
    --top_k 8
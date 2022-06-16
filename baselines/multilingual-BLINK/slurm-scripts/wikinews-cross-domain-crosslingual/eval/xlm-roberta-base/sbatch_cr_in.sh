#!/bin/bash

DATA=$PWD/data/wikinews-cross-lingual-crosslingual

cd multilingual-BLINK

BERT=xlm-roberta-base
MODEL=models/wikipedia-zero-shot-crosslingual/$BERT
TOPK=$MODEL/wikinews-cross-domain-crosslingual_retrieve_out/top8_candidates
OUT=$MODEL/wikinews-cross-domain-crosslingual_rank_out

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
#!/bin/bash

#SBATCH --time=0
#SBATCH --mem=3000
#SBATCH --output=slurm-%j.out

DATA=$PWD/data

cd bm25_ranker

TOKENIZER=bert-base-multilingual-uncased
LOG=logs_test/$TOKENIZER
PRED=predictions_test/$TOKENIZER
mkdir -p $LOG $PRED

BM25="BM25Plus"
MAX_LEN=16

python bm25_ranker.py \
    $DATA/wikipedia-zero-shot-multilingual/test.jsonl \
    $DATA/wikipedia-zero-shot-multilingual/label_dict.jsonl \
    $PRED/test_preds_context-${MAX_LEN}_$BM25.jsonl \
    $LOG/log_context-${MAX_LEN}_$BM25.txt \
    --bm25 $BM25 \
    --bert-tokenizer $TOKENIZER \
    --max-seq-len $MAX_LEN
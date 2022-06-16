#!/bin/bash

#SBATCH --time=0
#SBATCH --mem=3000
#SBATCH --output=slurm-%j.out

DATA=$PWD/data

cd bm25_ranker

TOKENIZER=bert-base-multilingual-uncased
LOG=logs_wn_cross-domain_test/$TOKENIZER
PRED=predictions_wn_cross-domain_test/$TOKENIZER
mkdir -p $LOG $PRED

BM25="BM25Plus"
MAX_LEN=16

python bm25_ranker.py \
    $DATA/wikinews-cross-domain-multilingual/test.jsonl \
    $DATA/wikinews-cross-domain-multilingual/label_dict.jsonl \
    $PRED/dev_preds_context-${MAX_LEN}_$BM25.jsonl \
    $LOG/log_context-${MAX_LEN}_$BM25.txt \
    --bm25 $BM25 \
    --bert-tokenizer $TOKENIZER \
    --max-seq-len $MAX_LEN
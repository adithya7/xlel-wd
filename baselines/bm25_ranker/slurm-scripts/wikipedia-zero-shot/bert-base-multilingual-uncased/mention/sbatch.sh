#!/bin/bash

#SBATCH --time=0
#SBATCH --mem=3000
#SBATCH --output=slurm-%j.out

DATA=$PWD/data

cd bm25_ranker

TOKENIZER=bert-base-multilingual-uncased
LOG=logs/$TOKENIZER
PRED=predictions/$TOKENIZER
mkdir -p $LOG $PRED

BM25="BM25Okapi"

python bm25_ranker.py \
    $DATA/wikipedia-zero-shot-multilingual/dev.jsonl \
    $DATA/wikipedia-zero-shot-multilingual/label_dict.jsonl \
    $PRED/dev_preds_mention_$BM25.jsonl \
    $LOG/log_mention_$BM25.txt \
    --bm25 $BM25 \
    --bert-tokenizer $TOKENIZER \
    --use-mention
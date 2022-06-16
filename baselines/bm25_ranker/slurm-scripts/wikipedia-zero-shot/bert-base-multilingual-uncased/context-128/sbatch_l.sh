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

BM25="BM25L"
MAX_LEN=128

python bm25_ranker.py \
    $DATA/zero_shot_multilingual/disjoint_sequences/dev.jsonl \
    $DATA/zero_shot_multilingual/disjoint_sequences/label_dict.jsonl \
    $PRED/dev_preds_context-${MAX_LEN}_$BM25.jsonl \
    $LOG/log_context-${MAX_LEN}_$BM25.txt \
    --bm25 $BM25 \
    --bert-tokenizer $TOKENIZER \
    --max-seq-len $MAX_LEN
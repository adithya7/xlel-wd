#!/bin/bash

VERSION=20220301

if test $# -ne 3; then
    echo "usage: bash extract_wn_text.sh <LG> <DUMPS_DIR> <OUT_DIR>"
    exit 1
fi

# en
LG=$1
# /wikinews_dumps
DATA=$2
# /wikinews_meta_info
OUT=$3

printf "processing categorylinks files"

mkdir -p $OUT/$LG

python get_publication_dates.py \
    $LG \
    $DATA/$LG/${LG}wikinews-${VERSION}-categorylinks.sql \
    $OUT/$LG/${LG}wikinews-${VERSION}-meta.tsv
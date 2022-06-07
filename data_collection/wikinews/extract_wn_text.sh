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
# /wikinews_extractor_out
OUT=$3

# create temporary working directory
TMP=$OUT/tmp
mkdir -p $TMP

printf "preprocessing XML files\n"

# preprocess XML before running wikiextractor
INPUT=$DATA/$LG/${LG}wikinews-${VERSION}-pages-meta-current.xml
PROC_INPUT=$TMP/${LG}wikinews-${VERSION}-pages-meta-current-cleaned.xml
python preprocess_wn.py $INPUT $PROC_INPUT

# run wikiextractor
TEMPLATES=$OUT/templates
OUT_DOCS=$OUT/docs/$LG

mkdir -p $OUT_DOCS $TEMPLATES

printf "extracting wikinews dump for lang: %s\n" $LG

wikiextractor \
    --json \
    --processes 12 \
    --templates $TEMPLATES/$LG \
    --output ${OUT_DOCS} \
    --bytes 1M \
    --links \
    $PROC_INPUT

# cleanup working directory
rm -rf $TMP
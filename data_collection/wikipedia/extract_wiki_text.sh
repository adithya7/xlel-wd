#!/bin/bash

LG=$1
ROOT=$2

RAW=$ROOT/dumps
VERSION=20210901

INPUT=${RAW}/${LG}wiki-${VERSION}-pages-articles.xml.bz2
TEMPLATES=$ROOT/templates
OUT=$ROOT/wikiextractor_out

mkdir -p $OUT/$LG $TEMPLATES

printf "extracting wikipedia dump for lang: %s" $LG

wikiextractor \
    --json \
    --processes 12 \
    --templates $TEMPLATES/$LG \
    --output $OUT/$LG \
    --bytes 1M \
    --links \
    $INPUT
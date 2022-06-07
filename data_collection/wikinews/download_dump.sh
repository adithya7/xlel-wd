#!/bin/bash

if test $# -ne 2; then
    echo "usage: download_dump.sh <LG> <OUT_DIR>"
    exit 1
fi

VERSION=20220301
LG=$1
OUT=$2

URL=https://dumps.wikimedia.org/${LG}wikinews/$VERSION/

echo $LG

XML=${LG}wikinews-${VERSION}-pages-meta-current.xml.bz2
wget --spider -q $URL/$XML
if test $? -eq 0; then
    mkdir -p $OUT/$LG
    wget -q --show-progress $URL/$XML -O $OUT/$LG/$XML
    bzip2 -df $OUT/$LG/$XML
fi

TITLES=${LG}wikinews-${VERSION}-all-titles.gz
wget --spider -q $URL/$TITLES
if test $? -eq 0; then
    mkdir -p $OUT/$LG
    wget -q --show-progress $URL/$TITLES -O $OUT/$LG/$TITLES
    gzip -df $OUT/$LG/$TITLES
fi

CATEGORIES=${LG}wikinews-${VERSION}-categorylinks.sql.gz
wget --spider -q $URL/$CATEGORIES
if test $? -eq 0; then
    mkdir -p $OUT/$LG
    wget -q --show-progress $URL/$CATEGORIES -O $OUT/$LG/$CATEGORIES
    gzip -df $OUT/$LG/$CATEGORIES
fi
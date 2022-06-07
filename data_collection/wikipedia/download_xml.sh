#!/bin/bash

LG=$1
OUT=$2

VERSION=20210901

printf "downloading dump dated %s for language %s\n" $VERSION $LG

URL=https://dumps.wikimedia.org/${LG}wiki/${VERSION}/${LG}wiki-${VERSION}-pages-articles.xml.bz2
LOG=dump_download.log

mkdir -p $OUT/dumps

wget -nv -a $LOG -P $OUT/dumps $URL
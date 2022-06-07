# Multilingual Event Linking

## Overview

- [Download data from 🤗 datasets](#download-data-from-🤗-datasets)
- [BM25](#bm25)
- [Multilingual BLINK](#multilingual-blink)

## Download data from 🤗 datasets

The XLEL-WD dataset can be download directly from 🤗 datasets. Run the below command to download the [event dictionary](https://huggingface.co/datasets/adithya7/xlel_wd_dictionary) and the [mentions](https://huggingface.co/datasets/adithya7/xlel_wd) for multilingual Wikinews zero-shot evaluation.

```bash
python baselines/download_data.py \
    --config wikinews-zero-shot \
    --task multilingual \
    --out-dir wn_zeshel_multilingual
```

Configuration options: `wikipedia-zero-shot`, `wikinews-zero-shot`, `wikinews-cross-domain`.
Task options: `multilingual`, `crosslingual`.

## BM25

## Multilingual BLINK

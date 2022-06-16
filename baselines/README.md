# Multilingual Event Linking

## Overview

- [Setup](#setup)
- [Download data from 🤗 datasets](#download-data-from-🤗-datasets)
- [BM25](#bm25)
- [Multilingual BLINK](#multilingual-blink)

## Setup

```bash
# install torch, transformers, datasets and rank-bm25
python -m pip install -r baselines/requirements.txt -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html
```

## Download data from 🤗 datasets

The XLEL-WD dataset can be download directly from 🤗 datasets.

Example: run the following script to download the [event dictionary](https://huggingface.co/datasets/adithya7/xlel_wd_dictionary) and the [mentions](https://huggingface.co/datasets/adithya7/xlel_wd) for multilingual Wikinews zero-shot evaluation.

```bash
# dataset is downloaded to baselines/data/wikinews-zero-shot-multilingual
python download_data.py \
    --config wikinews-zero-shot \
    --task multilingual \
    --out-dir data
```

Configuration options: `wikipedia-zero-shot`, `wikinews-zero-shot`, `wikinews-cross-domain`.
Task options: `multilingual`, `crosslingual`.

## BM25

Example run on `wikinews-zero-shot` (config) and `multilingual` (task).

```bash
bash bm25_ranker/slurm-scripts/wikinews-zero-shot/bert-base-multilingual-uncased/context-16/sbatch_plus_test.sh
```

## Multilingual BLINK

Download pretrained model checkpoints. Checkpoints for each task (multilingual and crosslingual) are about 5G size.

```bash
python multilingual-BLINK/models/download_models.py --task crosslingual --out multilingual-BLINK/models
```

Example run on `wikipedia-zero-shot` (config) and `crosslingual` (task) using `bert-base-multilingual-uncased`.

```bash
# biencoder
sbatch multilingual-BLINK/slurm-scripts/wikipedia-zero-shot-crosslingual/eval/bert-base-multilingual-uncased/sbatch_bi_in.sh
# crossencoder
sbatch multilingual-BLINK/slurm-scripts/wikipedia-zero-shot-crosslingual/eval/bert-base-multilingual-uncased/sbatch_cr_in.sh
```

# Data Collection: XLEL-WD

Steps to compile an event linking dataset using Wikidata, Wikipedia and Wikinews dumps.

## Overview

- [Install requirements](#install-requirements)
- [Download Wikimedia dumps](#download-wikimedia-dumps)
  - [Extract Wikidata](#extract-wikidata)
  - [Extract Wikipedia](#extract-wikipedia)
  - [Extract Wikinews](#extract-wikinews)
- [Prepare event dictionary](#prepare-event-dictionary)
  - [Collect candidate Wikidata events](#collect-candidate-wikidata-events)
  - [Collect event properties](#collect-event-properties)
  - [Prune Wikidata events](#prune-wikidata-events)
- [Mention linking](#mention-linking)
  - [Collect hyperlinks from Wikipedia](#collect-hyperlinks-from-wikipedia)
  - [Postprocess contexts](#postprocess-contexts)
  - [Collect label descriptions](#collect-label-descriptions)
- [Wikipedia -> Wikidata dataset](#wikipedia---wikidata-dataset)
- [Wikinews -> Wikidata dataset](#wikinews---wikidata-dataset)

## Install requirements

```bash
# tested on python 3.9
python -m pip install -r requirements.txt
# set data path
DATA=~/data
```

## Download Wikimedia dumps

Download Wikidata, Wikipedia and Wikinews dumps. **Note**: these are large downloads!

### Extract Wikidata

Download Wikidata JSON dump (approx 66G).

```bash
mkdir -p $DATA/wikidata_dumps
wget -nv -P $DATA/wikidata_dumps https://dumps.wikimedia.org/wikidatawiki/entities/20211004/wikidata-20211004-all.json.bz2
```

### Extract Wikipedia

1. Download XML dumps for various language Wikipedia (upto 18G).

```bash
mkdir -p $DATA/wikipedia_dumps/
# downloads Wikipedia dumps (.xml.bz2) to dumps/
# specify language code
while read -r lg _ || [[ -n $lg ]]; do
    bash wikipedia/download_xml.sh $lg $DATA/wikipedia_dumps/
done < langs.tsv
```

2. Extract document text (w/ hyperlinks) from the language dumps using wikiextractor.

```bash
while read -r lg _ || [[ -n $lg ]]; do
    bash wikipedia/extract_wiki_text.sh $lg $DATA/wikipedia_dumps/
done < langs.tsv
```

### Extract Wikinews

1. Download XML dumps (up to 2.3G).

```bash
while read -r lg _ || [[ -n $lg ]]; do
    bash wikinews/download_dump.sh $lg $DATA/wikinews_dumps;
done < langs.tsv
```

2. Extract document text (w/ hyperlinks) from the language dumps using wikiextractor.

```bash
for lg in `ls $DATA/wikinews_dumps`; do
    bash wikinews/extract_wn_text.sh $lg $DATA/wikinews_dumps $DATA/wikinews_extractor_out;
done;
```

3. Extract meta information (publication dates) for each language.

```bash
for lg in `ls $DATA/wikinews_dumps/`; do 
    bash wikinews/get_meta_info.sh $lg $DATA/wikinews_dumps/ $DATA/wikinews_meta/;
done;
```

## Prepare event dictionary

For reproducibility reasons, we use a fixed Wikidata dump (e.g., 20211004 version) instead of API queries.

### Collect candidate Wikidata events

For each item in Wikidata, use spatial and temporal properties to identify events. For each event, collect the titles of corresponding language Wikipedia pages. **Note**: given the large size of Wikidata dump, this script takes a really long time (~95 hrs on a single CPU!).

```bash
python wikidata/get_wd2wikipedia_links.py \
    $DATA/wikidata_dumps/wikidata-20211004-all.json.bz2 \
    $DATA/wd2wikipedia_links.tsv
```

### Collect event properties

Collect properties for all the candidate events. These properties are later utilized for pruning non-event items and hierarchy discovery among events.

```bash
awk -F'\t' '{if (NR > 1) print $3}' $DATA/wd2wikipedia_links.tsv | sort -u > $DATA/candidate_wd_items.txt
python wikidata/get_wd_props.py \
    $DATA/wikidata_dumps/wikidata-20211004-all.json.bz2 \
    $DATA/candidate_wd_items.txt \
    $DATA/wd_item_props.jsonl
```

Get the labels and descriptions for candidate events in all the available languages on Wikidata.

```bash
python wikidata/get_wd_descriptions.py \
    $DATA/wikidata_dumps/wikidata-20211004-all.json.bz2 \
    $DATA/candidate_wd_items.txt \
    $DATA/wd_item_descriptions.jsonl
```

Get the article title and descriptions for the Wikipedia pages that correspond to the events from Wikidata. Collect information for all the languages from Wikidata.

```bash
python wikipedia/get_wiki_descriptions.py \
    $DATA/wikipedia_dumps/wikiextractor_out \
    $DATA/wd2wikipedia_links.tsv \
    $DATA/wiki_event_descriptions.tsv
```

### Prune Wikidata events

Prune candidate events based on the presence of certain properties. See [resources/exclusion_wd_props.tsv] for the full list of exclusion properties.

```bash
python postprocess/prune_wd_items.py \
    $DATA/wd2wikipedia_links.tsv \
    $DATA/wd_item_props.jsonl \
    $DATA/wd_item_descriptions.jsonl \
    $DATA/wiki_event_descriptions.tsv \
    resources/exclusion_wd_props.tsv \
    $DATA/wd2wikipedia_links.postproc.tsv \
    --skipped-out $DATA/wd2wikipedia_links.skipped.tsv
```

## Mention linking

### Collect hyperlinks from Wikipedia

Iterate through the multilingual Wikipedia dumps and find mention spans that link to events from Wikidata.

```bash
python wikipedia/collect_wikipedia_inlinks.py \
    $DATA/wikipedia_dumps/wikiextractor_out/ \
    $DATA/wd2wikipedia_links.postproc.tsv \
    $DATA/xlec-data.tsv
```

### Postprocess contexts

Postprocess at context level,

```bash
python postprocess/postprocess_inlink_contexts.py \
    $DATA/xlec-data.tsv \
    $DATA/wiki_event_descriptions.tsv \
    $DATA/xlec-data.context-postproc.tsv \
    --skipped-out $DATA/xlec-data.context-skipped.tsv
```

Postprocess languages,

```bash
python postprocess/postprocess_inlink_langs.py \
    $DATA/xlec-data.context-postproc.tsv \
    resources/xlm-roberta-langs.json \
    $DATA/xlec-data.lang-postproc.tsv \
    --skipped-out $DATA/xlec-data.lang-skipped.tsv
```

Limit the frequency of mention spans.

```bash
python postprocess/sample_inlink_contexts.py \
    $DATA/xlec-data.lang-postproc.tsv \
    $DATA/xlec-data.sampled.tsv \
    --sample-size 20
```

Postprocess at item level,

```bash
python postprocess/postprocess_inlink_items.py \
    $DATA/xlec-data.sampled.tsv \
    $DATA/wiki_event_descriptions.tsv \
    $DATA/xlec-data.item-postproc.tsv \
    --skipped-out $DATA/xlec-data.item-skipped.tsv
```

### Collect label descriptions

Write the label descriptions (from Wikipedia) for the (item, language) pairs from final dataset.

```bash
awk -F'\t' 'FNR==NR {a[$1$2]++; b[$1]++; next} {if (($1$2 in a) || ($1 in b && $2 == "en")) {print $0}}' $DATA/xlec-data.item-postproc.tsv $DATA/wiki_event_descriptions.tsv > $DATA/label_desc.tsv
```

## Wikipedia -> Wikidata dataset

Generate train, dev and test splits for zero-shot event linking task.

```bash
python prepare_splits.py \
    $DATA/xlec-data.item-postproc.tsv \
    $DATA/label_desc.tsv \
    $DATA/wd_item_props.jsonl \
    $DATA/zero_shot_splits/
```

Convert the data into BLINK format for both crosslingual and multilingual evaluations.

```bash
# cross-lingual
python wikipedia/convert_to_blink.py \
    $DATA/zero_shot_splits/disjoint_sequences/ \
    $DATA/label_desc.tsv \
    $DATA/zero_shot_crosslingual/disjoint_sequences/ \
    --en-label
# multi-lingual
python wikipedia/convert_to_blink.py \
    $DATA/zero_shot_splits/disjoint_sequences/ \
    $DATA/label_desc.tsv \
    $DATA/zero_shot_multilingual/disjoint_sequences/
```

## Wikinews -> Wikidata dataset

```bash
python wikinews/collect_wikinews_inlinks.py \
    $DATA/wikinews_extractor_out/docs \
    $DATA/label_desc.tsv \
    $DATA/wd2wikinews_links.tsv \
    $DATA/wikinews_meta \
    $DATA/wn_xlel_data.tsv
```

Convert to blink format for cross-domain evaluation.

```bash
# crosslingual
python wikinews/convert_to_blink.py \
    $DATA/wn_xlel_data.tsv \
    $DATA/label_desc.tsv \
    $DATA/wn_eval/crosslingual \
    --en-label
# multilingual
python wikinews/convert_to_blink.py \
    $DATA/wn_xlel_data.tsv \
    $DATA/label_desc.tsv \
    $DATA/wn_eval/multilingual
```

Convert to blink format for zero-shot evaluation.

```bash
# crosslingual
python wikinews/convert_to_blink_zeshel.py \
    $DATA/wn_xlel_data.tsv \
    $DATA/label_desc.tsv \
    $DATA/zero_shot_splits/disjoint_sequences/ \
    dev,test \
    $DATA/wn_zeshel_eval/crosslingual \
    --en-label
# multilingual
python wikinews/convert_to_blink_zeshel.py \
    $DATA/wn_xlel_data.tsv \
    $DATA/label_desc.tsv \
    $DATA/zero_shot_splits/disjoint_sequences/ \
    dev,test \
    $DATA/wn_zeshel_eval/multilingual
```

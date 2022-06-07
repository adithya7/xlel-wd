import argparse
import csv
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import pandas as pd
from tqdm import tqdm
from utils import load_wiki_event_descriptions, load_xlel_tsv


def collect_monolingual_items(wiki_inlinks: pd.DataFrame) -> Set:

    logging.info(f"skipping monolingual items")

    item2langs = defaultdict(set)
    for _, row in tqdm(wiki_inlinks.iterrows()):
        item2langs[row["Wikidata Item"]].add(row["Wikipedia Language"])

    monolingual_items = set()
    for item in item2langs:
        if len(item2langs[item]) == 1:
            monolingual_items.add(item)

    logging.info(f"skipped {len(monolingual_items)} items")

    return monolingual_items


def collect_low_frequency_items(wiki_inlinks: pd.DataFrame) -> Set:

    FREQ_THRESHOLD = 30

    logging.info(f"skipping low frequency items")

    item2count = defaultdict(int)
    for _, row in tqdm(wiki_inlinks.iterrows()):
        item2count[row["Wikidata Item"]] += 1

    low_threshold_items = set()
    for item in item2count:
        if item2count[item] < FREQ_THRESHOLD:
            low_threshold_items.add(item)

    logging.info(f"skipped {len(low_threshold_items)} items")

    return low_threshold_items


def collect_low_diversity_items(wiki_inlinks: pd.DataFrame) -> Set:

    THRESHOLD = 0.5

    logging.info(
        f"skipping items with high overlap of mention and title, threshold: {THRESHOLD}"
    )

    event2total = defaultdict(int)
    event2match = defaultdict(int)

    for _, row in tqdm(wiki_inlinks.iterrows()):
        event = row["Wikidata Item"]
        title = row["Wikipedia Title"]
        context = row["Context"]
        _, mention, _ = re.match(r"(.*)<a> (.*) </a>(.*)", context).groups()
        if mention == title:
            event2match[event] += 1
        event2total[event] += 1

    low_diversity_items = set()
    for event in event2total:
        if (event2match[event] / event2total[event]) >= THRESHOLD:
            low_diversity_items.add(event)

    logging.info(f"skipped {len(low_diversity_items)} items")

    return low_diversity_items


def write_skipped_links(skipped_items: Dict, item2desc: Dict, file_path: Path):

    skipped_item_set = set()

    out = {"Wikidata Item": [], "Comment": [], "Label": [], "Description": []}

    for skip_label in skipped_items:
        for wd_item in skipped_items[skip_label]:
            out["Wikidata Item"] += [wd_item]
            out["Comment"] += [skip_label]
            out["Label"] = item2desc[wd_item]["en"]["label"]
            out["Description"] = item2desc[wd_item]["en"]["description"]
            skipped_item_set.add(wd_item)

    df = pd.DataFrame.from_dict(out)
    df.to_csv(file_path, sep="\t", index=False, quoting=csv.QUOTE_NONE)

    logging.info(f"total items skipped: {len(skipped_item_set)}")

    return skipped_item_set


def write_out(wiki_inlinks: pd.DataFrame, skipped_qids: Set, file_path: Path):

    out_data = []

    for _, row in wiki_inlinks.iterrows():
        if row["Wikidata Item"] not in skipped_qids:
            out_data += [row]

    pd.DataFrame(out_data).to_csv(
        file_path, sep="\t", index=False, quoting=csv.QUOTE_NONE
    )


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data", type=Path, help="Path to XLEL tsv")
    parser.add_argument(
        "wiki_event_desc", type=Path, help="path to Wikipedia event descriptions"
    )
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--skipped-out", type=Path, default=None, help="path to write skipped links"
    )

    args = parser.parse_args()

    item2desc = load_wiki_event_descriptions(args.wiki_event_desc)

    wiki_inlinks = load_xlel_tsv(
        args.data,
        return_keys=[
            "Wikidata Item",
            "Wikipedia Language",
            "Wikipedia Title",
            "Wikipedia Inlink Title",
            "Context",
        ],
    )

    skipped_items = {}

    out_items = collect_monolingual_items(wiki_inlinks)
    skipped_items["MONOLINGUAL"] = out_items

    out_items = collect_low_frequency_items(wiki_inlinks)
    skipped_items["LOW_FREQUENCY"] = out_items

    out_items = collect_low_diversity_items(wiki_inlinks)
    skipped_items["LOW_DIVERSITY"] = out_items

    skipped_item_set = write_skipped_links(skipped_items, item2desc, args.skipped_out)
    write_out(wiki_inlinks, skipped_item_set, args.out)

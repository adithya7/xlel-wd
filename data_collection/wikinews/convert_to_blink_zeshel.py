import argparse
import csv
import json
import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd
from tqdm import tqdm


def load_events(file_path: Path) -> Dict:
    event2desc = {}

    logging.info(f"loading events from {file_path}")
    df = pd.read_csv(file_path, sep="\t", quoting=csv.QUOTE_NONE)
    for _, row in tqdm(df.iterrows()):
        if row["Wikidata Item"] not in event2desc:
            event2desc[row["Wikidata Item"]] = {}
        event2desc[row["Wikidata Item"]][row["Wikipedia Language"]] = {
            "title": row["Wikipedia Title"],
            "description": row["Wikipedia Description"],
        }
    logging.info(f"found descriptions for {len(event2desc)} events")
    return event2desc


def load_unseen_events(dir_path: Path, splits: List[str]) -> Set:
    wd_items = set()
    file_paths = [f"{dir_path}/{data_split}.tsv" for data_split in splits]
    for file_path in file_paths:
        with open(Path(file_path), "r") as rf:
            header = rf.readline()
            for line in rf:
                splits = line.split("\t")
                wd_items.add(splits[0])
    return wd_items


def write_blink_jsonl(
    tsv_path: Path, out_path: Path, event2desc: Dict, unseen_events: Set, use_en_label: bool = False
):

    data_instances = []
    logging.info(f"loading data from {tsv_path}")
    count = 0
    with open(tsv_path, "r") as rf:
        header = rf.readline()
        for line in tqdm(rf):
            columns = line.strip().split("\t")
            (
                wikidata_item,
                lg,
                title,
                inlink_page_id,
                inlink_title,
                inlink_date,
                context,
                long_context,
                event_source,
            ) = columns

            context_left, mention, context_right = re.match(r"(.*)<a> (.*) </a>(.*)", context).groups()

            label_lang = "en" if use_en_label else lg
            try:
                label_title = event2desc[wikidata_item][label_lang]["title"]
                label_desc = event2desc[wikidata_item][label_lang]["description"]
            except:
                count += 1
                continue

            if wikidata_item not in unseen_events:
                continue

            inlink_date = inlink_date.replace("_", " ")
            data_instances += [
                {
                    "context_title": inlink_title,
                    "context_date": inlink_date,
                    "context_left": context_left,
                    "context_right": context_right,
                    "mention": mention,
                    "context_lang": lg,
                    "label_description": label_desc,
                    "label_id": wikidata_item.strip("Q"),
                    "label_title": label_title,
                }
            ]

    # shuffle instances to facilitate use of in-batch negatives in BLINK bi-encoder training
    random.shuffle(data_instances)

    logging.info(f"writing data in blink format to {out_path}")
    with open(out_path, "w") as wf:
        for _instance in data_instances:
            wf.write(json.dumps(_instance, ensure_ascii=False) + "\n")


def write_label_dict_jsonl(event2desc: Dict, out_path: Path, use_en_label: bool = False):

    data_instances = []

    for _item in event2desc:
        for lg in event2desc[_item]:
            if use_en_label and lg != "en":
                continue
            data_instances += [
                {
                    "label_id": _item.strip("Q"),
                    "label_title": event2desc[_item][lg]["title"],
                    "label_desc": event2desc[_item][lg]["description"],
                    "label_lang": lg,
                }
            ]

    logging.info(f"writing event dictionary to {out_path}")
    with open(out_path, "w") as wf:
        for _instance in data_instances:
            wf.write(json.dumps(_instance, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="convert WN data into jsonl format for evaluation")
    parser.add_argument("tsv", type=Path, help="path to tsv file")
    parser.add_argument(
        "label_desc", type=Path, help="path to label_desc tsv, reuse the one from disjoint sequences"
    )
    parser.add_argument(
        "sequences",
        type=Path,
        help="path to sequences from Wikipedia corpus, zero_shot_splits/disjoint_sequences/",
    )
    parser.add_argument("splits", type=str, help="dev,test")
    parser.add_argument("out_dir", type=Path, help="output directory to write evaluation set and label desc")
    parser.add_argument("--en-label", action="store_true", help="use English description for labels")

    args = parser.parse_args()

    event2desc = load_events(args.label_desc)
    unseen_events = load_unseen_events(args.sequences, args.splits.split(","))

    args.out_dir.mkdir(exist_ok=True, parents=True)
    write_blink_jsonl(
        args.tsv, args.out_dir / "test.jsonl", event2desc, unseen_events, use_en_label=args.en_label
    )
    write_label_dict_jsonl(event2desc, args.out_dir / "label_dict.jsonl", use_en_label=args.en_label)

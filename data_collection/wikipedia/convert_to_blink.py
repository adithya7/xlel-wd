import argparse
import csv
import json
import logging
import pandas as pd
from pathlib import Path
import re
import random
from tqdm import tqdm
from typing import Dict


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


def write_blink_jsonl(tsv_path: Path, out_path: Path, event2desc: Dict, use_en_label: bool = False):

    data_instances = []
    logging.info(f"loading data from {tsv_path}")
    with open(tsv_path, "r") as rf:
        header = rf.readline()
        for line in tqdm(rf):
            columns = line.strip().split("\t")
            wikidata_item, wikipedia_lang, wikipedia_title, wikipedia_inlink_title, context = columns

            context_left, mention, context_right = re.match(r"(.*)<a> (.*) </a>(.*)", context).groups()

            label_lang = "en" if use_en_label else wikipedia_lang
            label_title = event2desc[wikidata_item][label_lang]["title"]
            label_desc = event2desc[wikidata_item][label_lang]["description"]

            data_instances += [
                {
                    "context_left": context_left,
                    "context_right": context_right,
                    "mention": mention,
                    "context_lang": wikipedia_lang,
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

    # shuffle instances to facilitate use of in-batch negatives in BLINK bi-encoder training
    random.shuffle(data_instances)

    logging.info(f"writing event dictionary to {out_path}")
    with open(out_path, "w") as wf:
        for _instance in data_instances:
            wf.write(json.dumps(_instance, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="convert dataset into BLINK format")
    parser.add_argument("tsv_dir", type=Path, help="dataset tsv")
    parser.add_argument("label_tsv", type=Path, help="label description tsv")
    parser.add_argument("out_dir", type=Path, help="output directory")
    parser.add_argument("--en-label", action="store_true", help="use English description for labels")

    args = parser.parse_args()

    args.out_dir.mkdir(exist_ok=True, parents=True)

    event2desc = load_events(args.label_tsv)

    write_blink_jsonl(
        args.tsv_dir / "train.tsv", args.out_dir / "train.jsonl", event2desc, use_en_label=args.en_label
    )
    write_blink_jsonl(
        args.tsv_dir / "dev.tsv", args.out_dir / "dev.jsonl", event2desc, use_en_label=args.en_label
    )
    write_blink_jsonl(
        args.tsv_dir / "test.tsv", args.out_dir / "test.jsonl", event2desc, use_en_label=args.en_label
    )

    write_label_dict_jsonl(event2desc, args.out_dir / "label_dict.jsonl", use_en_label=args.en_label)

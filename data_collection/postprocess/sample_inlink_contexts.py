import argparse
from collections import defaultdict
import csv
import pandas as pd
from pathlib import Path
import random
import re
from tqdm import tqdm
from typing import Dict

from utils import load_xlel_tsv


def get_event2mentions(wiki_inlinks: pd.DataFrame) -> Dict:
    """
    for each event, in each language, groups mentions by the mention_txt
    """
    event2mentions = {}
    for _, row in tqdm(wiki_inlinks.iterrows()):
        event = row["Wikidata Item"]
        lg = row["Wikipedia Language"]
        _, mention, _ = re.match(r"(.*)<a> (.*) </a>(.*)", row["Context"]).groups()

        if event not in event2mentions:
            event2mentions[event] = {}
        if lg not in event2mentions[event]:
            event2mentions[event][lg] = defaultdict(list)

        event2mentions[event][lg][mention] += [row]

    return event2mentions


def sample_mentions(event2mentions: Dict, MENTION_LIMIT: int, seed: int):

    """
    for each tuple (event, lg, mention_txt), sample <=MENTION_LIMIT contexts.
    """

    out_event2mentions = {}

    random.seed(seed)
    for event in tqdm(event2mentions):
        out_event2mentions[event] = {}
        for lg in event2mentions[event]:
            out_event2mentions[event][lg] = {}
            for mention_txt in event2mentions[event][lg]:
                data_instances = event2mentions[event][lg][mention_txt]
                num_instances = len(data_instances)
                sampled_instances = [
                    data_instances[_idx]
                    for _idx in random.sample(range(num_instances), k=min(num_instances, MENTION_LIMIT))
                ]
                out_event2mentions[event][lg][mention_txt] = sampled_instances

    return out_event2mentions


def write_jsonl(event2mentions: Dict, file_path: Path):
    out_data = []
    for event in event2mentions:
        for lg in event2mentions[event]:
            for _, data_instances in event2mentions[event][lg].items():
                out_data += data_instances
    pd.DataFrame(out_data).to_csv(file_path, sep="\t", index=False, quoting=csv.QUOTE_NONE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data", type=Path, help="path to XLEL data")
    parser.add_argument("out", type=Path, help="")
    parser.add_argument(
        "--sample-size", type=int, default=20, help="maximum number of mentions reps allowed per event"
    )
    parser.add_argument("--seed", type=int, default=41)

    args = parser.parse_args()

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

    event2mentions = get_event2mentions(wiki_inlinks)
    sampled_event2mentions = sample_mentions(event2mentions, MENTION_LIMIT=args.sample_size, seed=args.seed)
    write_jsonl(sampled_event2mentions, args.out)

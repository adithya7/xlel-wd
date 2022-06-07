import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Dict, List

import networkx as nx
import pandas as pd
from tqdm import tqdm
from utils import load_wd_properties


def load_events(file_path: Path) -> List:
    logging.info(f"loading events from {file_path}")
    df = pd.read_csv(
        file_path,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        usecols=["Wikidata Item", "Wikipedia Language", "Wikipedia Title"],
    )
    event2label = {}
    for _, row in tqdm(df.iterrows()):
        if row["Wikipedia Language"] == "en":
            event2label[row["Wikidata Item"]] = row["Wikipedia Title"]

    logging.info(f"found English titles for {len(event2label)} items")

    return event2label


def collect_event_hierarchies(item2props: Dict, event2label: Dict) -> Dict:
    """
    `event2label`
        - list of events, with a mapping to their English Wikipedia titles
    `item2props`
        - for each Wikidata item (qid), a mapping from pid to values
    """

    FOLLOWS = "P155"
    FOLLOWED_BY = "P156"
    HAS_PART = "P527"
    PART_OF = "P361"

    G = nx.Graph()
    # add events as nodes
    for event in event2label:
        G.add_node(event)
    # add edges between nodes if they are related by any of the above four properties
    for _item in tqdm(item2props):
        _item_props = item2props[_item]
        for prop in [FOLLOWS, FOLLOWED_BY, HAS_PART, PART_OF]:
            for _related_item in _item_props.get(prop, []):
                value = _related_item["value"]["id"]["value"]
                G.add_edge(_item, value)

    group2event = {}
    for idx, component in enumerate(nx.connected_components(G)):
        group2event[idx] = list(component)
    event2group = {
        event: idx for idx, events in group2event.items() for event in events
    }

    return event2group, group2event


def split_zero_shot_dissimilar(group2event: Dict) -> Dict:

    group_indices = [x for x in group2event]
    random.shuffle(group_indices)

    logging.info(f"preparing zero-shot splits with dissimilar events")

    TRAIN = int(0.8 * len(group_indices))
    DEV = int(0.1 * len(group_indices))
    TEST = len(group_indices) - TRAIN - DEV

    train_indices = group_indices[:TRAIN]
    dev_indices = group_indices[TRAIN : TRAIN + DEV]
    test_indices = group_indices[TRAIN + DEV : TRAIN + DEV + TEST]

    logging.info(
        f"train sequences: {len(train_indices)}, dev sequences: {len(dev_indices)}, test sequences: {len(test_indices)}"
    )

    train_events, dev_events, test_events = [], [], []
    for idx in train_indices:
        train_events += group2event[idx]
    for idx in dev_indices:
        dev_events += group2event[idx]
    for idx in test_indices:
        test_events += group2event[idx]

    logging.info(
        f"train: {len(train_events)}, dev: {len(dev_events)}, test: {len(test_events)}"
    )

    event2split = {}

    for _event in train_events:
        event2split[_event] = "train"
    for _event in dev_events:
        event2split[_event] = "dev"
    for _event in test_events:
        event2split[_event] = "test"

    return event2split


def write_splits(data_path: Path, out_path: Path, splits: Dict):

    out_path.mkdir(exist_ok=True)

    train_path = out_path / "train.tsv"
    dev_path = out_path / "dev.tsv"
    test_path = out_path / "test.tsv"

    wf_data_train = open(train_path, "w")
    wf_data_dev = open(dev_path, "w")
    wf_data_test = open(test_path, "w")

    logging.info(f"loading data from {data_path}")

    logging.info(f"writing train split to {train_path}")
    logging.info(f"writing dev split to {dev_path}")
    logging.info(f"writing test split to {test_path}")

    with open(data_path, "r") as rf:
        header = rf.readline()
        wf_data_train.write(header)
        wf_data_dev.write(header)
        wf_data_test.write(header)
        for line in tqdm(rf):
            columns = line.split("\t")
            event_id = columns[0]  # wikidata item id
            event_split = splits[event_id]
            if event_split == "train":
                wf_data_train.write(line)
            if event_split == "dev":
                wf_data_dev.write(line)
            if event_split == "test":
                wf_data_test.write(line)

    wf_data_train.close()
    wf_data_dev.close()
    wf_data_test.close()


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(
        description="prepare train, dev and test splits for the event linking dataset"
    )
    parser.add_argument("data_tsv", type=Path, help="dataset tsv")
    parser.add_argument("label_tsv", type=Path, help="label description tsv")
    parser.add_argument(
        "wd_item_props", type=Path, help="path to Wikidata item properties"
    )
    parser.add_argument("out", type=Path, help="output directory")

    args = parser.parse_args()

    args.out.mkdir(exist_ok=True)

    random.seed(41)

    event2label = load_events(args.label_tsv)
    item2props = load_wd_properties(args.wd_item_props)

    event2group, group2event = collect_event_hierarchies(item2props, event2label)
    event2split = split_zero_shot_dissimilar(group2event)
    write_splits(args.data_tsv, args.out / "disjoint_spatiotemporal", event2split)

import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set

import pandas as pd
from tqdm import tqdm
from utils import (
    load_exclusion_wd_props,
    load_wd2wikipedia_links,
    load_wd_descriptions,
    load_wd_properties,
    load_wiki_event_descriptions,
)


def skip_non_event_props(wiki_links, item2wd_props: Dict, non_event_props):
    """
    Skip wikidata items with specific non-event properties
    """

    logging.info(f"skipping items based on properties")
    skipped_wiki_links = []
    for _, row in tqdm(wiki_links.iterrows()):
        skipItem = False
        for prop in item2wd_props[row["QID"]]:
            if prop in non_event_props:
                disallowed_parent_qids = non_event_props[prop]
                if "*" in disallowed_parent_qids:
                    skipItem = True
                else:
                    assert prop == "P31"
                    parent_qids = [
                        x["value"]["id"]["value"]
                        for x in item2wd_props[row["QID"]][prop]
                    ]
                    if len(set(parent_qids) & set(disallowed_parent_qids)) > 0:
                        skipItem = True
        if skipItem:
            skipped_wiki_links += [row]

    logging.info(f"skipping {len(skipped_wiki_links)} wd2wikipedia links")

    return pd.DataFrame(skipped_wiki_links)


def skip_empty_en_labels(wiki_links, item2wd_desc: Dict):
    """
    Skip items if there is no English label
    """

    logging.info("skipping items based on English labels")

    skipped_wiki_links = []
    for _, row in tqdm(wiki_links.iterrows()):
        en_label = ""
        if row["QID"] in item2wd_desc:
            if "en" in item2wd_desc[row["QID"]]:
                en_label = item2wd_desc[row["QID"]]["en"]["label"]
        if en_label == "":
            skipped_wiki_links += [row]

    logging.info(f"skipping {len(skipped_wiki_links)} wd2wikipedia links")

    return pd.DataFrame(skipped_wiki_links)


def skip_monolingual_items(wiki_links):
    """
    Skip items that appear only in one language
    """

    logging.info("skipping monolingual items")

    item2lang = defaultdict(int)
    for _, row in tqdm(wiki_links.iterrows()):
        item2lang[row["QID"]] += 1

    skipped_wiki_links = []
    for _, row in tqdm(wiki_links.iterrows()):
        if item2lang[row["QID"]] == 1:
            skipped_wiki_links += [row]

    logging.info(f"skipping {len(skipped_wiki_links)} wd2wikipedia links")

    return pd.DataFrame(skipped_wiki_links)


def skip_wd_classes(wiki_links, item2wd_props: Dict):
    """
    Skip item X if there exists ? INSTANCE_OF X
    """
    INSTANCE_OF = "P31"

    logging.info(f"skipping items that has Wikidata instances")

    exclusion_set = set()
    for _, row in tqdm(wiki_links.iterrows()):
        wd_item_props = item2wd_props[row["QID"]]
        if INSTANCE_OF in wd_item_props:
            parent_items = [
                x["value"]["id"]["value"] for x in wd_item_props[INSTANCE_OF]
            ]
            exclusion_set.update(parent_items)

    skipped_wiki_links = []
    for _, row in tqdm(wiki_links.iterrows()):
        if row["QID"] in exclusion_set:
            skipped_wiki_links += [row]

    logging.info(f"skipping {len(skipped_wiki_links)} wd2wikipedia links")

    return pd.DataFrame(skipped_wiki_links)


def skip_non_unique_labels(wiki_links, item2wd_desc: Dict):
    """
    Skip items that share English Wikidata labels.
    """

    logging.info("skipping items with non-unique English labels")

    label2count = defaultdict(int)
    finished_items = set()
    for _, row in tqdm(wiki_links.iterrows()):
        wd_item = row["QID"]
        if wd_item not in item2wd_desc:
            continue
        if wd_item not in finished_items:
            # for each item, increment counter only once
            if "en" in item2wd_desc[wd_item]:
                en_wd_label = item2wd_desc[wd_item]["en"]["label"]
                label2count[en_wd_label] += 1
                finished_items.add(wd_item)

    skipped_wiki_links = []
    for _, row in tqdm(wiki_links.iterrows()):
        wd_item = row["QID"]
        if wd_item not in item2wd_desc:
            continue
        if "en" in item2wd_desc[wd_item]:
            en_wd_label = item2wd_desc[wd_item]["en"]["label"]
            if label2count[en_wd_label] > 1:
                skipped_wiki_links += [row]

    logging.info(f"skipping {len(skipped_wiki_links)} wd2wikipedia links")

    return pd.DataFrame(skipped_wiki_links)


def skip_non_english_wiki(wiki_links, item2wiki_desc: Dict):
    """
    Skip items if there is no associated English Wikipedia Article or no English label/description
    """

    logging.info(
        "skipping items with no associated en-wiki article or missing label/description"
    )

    skipped_wiki_links = []
    for _, row in tqdm(wiki_links.iterrows()):
        if row["QID"] in item2wiki_desc:
            if "en" not in item2wiki_desc[row["QID"]]:
                skipped_wiki_links += [row]
            else:
                if (
                    item2wiki_desc[row["QID"]]["en"]["label"] == ""
                    or item2wiki_desc[row["QID"]]["en"]["description"] == ""
                ):
                    skipped_wiki_links += [row]
        else:
            skipped_wiki_links += [row]

    logging.info(f"skipping {len(skipped_wiki_links)} wd2wikipedia links")

    return pd.DataFrame(skipped_wiki_links)


def write_skipped_items(skipped_data: Dict, file_path: Path):

    out_data = []
    skipped_item_set = set()

    comments = []
    for skip_label, df in skipped_data.items():
        for _, row in df.iterrows():
            out_data += [row]
            comments += [skip_label]
            skipped_item_set.add(row["QID"])

    df = pd.DataFrame(out_data)
    df["Comment"] = comments
    df.to_csv(file_path, sep="\t", index=False, quoting=csv.QUOTE_NONE)

    logging.info(f"total items skipped: {len(skipped_item_set)}")

    return skipped_item_set


def write_out(wiki_links: pd.DataFrame, skipped_qids: Set, file_path: Path):

    out_data = []

    for _, row in wiki_links.iterrows():
        if row["QID"] not in skipped_qids:
            out_data += [row]

    pd.DataFrame(out_data).to_csv(
        file_path, sep="\t", index=False, quoting=csv.QUOTE_NONE
    )


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("log_prune_wd_items.txt"),
        ],
    )

    parser = argparse.ArgumentParser(description="prune WD items")
    parser.add_argument(
        "wd2wikipedia_links", type=Path, help="Wikidata to Wikipedia links"
    )
    parser.add_argument("wd_item_prop", type=Path, help="Wikidata properties")
    parser.add_argument(
        "wd_item_desc", type=Path, help="Wikidata labels and descriptions"
    )
    parser.add_argument(
        "wiki_event_desc", type=Path, help="Wikipedia event descriptions"
    )
    parser.add_argument("skip_props", type=Path, help="Wikidata props to exclude")
    parser.add_argument(
        "out", type=Path, help="path to write pruned Wikidata to Wikipedia links"
    )
    parser.add_argument(
        "--skipped-out", type=Path, default=None, help="path to write skipped links"
    )

    args = parser.parse_args()

    item2wd_props = load_wd_properties(args.wd_item_prop)
    item2wd_desc = load_wd_descriptions(args.wd_item_desc)
    item2wiki_desc = load_wiki_event_descriptions(args.wiki_event_desc)

    non_event_props = load_exclusion_wd_props(args.skip_props)

    wiki_links = load_wd2wikipedia_links(args.wd2wikipedia_links)

    skipped_links = {}

    out_items = skip_non_event_props(wiki_links, item2wd_props, non_event_props)
    skipped_links["NON_EVENT_PROP"] = out_items

    out_items = skip_empty_en_labels(wiki_links, item2wd_desc)
    skipped_links["NULL_ENGLISH_WD_LABEL"] = out_items

    out_items = skip_monolingual_items(wiki_links)
    skipped_links["MONOLINGUAL_ITEM"] = out_items

    out_items = skip_wd_classes(wiki_links, item2wd_props)
    skipped_links["WD_CLASSES"] = out_items

    out_items = skip_non_unique_labels(wiki_links, item2wd_desc)
    skipped_links["NON_UNIQUE_WD_LABEL"] = out_items

    out_items = skip_non_english_wiki(wiki_links, item2wiki_desc)
    skipped_links["MISSING_EN_WIKI"] = out_items

    skipped_qid_set = write_skipped_items(skipped_links, args.skipped_out)
    write_out(wiki_links, skipped_qid_set, args.out)

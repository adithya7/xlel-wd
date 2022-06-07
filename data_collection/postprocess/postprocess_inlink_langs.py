import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Set

import pandas as pd
from tqdm import tqdm
from utils import load_xlel_tsv


def load_langs(file_path: Path) -> Set:
    with open(file_path, "r") as rf:
        data = json.load(rf)
    langs = set(list(data.values()))
    return langs


def skip_langs(wiki_inlinks: pd.DataFrame, langs: Set):

    valid_rows, skipped_rows = [], []
    skip_labels = []

    for _, row in tqdm(wiki_inlinks.iterrows()):
        if row["Wikipedia Language"] in langs:
            valid_rows += [row]
        else:
            skipped_rows += [row]
            skip_labels += [f"UNSUPPORTED_LANG_{row['Wikipedia Language']}"]

    skipped_df = pd.DataFrame(skipped_rows)
    skipped_df["Comment"] = skip_labels

    valid_df = pd.DataFrame(valid_rows)

    return skipped_df, valid_df


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("data", type=Path, help="Path to XLEL tsv")
    parser.add_argument("langs", type=Path, help="list of valid languages")
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--skipped-out", type=Path, default=None, help="path to write skipped links"
    )

    args = parser.parse_args()

    langs = load_langs(args.langs)

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

    skipped_df, valid_df = skip_langs(wiki_inlinks, langs)
    skipped_df.to_csv(args.skipped_out, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    valid_df.to_csv(args.out, sep="\t", index=False, quoting=csv.QUOTE_NONE)

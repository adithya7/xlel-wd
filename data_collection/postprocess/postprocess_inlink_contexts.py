import argparse
import csv
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm
from utils import load_wiki_event_descriptions, load_xlel_tsv


def search_nearby_years(src_str, tgt_str):
    TIME_WINDOW = 10
    for _year in re.findall(r"[0-9]{4}", src_str):
        _year_int = int(_year)
        for i in range(_year_int - TIME_WINDOW, _year_int + TIME_WINDOW):
            if str(i) in tgt_str:
                return True
    return False


def collect_mentions(
    wiki_inlinks: pd.DataFrame, item2desc: Dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    CONTEXT_MIN_THRESHOLD = 100
    CONTEXT_MAX_THRESHOLD = 2000

    skipped_rows = []
    skip_labels = []
    wiki_labels = []
    wiki_desc = []

    valid_rows = []

    for _, row in tqdm(wiki_inlinks.iterrows()):

        toSkipRow = False

        # use Wikipedia English label and description to check for numeric values
        foundNum = False
        label_numeric_values = set(
            re.findall(r"[0-9]+", item2desc[row["Wikidata Item"]]["en"]["label"])
        )
        desc_numeric_values = set(
            re.findall(r"[0-9]+", item2desc[row["Wikidata Item"]]["en"]["description"])
        )
        context_numeric_values = set(re.findall(r"[0-9]+", row["Context"]))

        if len(label_numeric_values) == 0:
            foundNum = True
        elif len(label_numeric_values) > 0:
            if (len(label_numeric_values & context_numeric_values) > 0) or (
                len(desc_numeric_values & context_numeric_values) > 0
            ):
                foundNum = True

        foundNum |= search_nearby_years(
            f'{item2desc[row["Wikidata Item"]]["en"]["label"]} {item2desc[row["Wikidata Item"]]["en"]["description"]}',
            row["Context"],
        )

        # use language Wikipedia event article title to check for numeric values
        # this is important for languages that use special number format (e.g., Persian, Arabic)
        # ref: https://stackoverflow.com/questions/11879025/string-maketrans-for-english-and-persian-numbers
        # todo: how to tackle Hebrew

        src_numeral_chars = "۱۲۳۴۵۶۷۸۹۰١٢٣٤٥٦٧٨٩٠"
        tgt_numeral_chars = "12345678901234567890"
        src2tgt = str.maketrans(src_numeral_chars, tgt_numeral_chars)
        tgt_label = row["Wikipedia Title"].translate(src2tgt)
        tgt_context = row["Context"].translate(src2tgt)
        tgt_label_numeric_values = set(re.findall(r"[0-9]+", tgt_label))
        tgt_context_numeric_values = set(re.findall(r"[0-9]+", tgt_context))

        if len(tgt_label_numeric_values) == 0:
            foundNum = True
        elif len(tgt_label_numeric_values) > 0:
            if len(tgt_label_numeric_values & tgt_context_numeric_values) > 0:
                foundNum = True

        foundNum |= search_nearby_years(tgt_label, tgt_context)

        if not foundNum:
            # skip row due to missing numerals
            skipped_rows += [row]
            wiki_labels += [item2desc[row["Wikidata Item"]]["en"]["label"]]
            wiki_desc += [item2desc[row["Wikidata Item"]]["en"]["description"]]
            skip_labels += ["MISSING_NUMERAL"]
            toSkipRow = True

        # check if the context is too small or too large
        if len(row["Context"]) < CONTEXT_MIN_THRESHOLD:
            # skip row due to short character length
            skipped_rows += [row]
            wiki_labels += [item2desc[row["Wikidata Item"]]["en"]["label"]]
            wiki_desc += [item2desc[row["Wikidata Item"]]["en"]["description"]]
            skip_labels += ["LOW_CHAR_COUNT"]
            toSkipRow = True

        elif len(row["Context"]) > CONTEXT_MAX_THRESHOLD:
            # skip row due to long character length
            skipped_rows += [row]
            wiki_labels += [item2desc[row["Wikidata Item"]]["en"]["label"]]
            wiki_desc += [item2desc[row["Wikidata Item"]]["en"]["description"]]
            skip_labels += ["HIGH_CHAR_COUNT"]
            toSkipRow = True

        # check for lg-wiki description
        if row["Wikipedia Language"] not in item2desc[row["Wikidata Item"]]:
            skipped_rows += [row]
            wiki_labels += [""]
            wiki_desc += [""]
            skip_labels += ["MISSING LG_WIKI"]
            toSkipRow = True

        if not toSkipRow:
            valid_rows += [row]

    skip_label_counter = Counter(skip_labels)
    logging.info(", ".join([f"{k}:{v}" for k, v in skip_label_counter.items()]))

    skipped_df = pd.DataFrame(skipped_rows)
    skipped_df["Label"] = wiki_labels
    skipped_df["Description"] = wiki_desc
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
    parser.add_argument(
        "wiki_event_desc", type=Path, help="path to Wikipedia Event descriptions"
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

    skipped_df, valid_df = collect_mentions(wiki_inlinks, item2desc)
    skipped_df.to_csv(args.skipped_out, sep="\t", index=False, quoting=csv.QUOTE_NONE)
    valid_df.to_csv(args.out, sep="\t", index=False, quoting=csv.QUOTE_NONE)

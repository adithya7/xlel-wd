import argparse
import bz2
import json
from pathlib import Path
from typing import Set

import ijson
from tqdm import tqdm


def load_candidate_qids(file_path: Path) -> Set:
    candidate_qids = set()
    with open(file_path, "r") as rf:
        for line in rf:
            candidate_qids.add(line.strip())
    return candidate_qids


def collect_event_items(json_path: Path, candidate_qids: Set, out_path: Path):
    """
    iterate through Wikidata dump and collect label and description for candidate event items.
    params,
        - `json_path`: path to Wikidata dump
        - `candidate_qids`: Use the pre-identified candidate pool of event items to restrict the search
        - `out_path`: path to write event item descriptions
    """
    wiki_items = []

    qid = ""
    lang = ""
    lang2labels = {}
    lang2descriptions = {}
    labelEnd, descEnd = False, False
    isCandidateQID = False

    pbar = tqdm()
    for prefix, event, value in ijson.parse(bz2.open(json_path)):

        if (prefix, event) == ("item", "end_map"):
            # reset item
            qid = ""
            lang = ""
            lang2labels = {}
            lang2descriptions = {}
            isCandidateQID = False
            labelEnd, descEnd = False, False

            pbar.update(1)

        elif (prefix, event) == ("item.id", "string"):
            qid = value
            isCandidateQID = qid in candidate_qids

        elif isCandidateQID:
            if (
                prefix.startswith("item.labels.")
                and prefix.endswith(".language")
                and event == "string"
            ):
                lang = value
            elif prefix == f"item.labels.{lang}.value" and event == "string":
                lang2labels[lang] = value
            if (
                prefix.startswith("item.descriptions.")
                and prefix.endswith(".language")
                and event == "string"
            ):
                lang = value
            elif prefix == f"item.descriptions.{lang}.value" and event == "string":
                lang2descriptions[lang] = value
            elif (prefix, event) == ("item.labels", "end_map"):
                labelEnd = True
            elif (prefix, event) == ("item.descriptions", "end_map"):
                descEnd = True
            elif labelEnd and descEnd:
                for lang in lang2labels:
                    wiki_items += [
                        {
                            "qid": qid,
                            "lang": lang,
                            "label": lang2labels.get(lang, ""),
                            "description": lang2descriptions.get(lang, ""),
                        }
                    ]
                labelEnd, descEnd = False, False

        if len(wiki_items) >= 1000:
            with open(out_path, "a") as wf:
                for _item in wiki_items:
                    wf.write(json.dumps(_item, ensure_ascii=False))
                    wf.write("\n")
            wiki_items = []

    pbar.close()

    if len(wiki_items) > 0:
        with open(out_path, "a") as wf:
            for _item in wiki_items:
                wf.write(json.dumps(_item, ensure_ascii=False))
                wf.write("\n")
        wiki_items = []

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="collect event items from Wikidata dump"
    )
    parser.add_argument("json", type=Path, help="path to json dump")
    parser.add_argument("qids", type=Path, help="list of QIDs of interest")
    parser.add_argument("out", type=Path, help="path to output jsonl")

    args = parser.parse_args()

    args.out.unlink(missing_ok=True)

    candidate_qids = load_candidate_qids(args.qids)

    collect_event_items(args.json, candidate_qids, args.out)

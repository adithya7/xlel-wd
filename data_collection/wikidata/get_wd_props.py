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
    iterate through Wikidata dump and collect properties for candidate event items.
    params,
        - `json_path`: path to Wikidata dump
        - `candidate_qids`: Use the pre-identified candidate pool of event items to restrict the search
        - `out_path`: path to write event item properties
    """
    wiki_items = []

    qid = ""
    item_properties = {}
    property_id = ""
    map_key = ""
    isCandidateQID = False

    pbar = tqdm()
    for prefix, event, value in ijson.parse(bz2.open(json_path)):

        if (prefix, event) == ("item", "end_map"):
            # reset item
            qid = ""
            item_properties = {}
            property_id = ""
            map_key = ""
            isCandidateQID = False

            pbar.update(1)

        elif (prefix, event) == ("item.id", "string"):
            qid = value
            isCandidateQID = qid in candidate_qids

        elif isCandidateQID:
            if (prefix, event) == ("item.claims", "map_key"):
                property_id = value
                item_properties[property_id] = []

            elif (
                prefix == f"item.claims.{property_id}.item.mainsnak"
                and event == "map_key"
                and value == "datavalue"
            ):
                # each property can take multiple (data)values
                # add a new datavalue
                item_properties[property_id].append({})

            elif (
                prefix == f"item.claims.{property_id}.item.mainsnak.datavalue"
                and event == "map_key"
            ):
                # keep track of value and type for each datavalue
                item_properties[property_id][-1][value] = {}
                map_key = ""

            elif (
                prefix == f"item.claims.{property_id}.item.mainsnak.datavalue.value"
                and event == "map_key"
            ):
                map_key = value

            elif (
                prefix
                == f"item.claims.{property_id}.item.mainsnak.datavalue.value.{map_key}"
            ):
                item_properties[property_id][-1]["value"][map_key] = {
                    "type": str(event),
                    "value": str(value),
                }

            elif (
                prefix == f"item.claims.{property_id}.item.mainsnak.datavalue.value"
                and map_key == ""
                and event != "start_map"
            ):
                item_properties[property_id][-1]["value"] = {
                    "type": str(event),
                    "value": str(value),
                }

            elif (
                prefix == f"item.claims.{property_id}.item.mainsnak.datavalue.type"
                and event == "map_key"
            ):
                map_key = value

            elif (
                prefix
                == f"item.claims.{property_id}.item.mainsnak.datavalue.type.{map_key}"
            ):
                item_properties[property_id][-1]["type"][map_key] = {
                    "type": str(event),
                    "value": str(value),
                }

            elif (
                prefix == f"item.claims.{property_id}.item.mainsnak.datavalue.type"
                and map_key == ""
                and event != "start_map"
            ):
                item_properties[property_id][-1]["type"] = {
                    "type": str(event),
                    "value": str(value),
                }

            elif (prefix, event) == ("item.claims", "end_map"):
                for _property_id, _property_values in item_properties.items():
                    wiki_items += [
                        {"qid": qid, "pid": _property_id, "datavalue": _property_values}
                    ]

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
    parser.add_argument("out", type=Path, help="path to output tsv")

    args = parser.parse_args()

    args.out.unlink(missing_ok=True)

    candidate_qids = load_candidate_qids(args.qids)

    collect_event_items(args.json, candidate_qids, args.out)

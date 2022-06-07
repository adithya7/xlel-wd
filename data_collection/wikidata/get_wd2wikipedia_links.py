import argparse
import bz2
import re
from pathlib import Path

import ijson
from tqdm import tqdm


def collect_event_items(json_path: Path, out_path: Path):

    qid = ""
    item_props = []
    wikilinks = []
    isEvent = False

    pbar = tqdm()
    for prefix, event, value in ijson.parse(bz2.open(json_path)):

        if (prefix, event) == ("item", "end_map"):
            # reset item
            qid = ""
            item_props = []
            isEvent = False
            pbar.update(1)
        elif (prefix, event) == ("item.id", "string"):
            qid = value
        elif (prefix, event) == ("item.claims", "map_key"):
            item_props += [value]
        elif (prefix, event) == ("item.claims", "end_map"):
            hasTemporal, hasSpatial = False, False
            """
            (start_time, end_time)
            (point in time)
            (duration)
            """
            if (
                ("P580" in item_props and "P582" in item_props)
                or ("P585" in item_props)
                or ("P2047" in item_props)
            ):
                hasTemporal = True
            """
            (location)
            (coordinate location)
            """
            if "P276" in item_props or "P625" in item_props:
                hasSpatial = True
            if hasTemporal and hasSpatial:
                isEvent = True
        elif re.search(r"item\.sitelinks\.[a-z]+wiki\.title", prefix) is not None:
            lg = re.search(r"item\.sitelinks\.([a-z]+)wiki\.title", prefix).group(1)
            if isEvent:
                wikilinks += [[lg, value, qid]]

        if len(wikilinks) >= 10000:
            with open(out_path, "a") as wf:
                for lg, title, qid in wikilinks:
                    wf.write(f"{lg}\t{title}\t{qid}\n")
            wikilinks = []

    pbar.close()

    if len(wikilinks) > 0:
        with open(out_path, "a") as wf:
            for lg, title, qid in wikilinks:
                wf.write(f"{lg}\t{title}\t{qid}\n")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="collect event items from Wikidata dump"
    )
    parser.add_argument("json", type=Path, help="path to json dump")
    parser.add_argument("out", type=Path, help="path to output tsv")

    args = parser.parse_args()

    args.out.unlink(missing_ok=True)

    with open(args.out, "w") as wf:
        wf.write(f"Language\tWikipedia Title\tQID\n")

    collect_event_items(args.json, args.out)

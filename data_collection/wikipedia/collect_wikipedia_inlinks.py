import argparse
import csv
import json
import logging
import re
import urllib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm
from utils import get_title2wd, load_wd2wikipedia_links


def collect_inlink_titles(wiki2qid: Dict, wiki_inputs: Tuple[str, Path]):

    lang, wiki_file_path = wiki_inputs

    wiki_inlinks = defaultdict(list)
    skipped_page_count = 0

    with open(wiki_file_path, "r", errors="ignore", encoding="utf-8") as rf:
        for line in rf:
            try:
                page = json.loads(line.strip())
            except:
                skipped_page_count += 1
                continue

            title = page["title"]

            if (lang, title) in wiki2qid:
                # the current Wikipedia page is also a seed article
                # skip inlinks from seed articles to avoid ambiguity.
                continue

            text = page["text"]
            text = re.sub(
                r"&lt;a href=\"([^\"]*)\"&gt;([^\/]*)&lt;/a&gt;",
                lambda m: f'<a href="{urllib.parse.unquote(m.group(1))}">{m.group(2)}</a>',
                text,
            )

            for context in text.split("\n"):
                for m in re.finditer(r"<a href=\"([^\"]*)\">([^<]*)</a>", context):
                    outlink_title = m.group(1)

                    if (lang, outlink_title) not in wiki2qid:
                        # not an event
                        continue

                    corrected_context = (
                        context[: m.start(0)]
                        + f"<a> {m.group(2)} </a>"
                        + context[m.end(0) :]
                    )
                    corrected_context = re.sub(
                        r"<a href=\"([^\"]*)\">([^<]*)</a>", r"\2", corrected_context
                    )
                    wiki_inlinks[(lang, outlink_title)] += [(title, corrected_context)]

    logging.debug(f"skipped {skipped_page_count} pages at {wiki_file_path}")

    return wiki_inlinks


def write_out(wiki_inlinks_list: List[Dict], wiki2qid: Dict, out_path: Path):

    """
    Wikidata Item, Wikipedia Language, Wikipedia Title, Wikipedia Inlink Title, Context
    """

    qid2inlinks = defaultdict(list)

    logging.info("writing output")
    pbar = tqdm()

    for wiki_inlinks in wiki_inlinks_list:
        for (lg, wiki_title), contexts in wiki_inlinks.items():
            pbar.update(1)
            qid = wiki2qid[(lg, wiki_title)]
            for inlink_title, ann_context in contexts:
                qid2inlinks[qid] += [
                    {
                        "lang": lg,
                        "wikipedia_title": wiki_title,
                        "wikipedia_inlink_title": inlink_title,
                        "context": ann_context,
                    }
                ]

    pbar.close()

    out_dict = {
        "Wikidata Item": [],
        "Wikipedia Language": [],
        "Wikipedia Title": [],
        "Wikipedia Inlink Title": [],
        "Context": [],
    }
    for qid, inlinks in qid2inlinks.items():
        for inlink in inlinks:
            out_dict["Wikidata Item"] += [qid]
            out_dict["Wikipedia Language"] += [inlink["lang"]]
            out_dict["Wikipedia Title"] += [inlink["wikipedia_title"]]
            out_dict["Wikipedia Inlink Title"] += [inlink["wikipedia_inlink_title"]]
            out_dict["Context"] += [inlink["context"]]

    df = pd.DataFrame.from_dict(out_dict)
    df.to_csv(out_path, sep="\t", index=False, quoting=csv.QUOTE_NONE)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(
        description="collect contexts from Wikipedia articles"
    )
    parser.add_argument("wiki_txt_dir", type=Path)
    parser.add_argument("wikidata_inlinks", type=Path, help="Wikidata inlinks")
    parser.add_argument("out", type=Path)

    args = parser.parse_args()

    langs = []
    for dir_path in args.wiki_txt_dir.iterdir():
        langs += [dir_path.name]

    wd2wikipedia_links = load_wd2wikipedia_links(args.wikidata_inlinks)
    wiki2qid = get_title2wd(wd2wikipedia_links)

    txt_files = []

    for lg in langs:
        for dir_path in (args.wiki_txt_dir / f"{lg}").iterdir():
            for file_path in dir_path.iterdir():
                txt_files += [(lg, file_path)]

    logging.info("collecting mentions")

    wiki_inlinks = []
    for txt_file in tqdm(txt_files):
        wiki_inlinks += [collect_inlink_titles(wiki2qid, txt_file)]

    write_out(wiki_inlinks, wiki2qid, args.out)

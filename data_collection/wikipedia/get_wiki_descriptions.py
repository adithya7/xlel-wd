import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm
from utils import load_title2wd


def collect_title_desc(wiki2item: Dict, wiki_inputs: Tuple[str, Path]):

    lang, wiki_file_path = wiki_inputs

    wiki_desc = []
    skipped_page_count = 0

    with open(wiki_file_path, "r", errors="ignore", encoding="utf-8") as rf:
        for line in rf:
            try:
                page = json.loads(line.strip())
            except:
                skipped_page_count += 1
                continue

            title = page["title"]

            if (lang, title) in wiki2item:
                wd_item = wiki2item[(lang, title)]
                first_para = page["text"].split("\n")[0]
                first_para = re.sub(
                    r"&lt;a href=\"([^\"]*)\"&gt;([^&]*)&lt;/a&gt;", r"\2", first_para
                )
                wiki_desc += [(wd_item, lang, title, first_para)]

    logging.debug(f"skipped {skipped_page_count} pages at {wiki_file_path}")

    return wiki_desc


def write_out(data: List, out_path: Path):

    """
    Wikidata Item, Wikipedia Language, Wikipedia Title, Wikipedia Description
    """

    with open(out_path, "w") as wf:
        wf.write(
            f"Wikidata Item\tWikipedia Language\tWikipedia Title\tWikipedia Description\n"
        )
        for row in data:
            wf.write("\t".join(row))
            wf.write("\n")


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(
        description="collect Wikipedia descriptions for event articles"
    )
    parser.add_argument("wiki_txt_dir", type=Path)
    parser.add_argument(
        "wd2wikipedia_links", type=Path, help="Wikidata to Wikipedia links"
    )
    parser.add_argument("out", type=Path)

    args = parser.parse_args()

    langs = []
    for dir_path in args.wiki_txt_dir.iterdir():
        langs += [dir_path.name]

    wiki2item = load_title2wd(args.wd2wikipedia_links)

    txt_files = []

    for lg in langs:
        for dir_path in (args.wiki_txt_dir / f"{lg}").iterdir():
            for file_path in dir_path.iterdir():
                txt_files += [(lg, file_path)]

    logging.info("collecting mentions")

    out = []
    for txt_file in tqdm(txt_files):
        out += collect_title_desc(wiki2item, txt_file)

    write_out(out, args.out)

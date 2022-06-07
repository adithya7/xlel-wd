import argparse
import csv
import json
import logging
import re
import urllib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from tqdm import tqdm

from utils import get_wiki_title2wd, get_wn_title2wd, load_wd2wn_links, load_wn_dates, load_xlel_label_desc


def collect_valid_lang_qids(wd2wiki_df: pd.DataFrame):
    # for cases where the inlink is to a Wikinews event page,
    #   we also make sure that there is a corresponding Wikipedia event page in that language
    #   so that we can use the same label_desc file in multilingual setup
    valid_lg_qids = set()
    for qid, lg in zip(wd2wiki_df["Wikidata Item"], wd2wiki_df["Wikipedia Language"]):
        valid_lg_qids.add((qid, lg))
    return valid_lg_qids


def collect_inlink_titles(
    wiki2qid: Dict, wn2qid: Dict, wn_files: Tuple[str, Path], id2dates: Dict, valid_lg_qids: Set
):

    lang, wn_file_path = wn_files

    wn_inlinks = defaultdict(list)
    skipped_page_count = 0

    skipped_pub_date = 0
    skipped_lg_qid = 0

    outlink_counts = defaultdict(int)

    with open(wn_file_path, "r", errors="ignore", encoding="utf-8") as rf:
        for line in rf:
            try:
                page = json.loads(line.strip())
            except:
                skipped_page_count += 1
                continue

            title = page["title"]

            lang_title = (lang, title)
            if (lang_title in wiki2qid) or (lang_title in wn2qid):
                # the current Wikipedia (or Wikinews) page is also a seed article
                # skip inlinks from seed articles to avoid ambiguity.
                continue

            page_id = page["id"]
            text = page["text"]

            # init hyperlink
            text = re.sub(
                r"&lt;a href=\"([^\"]*)\"&gt;([^\/]*)&lt;/a&gt;",
                lambda m: f'<a href="{urllib.parse.unquote(m.group(1))}">{m.group(2)}</a>',
                text,
            )
            # links to Wikipedia pages, text span doesn't match Wikipedia title
            # <a href="w">2008–2009 Israel–Gaza conflict|23 day offensive</a>
            text = re.sub(
                r'<a href="w">([^\|<]+)\|([^\|<]+)</a>',
                lambda m: f'<a href="wiki:{m.group(1)}">{m.group(2)}</a>',
                text,
            )
            # <a href="w:Gaza Strip">Gaza</a>
            # <a href=":w:Gaza Strip">Gaza</a>
            text = re.sub(
                r'<a href="[:]{0,1}w:([^>]+)">([^\|<]+)</a>',
                lambda m: f'<a href="wiki:{m.group(1)}">{m.group(2)}</a>',
                text,
            )
            # links to Wikipedia pages, text span matches Wikipedia title
            # <a href="w">Hamas</a>
            text = re.sub(
                r'<a href="w">([^<]+)</a>', lambda m: f'<a href="wiki:{m.group(1)}">{m.group(1)}</a>', text
            )
            # other cases, these are intralinks to Wikinews pages
            # <a href="England">England</a>, <a href="Agence France-Presse">AFP</a>

            # text = re.sub(r"[\n]+", " ", text)
            # for context in text.split("\n"):
            for m in re.finditer(r"<a href=\"(?P<outlink>[^\"]*)\">(?P<mention>[^<]*)</a>", text):
                if m["outlink"].startswith("wiki:"):
                    # links to Wikipedia
                    outlink_tuple = (lang, re.sub(r"^wiki:(.*)", r"\1", m["outlink"]))
                    outlink_counts["wiki"] += 1
                    if outlink_tuple not in wiki2qid:
                        # not an event
                        continue
                else:
                    # links to Wikinews
                    outlink_tuple = (lang, m["outlink"])
                    outlink_counts["wn"] += 1
                    if outlink_tuple not in wn2qid:
                        # not an event
                        continue
                    if outlink_tuple in wn2qid and (wn2qid[outlink_tuple], lang) not in valid_lg_qids:
                        skipped_lg_qid += 1
                        continue

                # prepare context
                full_context = text[: m.start(0)] + f"<a> {m['mention']} </a>" + text[m.end(0) :]
                # remove all other hyperlinks from the context
                full_context = re.sub(r"<a href=\"([^\"]*)\">([^<]*)</a>", r"\2", full_context)

                # prepare short context (paragraph)
                paragraphs = full_context.split("\n")
                short_context = list(filter(lambda x: "<a>" in x and "</a>" in x, paragraphs))
                assert len(short_context) == 1
                short_context = short_context[0]

                # remove newline characters from full context
                full_context = re.sub(r"[\n]+", " ", full_context)

                # prepare meta information (publication date)
                pub_dates = id2dates.get((lang, page_id), [])
                if len(pub_dates) != 1:
                    skipped_pub_date += 1
                    continue
                pub_dates = pub_dates[0]

                wn_inlinks[outlink_tuple] += [(page_id, title, pub_dates, short_context, full_context)]

    if skipped_page_count > 0:
        logging.debug(f"skipped {skipped_page_count} pages at {wn_file_path}")

    if skipped_pub_date > 0:
        logging.warning(f"skipped {skipped_pub_date} contexts due to unclear publication dates")

    if skipped_lg_qid > 0:
        logging.warning(f"skipped {skipped_lg_qid} contexts due to missing language Wikipedia event")

    return wn_inlinks, outlink_counts


def write_out(wn_inlinks_list: List[Dict], wiki2qid: Dict, wn2qid: Dict, out_path: Path):

    """
    Wikidata Item, Wikipedia Language, Wikipedia Title, Wikipedia Inlink Title, Context
    """

    qid2inlinks = defaultdict(list)

    logging.info("writing output")
    pbar = tqdm()

    for wn_inlinks in wn_inlinks_list:
        for (lg, wiki_title), contexts in wn_inlinks.items():

            pbar.update(1)

            event_source = "wikipedia"
            event_tuple = (lg, wiki_title)
            qid = wiki2qid.get(event_tuple, None)
            if qid is None:
                event_source = "wikinews"
                qid = wn2qid.get(event_tuple, None)
            assert qid is not None
            for inlink_id, inlink_title, inlink_date, short_ann_context, full_ann_context in contexts:
                qid2inlinks[qid] += [
                    {
                        "lang": lg,
                        "event_title": wiki_title,
                        "wikinews_inlink_id": inlink_id,
                        "wikinews_inlink_title": inlink_title,
                        "wikinews_pub_date": inlink_date,
                        "short_context": short_ann_context,
                        "full_context": full_ann_context,
                        "event_source": event_source,
                    }
                ]

    pbar.close()

    out_dict = {
        "Wikidata Item": [],
        "Language": [],
        "Title": [],
        "Inlink ID": [],
        "Inlink Title": [],
        "Inlink Date": [],
        "Context": [],
        "Long Context": [],
        "Event Source": [],
    }
    for qid, inlinks in qid2inlinks.items():
        for inlink in inlinks:
            out_dict["Wikidata Item"] += [qid]
            out_dict["Language"] += [inlink["lang"]]
            out_dict["Title"] += [inlink["event_title"]]
            out_dict["Inlink ID"] += [inlink["wikinews_inlink_id"]]
            out_dict["Inlink Title"] += [inlink["wikinews_inlink_title"]]
            out_dict["Inlink Date"] += [inlink["wikinews_pub_date"]]
            out_dict["Context"] += [inlink["short_context"]]
            out_dict["Long Context"] += [inlink["full_context"]]
            out_dict["Event Source"] += [inlink["event_source"]]

    df = pd.DataFrame.from_dict(out_dict)
    df.to_csv(out_path, sep="\t", index=False, quoting=csv.QUOTE_NONE)


if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
        handlers=[logging.StreamHandler()],
    )

    parser = argparse.ArgumentParser(description="collect contexts from Wikipedia articles")
    parser.add_argument("wn_txt_dir", type=Path)
    parser.add_argument("wd2wiki", type=Path, help="Wikidata links to Wikipedia (label_desc.tsv)")
    parser.add_argument("wd2wn", type=Path, help="Wikidata links to Wikinews")
    parser.add_argument("wn_meta", type=Path, help="path to directory Wikinews meta information")

    parser.add_argument("out", type=Path)

    args = parser.parse_args()

    langs = []
    for dir_path in args.wn_txt_dir.iterdir():
        langs += [dir_path.name]

    # collect mapping to QID from both Wikipedia and Wikinews titles
    wd2wiki_df = load_xlel_label_desc(args.wd2wiki)
    wd2wn_df = load_wd2wn_links(args.wd2wn)
    wiki2qid, qid2wiki = get_wiki_title2wd(wd2wiki_df)
    wn2qid = get_wn_title2wd(wd2wn_df)

    # collect valid lang, qids pairs
    valid_lg_qids = collect_valid_lang_qids(wd2wiki_df)

    # collect meta information (publication date) for Wikinews articles
    id2dates = load_wn_dates(args.wn_meta)

    txt_files = []

    for lg in langs:
        for dir_path in (args.wn_txt_dir / f"{lg}").iterdir():
            for file_path in dir_path.iterdir():
                txt_files += [(lg, file_path)]

    logging.info("collecting mentions")

    outlink_counts = {}
    for lg in langs:
        outlink_counts[lg] = defaultdict(int)

    wn_inlinks = []
    for txt_file in tqdm(txt_files):
        file_wn_inlinks, file_outlink_counts = collect_inlink_titles(
            wiki2qid, wn2qid, txt_file, id2dates, valid_lg_qids
        )
        wn_inlinks += [file_wn_inlinks]
        outlink_counts[txt_file[0]]["wiki"] += file_outlink_counts["wiki"]
        outlink_counts[txt_file[0]]["wn"] += file_outlink_counts["wn"]

    write_out(wn_inlinks, wiki2qid, wn2qid, args.out)

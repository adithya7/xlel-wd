import argparse
import re
from pathlib import Path

from tqdm import tqdm

lg2regex = {
    "en": r"'(?P<date>[a-zA-Z]+_[0-9]{1,2},_2[0-9]{3})'",
    "de": r"'(?P<date>[0-9]{1,2}\.[0-9]{1,2}\.2[0-9]{3})'",
    "es": r"'(?P<date>[0-9]{1,2}_de_[^']*_de_2[0-9]{3})'",
    "pt": r"'(?P<date>[0-9]{1,2}_de_[^']*_de_2[0-9]{3})'",
    "ca": r"'(?P<date>[0-9]{1,2}_de_[^']*_del_2[0-9]{3})'",
    "ar": r"'(?P<date>[0-9]{1,2}_[^']*_2[0-9]{3})'",
    "el": r"'(?P<date>[0-9]{1,2}_[^']*_2[0-9]{3})'",
    "fr": r"'(?P<date>[0-9]{1,2}_[^']*_2[0-9]{3})'",
    "he": r"'(?P<date>[0-9]{1,2}_[^']*_2[0-9]{3})'",
    "it": r"'(?P<date>[0-9]{1,2}_[^']*_2[0-9]{3})'",
    "nl": r"'(?P<date>[0-9]{1,2}_[^']*_2[0-9]{3})'",
    "pl": r"'(?P<date>[0-9]{1,2}_[^']*_2[0-9]{3})'",
    "ro": r"'(?P<date>[0-9]{1,2}_[^']*_2[0-9]{3})'",
    "ru": r"'(?P<date>[0-9]{1,2}_[^']*_2[0-9]{3})'",
    "sv": r"'(?P<date>[0-9]{1,2}_[^']*_2[0-9]{3})'",
    "uk": r"'(?P<date>[0-9]{1,2}_[^']*_2[0-9]{3})'",
    "ja": r"'(?P<date>2[0-9]{3}[^']*[0-9]{1,2}[^']*[0-9]{1,2}[^']*)'",
    "zh": r"'(?P<date>2[0-9]{3}[^']*[0-9]{1,2}[^']*[0-9]{1,2}[^']*)'",
    "ko": r"'(?P<date>2[0-9]{3}[^']*_[0-9]{1,2}[^']*_[0-9]{1,2}[^']*)'",
    "cs": r"'(?P<date>[0-9]{1,2}\._[^']*_2[0-9]{3}[\.]{0,1})'",
    "fi": r"'(?P<date>[0-9]{1,2}\._[^']*_2[0-9]{3}[\.]{0,1})'",
    "no": r"'(?P<date>[0-9]{1,2}\._[^']*_2[0-9]{3}[\.]{0,1})'",
    "sr": r"'(?P<date>[0-9]{1,2}\._[^']*_2[0-9]{3}[\.]{0,1})'",
    "hu": r"'(?P<date>2[0-9]{3}\._[^']*_[0-9]{1,2}\.)'",
    "tr": r"'(?P<date>2[0-9]{3}/[0-9]{1,2}/[0-9]{1,2})'",
    "bg": r"'(?P<date>Новини_от_[0-9]{1,2}_[^']*_2[0-9]{3}_г[\.]{0,1})'",
    "th": r"'(?P<date>[0-9]{1,2}_[^']*_2[0-9]{3})'",
    "ta": r"'(?P<date>[^']*_[0-9]{1,2},_2[0-9]{3})'",
}


def collect_mapping(script: str, lg_regex: str):
    id_tuples = []
    # r"\((?P<id>[0-9]+),'(?P<date>[a-zA-Z]+_[0-9]{1,2},_2[0-9]{3})','(?P<title>[^']+)'," : only works for English
    # for match in re.finditer(r"\((?P<id>[0-9]+),'(?P<date>[^']*2\d{3}[^']*)','(?P<title>[^']+)',", script):
    # for match in re.finditer(r"\((?P<id>[0-9]+),'(?P<date>[^']*2\d{3}[^']*)',", script):
    for match in re.finditer(r"\((?P<id>[0-9]+)," + lg_regex, script):
        id_tuples += [(match["id"], match["date"])]
    return id_tuples


def parse_sql_script(file_path: Path, lg: str):
    id_tuples = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as rf:
        for line in tqdm(rf):
            if line.startswith("INSERT INTO `categorylinks` VALUES"):
                id_tuples += collect_mapping(line, lg2regex[lg])
    return id_tuples


def write_out(data, file_path: Path):
    with open(file_path, "w") as wf:
        wf.write("ID\tDate\n")
        for _item in data:
            wf.write("\t".join(_item) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="collect publication dates for Wikinews articles")
    parser.add_argument("lg", type=str, help="language")
    parser.add_argument("categorylinks", type=Path, help="path to -categorylinks.sql file")
    parser.add_argument("out", type=Path, help="path to write output tsv")

    args = parser.parse_args()

    if args.lg not in lg2regex:
        print(f"skipping {args.lg}, no regex found!")
    else:
        id_tuples = parse_sql_script(args.categorylinks, args.lg)
        write_out(id_tuples, args.out)

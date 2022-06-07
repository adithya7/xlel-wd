import argparse
import re
from pathlib import Path

from tqdm import tqdm


def clean_xml(inp_file_path: Path, out_file_path: Path):
    with open(inp_file_path, "r") as rf, open(out_file_path, "w") as wf:
        for line in tqdm(rf):
            txt = line
            txt = re.sub(r"\{\{w\|([^\}]*)\}\}", r"[[w|\1]]", txt, re.I)
            txt = re.sub(r"\{\{w:([^\}]*)\}\}", r"[[w:\1]]", txt, re.I)
            
            wf.write(txt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="clean hyperlinks in Wikinews XML dumps")
    parser.add_argument("inp", type=Path, help="input XML")
    parser.add_argument("out", type=Path, help="output XML")

    args = parser.parse_args()

    clean_xml(args.inp, args.out)

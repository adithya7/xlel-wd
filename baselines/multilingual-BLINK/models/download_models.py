import argparse
import logging
import subprocess
from pathlib import Path

URLS = {
    "wikipedia-zero-shot-crosslingual": "https://cmu.box.com/shared/static/qgu528hu55x3sev5i9v8agiygmvbnx8y.tgz",
    "wikipedia-zero-shot-multilingual": "https://cmu.box.com/shared/static/kx73ohzmvda4gg95o6byszdc4aek8fes.tgz",
}


def parse_args():
    parser = argparse.ArgumentParser(description="download models")
    parser.add_argument(
        "--task",
        choices=["multilingual", "crosslingual"],
        help="wikipedia-zero-shot-<task>",
    )
    parser.add_argument("--out", type=Path, help="path to download and extract files")
    return parser.parse_args()


def download_and_extract(task: str, out: Path):

    file_name = f"wikipedia-zero-shot-{task}"
    logging.info(f"downloading {file_name}.tgz to {out}")

    out.mkdir(exist_ok=True, parents=True)
    subprocess.run(
        [
            "wget",
            "-q",
            "--show-progress",
            URLS[file_name],
            "-O",
            out / f"{file_name}.tgz",
        ]
    )
    subprocess.run(["tar", "-xvf", out / f"{file_name}.tgz", "-C", out])
    subprocess.run(["rm", out / f"{file_name}.tgz"])


if __name__ == "__main__":
    logging.basicConfig(
        format="%(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler()],
    )
    args = parse_args()
    download_and_extract(**vars(args))

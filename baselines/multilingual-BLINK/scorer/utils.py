import logging
import json
from pathlib import Path
from tqdm import tqdm


def read_dataset(data_dir: Path, mode: str):

    file_path = data_dir / f"{mode}.jsonl"
    samples = []

    logging.debug(f"loading sample from {file_path}")
    with open(file_path, "r", encoding="utf-8") as rf:
        for line in tqdm(rf):
            samples += [json.loads(line.strip())]

    return samples


def load_event_dict(file_path: Path):

    assert file_path is not None, "Error! event_dict_path is empty."

    id2title, id2desc = {}, {}
    event_count = 0

    logging.debug(f"Loading event descriptions from path: {file_path}")
    with open(file_path, "rt") as f:
        for line in f:
            sample = json.loads(line.rstrip())
            title = sample["label_title"]
            desc = sample["label_desc"]
            if desc != desc:
                desc = ""
            event_id = int(sample["label_id"])

            # storing language titles and descriptions are necessary for multilingual task setup
            lang = sample["label_lang"]
            if lang not in id2title:
                id2title[lang] = {}
                id2desc[lang] = {}

            id2title[lang][event_id] = title
            id2desc[lang][event_id] = desc
            event_count += 1

    logging.debug(
        f"loaded {event_count} event descriptions across {len(id2title)} languages from the dictionary"
    )

    return id2title, id2desc


class Stats:
    def __init__(self, top_k=1000):
        self.cnt = 0
        self.hits = []
        self.top_k = top_k
        self.rank = [1, 4, 8, 16, 32, 64, 100, 128, 256, 512]
        self.LEN = len(self.rank)
        for i in range(self.LEN):
            self.hits.append(0)

    def add(self, idx):
        self.cnt += 1
        if idx == -1:
            return
        for i in range(self.LEN):
            if idx < self.rank[i]:
                self.hits[i] += 1

    def extend(self, stats):
        self.cnt += stats.cnt
        for i in range(self.LEN):
            self.hits[i] += stats.hits[i]

    def output(self):
        scores = []
        # output_json = "Total: %d examples." % self.cnt
        for i in range(self.LEN):
            if self.top_k < self.rank[i]:
                break
            scores += [self.hits[i] / float(self.cnt)]
            # output_json += " r@%d: %.4f" % (self.rank[i], self.hits[i] / float(self.cnt))
        # return output_json
        return self.cnt, scores

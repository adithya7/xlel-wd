import argparse
import json
from copy import deepcopy
from pathlib import Path

from datasets import get_dataset_config_names, load_dataset


def prepare_task_data(config: str, task: str, out_dir: Path):
    """
    Prepare data in the BLINK format.
    Loads mentions and event dictionary
    """

    # load mentions from HF datasets repo
    configs = get_dataset_config_names("adithya7/xlel_wd")
    assert config in configs, f"invalid config: {config}"
    xlel_wd_dataset = load_dataset("adithya7/xlel_wd", config)

    # load dictionary from HF datasets repo
    xlel_wd_dictionary = load_dataset("adithya7/xlel_wd_dictionary")

    # create a mapping from label_id to label_title and label_desc
    # to include label_title and label_desc in mention samples
    event2desc = {}
    # output label dictionary, keep a list of all label_ids and their descriptions
    label_dict = []
    for event in xlel_wd_dictionary["dictionary"]:
        if event["label_id"] not in event2desc:
            event2desc[event["label_id"]] = {}
        event2desc[event["label_id"]][event["label_lang"]] = (
            event["label_title"],
            event["label_desc"],
        )
        if task == "crosslingual" and event["label_lang"] == "en":
            label_dict += [event]
        elif task == "multilingual":
            label_dict += [event]

    # write label dictionary
    with open(out_dir / "label_dict.jsonl", "w") as wf:
        for event in label_dict:
            wf.write(json.dumps(event, ensure_ascii=False) + "\n")

    # label mentions for each dataset split
    # include label_title and label_description (depd. on the task), rewrite the jsonlines files
    for data_split in xlel_wd_dataset:
        out_mentions = []
        for mention in xlel_wd_dataset[data_split]:
            out_mention = deepcopy(mention)
            label_id = mention["label_id"]
            lg = mention["context_lang"]
            if task == "multilingual":
                out_mention["label_title"] = event2desc[label_id][lg][0]
                out_mention["label_description"] = event2desc[label_id][lg][1]
            elif task == "crosslingual":
                out_mention["label_title"] = event2desc[label_id]["en"][0]
                out_mention["label_description"] = event2desc[label_id]["en"][1]
            out_mentions += [out_mention]

        with open(out_dir / f"{data_split}.jsonl", "w") as wf:
            for mention in out_mentions:
                wf.write(json.dumps(mention, ensure_ascii=False) + "\n")


def load_args():
    parser = argparse.ArgumentParser(
        description="load HF dataset and convert to blink format"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=[
            "wikipedia-zero-shot",
            "wikinews-zero-shot",
            "wikinews-cross-domain",
        ],
        help="experiment config name",
    )
    parser.add_argument(
        "--task", type=str, choices=["multilingual", "crosslingual"], help="task name"
    )
    parser.add_argument("--out-dir", type=Path, help="(output) BLINK format data path")

    return parser.parse_args()


if __name__ == "__main__":

    args = load_args()

    args.out.mkdir(exist_ok=True)
    prepare_task_data(args.config, args.task, args.out)

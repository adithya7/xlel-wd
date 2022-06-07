import argparse
from collections import defaultdict
import logging
import json
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm

from utils import read_dataset, load_event_dict, Stats


def score_retriever(args):
    samples = read_dataset(args.samples, args.mode)
    id2title, id2desc = load_event_dict(args.dict)
    preds = torch.load(args.preds / f"{args.mode}.t7")

    # gold label id for each sample in predictions
    gold_label_ids = preds["label_ids"]
    # label ids for the topk candidates for each sample in predictions
    topk_candidate_ids = preds["topk_candidate_ids"]
    # dataset indices for each sample in predictions
    data_indices = preds["data_idx"]

    # Stats per language
    lg_stats = {}

    # retrieve preds with label title and descriptions (for analysis)
    full_preds = []

    for gold_label_id, topk_ids, data_idx in tqdm(zip(gold_label_ids, topk_candidate_ids, data_indices)):
        sample = samples[data_idx.item()]

        pos = -1
        for k_idx, topk_id in enumerate(topk_ids):
            if topk_id.item() == gold_label_id.item():
                pos = k_idx

        sample_lg = sample["context_lang"]
        if sample_lg not in lg_stats:
            lg_stats[sample_lg] = Stats(top_k=args.topk)
        lg_stats[sample_lg].add(pos)

        pred_label_id = int(topk_ids[0].item())
        pred_dict = {
            "mention": sample["mention"],
            "context": f"{sample['context_left']}<E> {sample['mention']} </E>{sample['context_right']}",
            "context_lang": sample["context_lang"],
            "gold_label_id": int(gold_label_id.item()),
            "gold_label_title_desc": f"{sample['label_title']}: {sample['label_description']}",
            "pred_label_id": pred_label_id,
            "en_pred_label_title_desc": f"{id2title['en'][pred_label_id]}: {id2desc['en'][pred_label_id]}",
            "data_idx": data_idx.item(),
        }
        if sample["context_lang"] in id2title and sample["context_lang"] != "en":
            # only found in multilingual task
            pred_dict[
                "lg_pred_label_title_desc"
            ] = f"{id2title[sample['context_lang']][pred_label_id]}: {id2desc[sample['context_lang']][pred_label_id]}"

        full_preds += [pred_dict]

    total_stats = Stats(top_k=args.topk)
    langs = sorted(lg_stats.keys())
    for lg in langs:
        total_stats.extend(lg_stats[lg])

    for lg in langs:
        cnt, scores = lg_stats[lg].output()
        scores = "\t".join([f"{score*100:.1f}" for score in scores])
        logging.info(f"{lg}\t{cnt}\t{scores}")

    cnt, scores = total_stats.output()
    scores = "\t".join([f"{score*100:.1f}" for score in scores])
    logging.info(f"TOTAL\t{cnt}\t{scores}")

    with open(args.out, "w") as wf:
        json.dump(full_preds, wf, indent=2, ensure_ascii=False)


def score_ranker(args):

    samples = read_dataset(args.samples, args.mode)
    idx2score = {}

    with open(args.preds / f"{args.mode}_predictions.json", "r") as rf:
        for line in tqdm(rf):
            elm = json.loads(line.strip())
            lg = elm["context_lang"]
            idx2score[elm["data_idx"]] = int(int(elm["label_id"]) == int(elm["pred_id"]))

    unnormalized_lg_stats = defaultdict(list)
    normalized_lg_stats = defaultdict(list)

    for idx, sample in enumerate(samples):
        if idx in idx2score:
            normalized_lg_stats[sample["context_lang"]] += [idx2score[idx]]
            unnormalized_lg_stats[sample["context_lang"]] += [idx2score[idx]]
        else:
            unnormalized_lg_stats[sample["context_lang"]] += [0]

    langs = sorted(unnormalized_lg_stats.keys())

    # logging.info("normalized scores")
    # for lg in langs:
    #     values = np.array(normalized_lg_stats[lg])
    #     cnt = len(values)
    #     acc = np.sum(values) * 100 / cnt
    #     logging.info(f"{lg}\t{cnt}\t{acc:.1f}")

    # values = []
    # for lg in langs:
    #     values += normalized_lg_stats[lg]
    # values = np.array(values)
    # cnt = len(values)
    # acc = np.sum(values) * 100 / cnt
    # logging.info(f"TOTAL\t{cnt}\t{acc:.1f}")

    # logging.info("unnormalized scores")
    langs = sorted(unnormalized_lg_stats.keys())
    for lg in langs:
        values = np.array(unnormalized_lg_stats[lg])
        cnt = len(values)
        acc = np.sum(values) * 100 / cnt
        logging.info(f"{lg}\t{cnt}\t{acc:.1f}")

    values = []
    for lg in langs:
        values += unnormalized_lg_stats[lg]
    values = np.array(values)
    cnt = len(values)
    acc = np.sum(values) * 100 / cnt
    logging.info(f"TOTAL\t{cnt}\t{acc:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score linking outputs")
    parser.add_argument("--mode", type=str, help="dev or test")
    parser.add_argument("--samples", type=Path, help="path to dataset dir")
    parser.add_argument("--log", type=Path)

    subparsers = parser.add_subparsers(help="score retrieve or rank outputs")

    retrieve_parser = subparsers.add_parser("retrieve", help="score top-k predictions from retrieval model")
    retrieve_parser.add_argument("--preds", type=Path, help="path to retrieve top-k values")
    retrieve_parser.add_argument("--out", type=Path, help="write predictions")
    retrieve_parser.add_argument("--dict", type=Path, help="path to event dictionary")
    retrieve_parser.add_argument("--topk", type=int, default=8)
    retrieve_parser.set_defaults(func=score_retriever)

    rank_parser = subparsers.add_parser("rank", help="score predictions from ranker")
    rank_parser.add_argument("--preds", type=Path, help="path to predictions json")
    rank_parser.set_defaults(func=score_ranker)

    args = parser.parse_args()

    logging.basicConfig(
        format="%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log)],
    )

    args.func(args)

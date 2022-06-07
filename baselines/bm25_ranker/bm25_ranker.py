import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
import numpy as np
import logging

from rank_bm25 import BM25Okapi, BM25L, BM25Plus
from utils import Stats

from transformers import AutoTokenizer


def get_context_window(context_item: Dict, tokenize_fn, detokenize_fn, max_seq_len: int, use_mention: bool):
    """
    limit context to limited window
    """
    if use_mention:
        return context_item["mention"]

    mention_tokens = tokenize_fn(context_item["mention"])
    context_left_tokens = tokenize_fn(context_item["context_left"])
    context_right_tokens = tokenize_fn(context_item["context_right"])

    if len(mention_tokens) >= max_seq_len:
        # use full mention if its longer than max_seq_len by itself
        return detokenize_fn(mention_tokens)

    left_quota = (max_seq_len - len(mention_tokens)) // 2
    right_quota = max_seq_len - len(mention_tokens) - left_quota

    left_add = len(context_left_tokens)
    right_add = len(context_right_tokens)

    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = context_left_tokens[-left_quota:] + mention_tokens + context_right_tokens[:right_quota]

    return detokenize_fn(context_tokens)


def load_contexts(
    file_path: Path, tokenize_fn, detokenize_fn, max_seq_len: int = 128, use_mention: bool = False
):
    contexts = defaultdict(list)
    labels = defaultdict(list)
    data_indices = defaultdict(list)

    logging.info(f"loading contexts from {file_path}")
    data_idx = 0

    with open(file_path, "r") as rf:
        for line in tqdm(rf):
            data = json.loads(line)

            lang = data["context_lang"]
            context = get_context_window(
                data,
                tokenize_fn=tokenize_fn,
                detokenize_fn=detokenize_fn,
                max_seq_len=max_seq_len,
                use_mention=use_mention,
            )

            label_id = data["label_id"]

            contexts[lang] += [context]
            labels[lang] += [label_id]
            data_indices[lang] += [data_idx]
            data_idx += 1

    logging.info(f"found {data_idx} contexts across {len(contexts)} languages")

    return contexts, labels, data_indices


def load_labels(file_path: Path):
    labels = defaultdict(list)
    label_ids = defaultdict(list)

    logging.info(f"loading label descriptions from {file_path}")
    events = []

    with open(file_path, "r") as rf:
        for line in tqdm(rf):
            data = json.loads(line)

            lang = data["label_lang"]
            title = data["label_title"]
            desc = data["label_desc"]
            label_id = data["label_id"]

            desc = "" if desc != desc else desc  # replace with "" if desc is NaN

            full_desc = f"{title}\t{desc}"

            labels[lang] += [full_desc]
            label_ids[lang] += [label_id]
            events += [label_id]

    logging.info(
        f"found {len(events)} descriptions for {len(set(events))} events across {len(labels)} languages"
    )

    return labels, label_ids


def collect_topk(
    contexts: Dict[str, List[str]],
    context_label_ids: Dict[str, List[str]],
    data_indices: Dict[str, List[str]],
    labels: Dict[str, List[str]],
    label_ids: Dict[str, List[str]],
    k: int,
    tokenize_fn,
    bm25_variant: str,
):

    logging.info(f"predicting topk labels")

    stats = Stats(top_k=k)
    all_preds = []

    for lang in contexts:

        lg_stats = Stats(top_k=k)

        docs = [tokenize_fn(_item) for _item in labels[lang]]
        if bm25_variant == "BM25Okapi":
            bm25 = BM25Okapi(docs)
        elif bm25_variant == "BM25L":
            bm25 = BM25L(docs)
        elif bm25_variant == "BM25Plus":
            bm25 = BM25Plus(docs)

        queries = [tokenize_fn(_item) for _item in contexts[lang]]
        output = []
        for idx, query in tqdm(enumerate(queries), total=len(queries)):
            scores = bm25.get_scores(query)
            topk_indices = np.argsort(scores)[-k:][::-1]
            nns = [
                {"pred_label_desc": labels[lang][topk_idx], "pred_label_id": label_ids[lang][topk_idx]}
                for topk_idx in topk_indices
            ]
            output += [
                {
                    "data_idx": data_indices[lang][idx],
                    "context": contexts[lang][idx],
                    "gold_label": context_label_ids[lang][idx],
                    "nn_labels": nns,
                }
            ]

            # collect the position of gold label in the topk predictions, for scoring
            pointer = -1
            for idy, topk_idx in enumerate(topk_indices):
                if label_ids[lang][topk_idx] == context_label_ids[lang][idx]:
                    pointer = idy
            lg_stats.add(pointer)

        logging.info(f"lang: {lang}, {lg_stats.output()}")
        stats.extend(lg_stats)
        all_preds.extend(output)

    logging.info(f"langs: ALL, {stats.output()}")

    logging.info("prediction done!")

    return all_preds


def write_out(predictions: List, file_path: Path):
    with open(file_path, "w") as wf:
        for pred in tqdm(predictions):
            wf.write(json.dumps(pred, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="BM25 candidate ranking")
    parser.add_argument("contexts", type=Path, help="contexts jsonl")
    parser.add_argument("label_dict", type=Path, help="label description jsonl")
    parser.add_argument("out", type=Path)
    parser.add_argument("log", type=Path)
    parser.add_argument(
        "--bm25", type=str, default="BM25Okapi", help="bm25 variants among BM25Okapi, BM25L, BM25Plus"
    )
    parser.add_argument("--topk", type=int, default=64)
    parser.add_argument("--bert-tokenizer", type=str, default=None, help="BERT tokenizer")
    parser.add_argument("--use-mention", action="store_true", help="only use the mention text as context")
    parser.add_argument("--max-seq-len", type=int, default=128, help="maximum sequence length")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler(args.log)],
    )

    tokenize_fn = lambda x: x.lower().split(" ")
    detokenize_fn = lambda x: " ".join(x)
    if args.bert_tokenizer:
        bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_tokenizer)
        tokenize_fn = lambda x: bert_tokenizer.tokenize(x)
        detokenize_fn = lambda x: bert_tokenizer.convert_tokens_to_string(x)

    if args.use_mention:
        logging.info(f"using mention text as context")
    else:
        logging.info(f"max seq length of context: {args.max_seq_len}")
    contexts, context_label_ids, data_indices = load_contexts(
        args.contexts,
        tokenize_fn=tokenize_fn,
        detokenize_fn=detokenize_fn,
        max_seq_len=args.max_seq_len,
        use_mention=args.use_mention,
    )
    label_desc, label_ids = load_labels(args.label_dict)

    logging.info(f"using BM25 variant: {args.bm25}")
    topk_preds = collect_topk(
        contexts=contexts,
        context_label_ids=context_label_ids,
        data_indices=data_indices,
        labels=label_desc,
        label_ids=label_ids,
        k=args.topk,
        tokenize_fn=tokenize_fn,
        bm25_variant=args.bm25,
    )

    write_out(topk_preds, args.out)

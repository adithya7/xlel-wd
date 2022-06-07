# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import torch
import numpy as np
from copy import deepcopy
import json

from tqdm import tqdm

from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from blink.crossencoder.crossencoder import CrossEncoderRanker
import blink.crossencoder.data_process as data

import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser


logger = None


def load_event_dict(logger, params):

    path = params.get("event_dict_path", None)
    assert path is not None, "Error! event_dict_path is empty."

    id2title, id2desc = {}, {}
    event_count = 0

    logger.info("Loading event descriptions from path: " + path)
    with open(path, "rt") as f:
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

    logger.info(
        f"loaded {event_count} event descriptions across {len(id2title)} languages from the dictionary"
    )

    return id2title, id2desc


def modify(
    context_input, candidate_input, max_seq_length, sep_token_id: int = None, add_sep_token: bool = False
):
    """
    concatenate context and candidate into a single tensor
    add [SEP] token for XLM-RoBERTa and optional for bert-base-multilingual
    """
    new_input = []

    context_input = context_input.tolist()
    candidate_input = candidate_input.tolist()

    for i in range(len(context_input)):
        cur_input = context_input[i]
        cur_candidate = candidate_input[i]
        mod_input = []
        for j in range(len(cur_candidate)):
            # remove [CLS] token from candidate
            if add_sep_token:
                assert sep_token_id is not None
                sample = cur_input + [sep_token_id] + cur_candidate[j][1:]
            else:
                sample = cur_input + cur_candidate[j][1:]
            sample = sample[:max_seq_length]
            mod_input.append(sample)

        new_input.append(mod_input)

    return torch.LongTensor(new_input)


def predict(
    reranker,
    eval_dataloader,
    device,
    logger,
    context_length,
    use_gold_labels=True,
    save_predictions=True,
    silent=True,
    mode=None,
):
    reranker.model.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    normalized_eval_accuracy = 0.0
    normalized_eval_examples = 0

    all_logits = []
    all_pred_cand_indices = []

    for _, batch in enumerate(iter_):
        if use_gold_labels:
            batch = tuple(t.to(device) for t in batch)
            context_input = batch[0]
            label_input = batch[1]
        else:
            context_input = batch.to(device)

        with torch.no_grad():
            _, logits = reranker(context_input, context_length)

        logits = logits.detach().cpu().numpy()
        all_logits.extend(logits)

        if use_gold_labels:
            label_ids = label_input.cpu().numpy()
            batch_eval_accuracy, _ = utils.accuracy(logits, label_ids)
            normalized_eval_accuracy += batch_eval_accuracy
            normalized_eval_examples += context_input.size(0)

        if save_predictions:
            pred_cand_indices = np.argmax(logits, axis=1)
            all_pred_cand_indices.extend(pred_cand_indices)

    results["logits"] = all_logits
    if use_gold_labels:
        # normalized_eval_accuracy = -1
        if normalized_eval_examples > 0:
            normalized_eval_accuracy = normalized_eval_accuracy / normalized_eval_examples
        if logger:
            logger.info(f"{mode} eval accuracy: {normalized_eval_accuracy:.5f}")
        results["normalized_accuracy"] = normalized_eval_accuracy

    if save_predictions:
        results["pred_cand_indices"] = np.array(all_pred_cand_indices)

    return results


def write_predictions(file_path, samples, pred_label_ids, retrieved_data_indices, id2title, id2desc, logger):
    logger.info(f"total test samples: {len(samples)}")
    logger.info(f"retrieved test samples: {len(retrieved_data_indices)}")
    logger.info(f"ranked test samples: {len(pred_label_ids)}")

    assert len(retrieved_data_indices) == len(
        pred_label_ids
    ), f"# retrieved samples: {len(retrieved_data_indices)}, # ranker preds: {len(pred_label_ids)}"
    output = []

    # if language titles and descriptions are available for candidates (i.e., multilingual setup), load those instead of English
    use_multilingual_titles = len(id2title) > 1
    logger.info(f"use multilingual titles: {use_multilingual_titles}")

    for data_idx, pred in zip(retrieved_data_indices, pred_label_ids):
        sample = samples[data_idx]
        out_sample = deepcopy(sample)
        pred_label_id = int(pred[0])
        out_sample["pred_id"] = pred_label_id
        out_lg = sample["context_lang"] if use_multilingual_titles else "en"
        out_sample["pred_title"] = id2title[out_lg][pred_label_id]
        out_sample["pred_description"] = id2desc[out_lg][pred_label_id]
        out_sample["data_idx"] = int(data_idx)
        output += [json.dumps(out_sample, ensure_ascii=False)]

    logger.info(f"writing predictions to {file_path}")

    with open(file_path, "w") as wf:
        wf.write("\n".join(output) + "\n")


def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = CrossEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    # use [SEP] token to distingush context and candidate for XLM-RoBERTa
    # use type_token_ids for mBERT
    add_sep_token = model.config.type_vocab_size == 1

    device = reranker.device

    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]

    # use the event dictionary to load a mapping from label id to its title and description
    id2title, id2desc = load_event_dict(logger, params)

    # read dataset samples
    test_samples = utils.read_dataset(params["mode"], params["data_path"])
    logger.info(f"Read {len(test_samples)} {params['mode']} samples.")

    # load label ids and nn candidate label ids from biencoder predictions
    fname = os.path.join(params["topk_path"], f"{params['mode']}.t7")
    nn_data = torch.load(fname)
    idx2label, idx2nns = {}, {}
    for idx, data_idx in enumerate(nn_data["data_idx"].tolist()):
        idx2label[data_idx] = nn_data["label_ids"][idx]
        idx2nns[data_idx] = nn_data["topk_candidate_ids"][idx]

    # prepare data for crossencoder eval
    context_input, candidate_input, label_input = data.prepare_crossencoder_data(
        tokenizer=tokenizer,
        samples=test_samples,
        idx2label=idx2label,
        idx2nns=idx2nns,
        id2title=id2title,
        id2text=id2desc,
        max_context_length=params["max_context_length"],
        max_cand_length=params["max_cand_length"],
        logger=logger,
        debug=params["debug"],
    )
    # run evaluation and compute normalized accuracy using gold labels
    runEval = True if label_input is not None else False
    logger.info(f"found gold labels in the input data, running evaluation")

    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        if runEval:
            label_input = label_input[:max_n]

    # concatenate context and candidate inputs to pass as a single input to crossencoder
    context_input = modify(
        context_input,
        candidate_input,
        max_seq_length,
        add_sep_token=add_sep_token,
        sep_token_id=tokenizer.sep_token_id,
    )

    test_tensor_data = TensorDataset(context_input, label_input) if runEval else TensorDataset(context_input)
    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(test_tensor_data, sampler=test_sampler, batch_size=params["eval_batch_size"])

    # collect label ids for the topk candidates for samples with the gold event in the topk
    # retrieved data indices: indices of dataset samples that had gold label retrieved in the topk
    topk_candidate_label_ids, retrieved_data_indices = data.collect_topk_label_ids(
        idx2label=idx2label,
        idx2nns=idx2nns,
        samples=test_samples,
        logger=logger,
        debug=params["debug"],
        topk=params["top_k"],
    )

    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        test_samples = test_samples[:max_n]
        topk_candidate_label_ids = topk_candidate_label_ids[:max_n]
        if runEval:
            label_input = label_input[:max_n]

    results = predict(
        reranker,
        test_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
        use_gold_labels=runEval,
        save_predictions=params["save_preds"],
        silent=params["silent"],
        mode=params["mode"],
    )
    if params["save_preds"]:

        # extract the dataset label ids for the predictions
        print(topk_candidate_label_ids.shape)
        print(results["pred_cand_indices"].shape)
        assert (
            topk_candidate_label_ids.shape[0] == results["pred_cand_indices"].shape[0]
        ), f"mismatch in first dimensions of candidate label ids and predicted indices, {topk_candidate_label_ids.shape[0]}\t{results['pred_cand_indices'].shape[0]}"
        pred_label_ids = np.take_along_axis(
            topk_candidate_label_ids, np.expand_dims(results["pred_cand_indices"], axis=1), axis=1
        )
        write_predictions(
            os.path.join(params["output_path"], f"{params['mode']}_predictions.json"),
            test_samples,
            pred_label_ids,
            retrieved_data_indices,
            id2title=id2title,
            id2desc=id2desc,
            logger=logger,
        )


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_eval_args()

    args = parser.parse_args()
    print(args)

    params = args.__dict__

    mode_list = params["mode"].split(",")
    for mode in mode_list:
        new_params = params
        new_params["mode"] = mode
        main(new_params)

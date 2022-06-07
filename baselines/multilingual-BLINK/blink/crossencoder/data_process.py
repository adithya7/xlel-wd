# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch

import numpy as np
from tqdm import tqdm
import blink.biencoder.data_process as data
from blink.common.params import ENT_START_TAG, ENT_END_TAG


def prepare_crossencoder_mentions(
    tokenizer,
    samples,
    max_context_length=128,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    logger=None,
    debug=False,
):

    context_input_list = []  # samples X 128

    logger.info(f"collecting context encodings")

    for idx, sample in tqdm(enumerate(samples), total=len(samples)):
        context_tokens = data.get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
            logger=logger,
        )
        if context_tokens is None:
            # skip sample due to large mention length
            logger.info(f"skipped sample data idx: {idx}")
            continue
        tokens_ids = context_tokens["ids"]
        context_input_list.append(tokens_ids)
        if debug and len(context_input_list) >= 500:
            break

    context_input_list = np.asarray(context_input_list)
    return context_input_list


def prepare_crossencoder_candidates(
    tokenizer,
    idx2label,
    idx2nns,
    samples,
    id2title,
    id2text,
    max_cand_length=128,
    topk=100,
    logger=None,
    debug=False,
):

    logger.info(f"collecting candidate encodings")

    candidate_input_list = []  # samples X topk=10 X 128
    label_input_list = []  # samples

    # if language titles and descriptions are available for candidates (i.e., multilingual setup), load those instead of English
    use_multilingual_titles = len(id2title) > 1
    logger.info(f"use multilingual titles: {use_multilingual_titles}")

    for data_idx, sample in tqdm(enumerate(samples), total=len(samples)):
        if data_idx not in idx2label:
            logger.info(f"data_idx {data_idx} missing in the topk predictions")
            continue

        label = idx2label[data_idx]
        nns = idx2nns[data_idx]

        assert int(sample["label_id"]) == int(
            label
        ), f"mismatch between gold id in dataset and gold id found in topk predictions, data_idx: {data_idx}, sample_label_id: {sample['label_id']}, label: {label}"

        candidates = []

        label_id = -1
        for jdx, candidate_id in enumerate(nns[:topk]):

            if label.item() == candidate_id.item():
                label_id = jdx

            candidate_lg = sample["context_lang"] if use_multilingual_titles else "en"
            rep = data.get_candidate_representation(
                id2text[candidate_lg][candidate_id.item()],
                tokenizer,
                max_cand_length,
                id2title[candidate_lg][candidate_id.item()],
                logger=logger,
            )
            tokens_ids = rep["ids"]

            assert len(tokens_ids) == max_cand_length
            candidates.append(tokens_ids)

        label_input_list.append(label_id)
        candidate_input_list.append(candidates)

        if debug and len(label_input_list) >= 500:
            break

    label_input_list = np.asarray(label_input_list)
    candidate_input_list = np.asarray(candidate_input_list)

    return label_input_list, candidate_input_list


def collect_topk_label_ids(
    idx2label,
    idx2nns,
    samples,
    topk=100,
    logger=None,
    debug=False,
):
    logger.info(
        f"collecting label ids for the topk candidates for samples with the gold event in the topk predictions"
    )

    topk_candidate_label_ids = []
    retrieved_data_indices = []

    for data_idx, sample in tqdm(enumerate(samples), total=len(samples)):
        if data_idx not in idx2label:
            logger.info(f"data_idx {data_idx} missing in the topk predictions")
            continue

        label = idx2label[data_idx]
        nns = idx2nns[data_idx]

        assert int(sample["label_id"]) == int(
            label
        ), f"mismatch between gold id in dataset and gold id found in topk predictions, data_idx: {data_idx}, sample_label_id: {sample['label_id']}, label: {label}"

        label_id = -1
        for jdx, candidate_id in enumerate(nns[:topk]):

            if label.item() == candidate_id.item():
                label_id = jdx

        if label_id != -1:
            topk_candidate_label_ids += [nns.numpy()]
            retrieved_data_indices += [data_idx]

        if debug and len(topk_candidate_label_ids) >= 500:
            break

    topk_candidate_label_ids = np.array(topk_candidate_label_ids)
    retrieved_data_indices = np.array(retrieved_data_indices)

    return topk_candidate_label_ids, retrieved_data_indices


def filter_crossencoder_tensor_input(context_input_list, label_input_list, candidate_input_list):
    # remove the - 1 : examples for which gold is not among the candidates
    context_input_list_filtered = [
        x for x, y, z in zip(context_input_list, candidate_input_list, label_input_list) if z != -1
    ]
    label_input_list_filtered = [
        z for x, y, z in zip(context_input_list, candidate_input_list, label_input_list) if z != -1
    ]
    candidate_input_list_filtered = [
        y for x, y, z in zip(context_input_list, candidate_input_list, label_input_list) if z != -1
    ]
    return (
        context_input_list_filtered,
        label_input_list_filtered,
        candidate_input_list_filtered,
    )


def prepare_crossencoder_data(
    tokenizer,
    samples,
    idx2label,
    idx2nns,
    id2title,
    id2text,
    max_context_length,
    max_cand_length,
    logger,
    keep_all=False,
    debug=False,
    topk=16,
):

    # encode candidates (output of biencoder)
    label_input_list, candidate_input_list = prepare_crossencoder_candidates(
        tokenizer,
        idx2label,
        idx2nns,
        samples,
        id2title,
        id2text,
        max_cand_length=max_cand_length,
        logger=logger,
        debug=debug,
        topk=topk,
    )

    # encode mentions
    context_input_list = prepare_crossencoder_mentions(
        tokenizer, samples, max_context_length=max_context_length, logger=logger, debug=debug
    )

    assert (
        len(context_input_list) == len(label_input_list) == len(candidate_input_list)
    ), f"# contexts: {len(context_input_list)}, # labels: {len(label_input_list)}, # candidates: {len(candidate_input_list)}"

    logger.info(f"# contexts before filtering: {len(context_input_list)}")

    if not keep_all:
        # remove examples where the gold entity is not among the candidates
        (
            context_input_list,
            label_input_list,
            candidate_input_list,
        ) = filter_crossencoder_tensor_input(context_input_list, label_input_list, candidate_input_list)
    else:
        label_input_list = [0] * len(label_input_list)

    logger.info(f"# contexts after filtering: {len(context_input_list)}")

    context_input = torch.LongTensor(context_input_list)
    label_input = torch.LongTensor(label_input_list)
    candidate_input = torch.LongTensor(candidate_input_list)

    return (
        context_input,
        candidate_input,
        label_input,
    )

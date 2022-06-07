# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import math
from collections import defaultdict

import torch
from blink.common.params import ENT_END_TAG, ENT_START_TAG, ENT_TITLE_TAG
from torch.utils.data import TensorDataset
from tqdm import tqdm


def select_field(data, key1, key2=None):
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    logger=None,
    title_key="context_title",
    date_key="context_date",
):
    mention_tokens = []
    if sample[mention_key] and len(sample[mention_key]) > 0:
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    if len(mention_tokens) >= max_seq_length:
        logger.info(f"mention length {len(mention_tokens)} greater than {max_seq_length}")
        return None

    # if the context has title and date (e.g. wikinews), include meta information
    include_meta = title_key in sample and date_key in sample
    meta_tokens = []
    if (
        include_meta
        and sample[title_key]
        and len(sample[title_key]) > 0
        and sample[date_key]
        and len(sample[date_key]) > 0
    ):
        title_tokens = tokenizer.tokenize(sample[title_key])
        date_tokens = tokenizer.tokenize(sample[date_key])
        meta_tokens = title_tokens + [tokenizer.sep_token] + date_tokens + [tokenizer.sep_token]
    assert len(meta_tokens) + len(mention_tokens) < max_seq_length

    context_left = sample[context_key + "_left"]
    context_right = sample[context_key + "_right"]

    # tokenizer tokenizes the full text, but gives a warning if the left or right contexts are longer than model_max_length
    context_left = tokenizer.tokenize(context_left)
    context_right = tokenizer.tokenize(context_right)

    left_quota = (max_seq_length - len(mention_tokens) - len(meta_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - len(meta_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    # truncate left context from left and right context from right
    context_tokens = meta_tokens + context_left[-left_quota:] + mention_tokens + context_right[:right_quota]

    context_tokens = [tokenizer.cls_token] + context_tokens + [tokenizer.sep_token]
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def get_candidate_representation(
    candidate_desc, tokenizer, max_seq_length, candidate_title=None, title_tag=ENT_TITLE_TAG, logger=None
):
    cand_tokens = tokenizer.tokenize(candidate_desc)
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [tokenizer.cls_token] + cand_tokens + [tokenizer.sep_token]

    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [tokenizer.pad_token_id] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def process_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent=False,
    mention_key="mention",
    context_key="context",
    label_desc_key="label_description",
    label_title_key="label_title",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = []

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
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
            continue

        label_desc = sample[label_desc_key]
        label_title = sample[label_title_key]

        if label_desc != label_desc:
            label_desc = ""

        label_tokens = get_candidate_representation(
            label_desc,
            tokenizer,
            max_cand_length,
            candidate_title=label_title,
            title_tag=title_token,
            logger=logger,
        )
        label_idx = int(sample["label_id"])

        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
            "data_idx": [idx],
        }

        processed_samples.append(record)

    if debug and logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info("Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]]))
            logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
            logger.info("Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]]))
            logger.info("Src : %d" % sample["src"][0])
            logger.info("Label_id : %d" % sample["label_idx"][0])

    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"),
        dtype=torch.long,
    )
    cand_vecs = torch.tensor(
        select_field(processed_samples, "label", "ids"),
        dtype=torch.long,
    )
    label_idx = torch.tensor(
        select_field(processed_samples, "label_idx"),
        dtype=torch.long,
    )
    data_idx = torch.tensor(select_field(processed_samples, "data_idx"), dtype=torch.long)
    data = {
        "context_vecs": context_vecs,
        "cand_vecs": cand_vecs,
        "label_idx": label_idx,
        "data_idx": data_idx,
    }

    tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx, data_idx)

    return data, tensor_data


def process_lang_mention_data(
    samples,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent=False,
    mention_key="mention",
    context_key="context",
    label_desc_key="label_description",
    label_title_key="label_title",
    context_lang_key="context_lang",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    processed_samples = defaultdict(list)

    if debug:
        samples = samples[:200]

    if silent:
        iter_ = samples
    else:
        iter_ = tqdm(samples)

    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
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
            continue

        label_desc = sample[label_desc_key]
        label_title = sample[label_title_key]

        if label_desc != label_desc:
            label_desc = ""

        label_tokens = get_candidate_representation(
            label_desc,
            tokenizer,
            max_cand_length,
            candidate_title=label_title,
            title_tag=title_token,
            logger=logger,
        )
        label_idx = int(sample["label_id"])

        # data_idx is important to restore the ordering of the original dataset
        # crossencoder relies on the original ordering
        record = {
            "context": context_tokens,
            "label": label_tokens,
            "label_idx": [label_idx],
            "data_idx": [idx],
        }

        context_lang = sample[context_lang_key]
        processed_samples[context_lang].append(record)

    data = {}
    tensor_data = {}

    for lang in processed_samples:
        if debug and logger:
            logger.info(f"====Processed samples for language {lang}: ====")
            for sample in processed_samples[lang][:5]:
                logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
                logger.info("Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]]))
                logger.info("Label tokens : " + " ".join(sample["label"]["tokens"]))
                logger.info("Label ids : " + " ".join([str(v) for v in sample["label"]["ids"]]))
                logger.info("Src : %d" % sample["src"][0])
                logger.info("Label_id : %d" % sample["label_idx"][0])

        context_vecs = torch.tensor(
            select_field(processed_samples[lang], "context", "ids"),
            dtype=torch.long,
        )
        cand_vecs = torch.tensor(
            select_field(processed_samples[lang], "label", "ids"),
            dtype=torch.long,
        )
        label_idx = torch.tensor(
            select_field(processed_samples[lang], "label_idx"),
            dtype=torch.long,
        )
        data_idx = torch.tensor(select_field(processed_samples[lang], "data_idx"), dtype=torch.long)
        lang_data = {
            "context_vecs": context_vecs,
            "cand_vecs": cand_vecs,
            "label_idx": label_idx,
            "data_idx": data_idx,
        }
        lang_tensor_data = TensorDataset(context_vecs, cand_vecs, label_idx, data_idx)

        data[lang] = lang_data
        tensor_data[lang] = lang_tensor_data

    return data, tensor_data

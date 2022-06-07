# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import os
import torch
from tqdm import tqdm
from collections import defaultdict

from torch.utils.data import DataLoader, SequentialSampler

from blink.biencoder.biencoder import BiEncoderRanker
import blink.biencoder.data_process as data
import blink.biencoder.nn_prediction as nnquery
import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser


def load_event_dict(logger, params):
    """
    collect event dictionary for each language
    """
    path = params.get("event_dict_path", None)
    assert path is not None, "Error! event_dict_path is empty."

    event_list = defaultdict(list)
    logger.info("Loading event descriptions from path: " + path)
    with open(path, "rt") as f:
        for line in f:
            sample = json.loads(line.rstrip())
            title = sample["label_title"]
            desc = sample["label_desc"]
            if desc != desc:
                desc = ""
            event_id = int(sample["label_id"])
            lang = sample["label_lang"]
            event_list[lang] += [(event_id, title, desc)]

    for lang in event_list:
        logger.info(f"loaded {len(event_list[lang])} events from the dictionary for language {lang}")

    return event_list


def get_candidate_pool_tensor(
    event_desc_list,
    tokenizer,
    max_seq_length,
    logger,
):
    # TODO: add multiple thread process
    logger.info("Convert candidate text to id")
    cand_pool_ids = []
    cand_pool_reprs = []
    for event_desc_tuple in tqdm(event_desc_list):
        event_id, event_title, event_desc = event_desc_tuple

        rep = data.get_candidate_representation(
            event_desc,
            tokenizer,
            max_seq_length,
            event_title,
        )
        cand_pool_reprs.append(rep["ids"])
        cand_pool_ids.append(event_id)

    cand_pool_reprs = torch.LongTensor(cand_pool_reprs)
    assert int(cand_pool_reprs.shape[0]) == len(
        cand_pool_ids
    ), f"cand ids: {len(cand_pool_ids)}, cand pool: {int(cand_pool_reprs.shape[0])}"
    return cand_pool_reprs, cand_pool_ids


def encode_candidate(
    reranker,
    candidate_pool,
    encode_batch_size,
    silent,
    logger,
):

    reranker.model.eval()
    device = reranker.device
    sampler = SequentialSampler(candidate_pool)
    data_loader = DataLoader(candidate_pool, sampler=sampler, batch_size=encode_batch_size)
    if silent:
        iter_ = data_loader
    else:
        iter_ = tqdm(data_loader)

    cand_encode_list = None
    for step, batch in enumerate(iter_):
        cands = batch
        cands = cands.to(device)
        cand_encode = reranker.encode_candidate(cands)
        if cand_encode_list is None:
            cand_encode_list = cand_encode
        else:
            cand_encode_list = torch.cat((cand_encode_list, cand_encode))

    return cand_encode_list


def load_or_generate_candidate_pool(tokenizer, params, logger):

    event_desc_list = load_event_dict(logger, params)

    # generate candidate pools for each language
    candidate_pool = {}
    candidate_ids = {}
    for lang in event_desc_list:
        # compute candidate pool from event list
        lang_candidate_pool, lang_candidate_ids = get_candidate_pool_tensor(
            event_desc_list[lang],
            tokenizer,
            params["max_cand_length"],
            logger,
        )

        assert int(lang_candidate_pool.shape[0]) == len(
            lang_candidate_ids
        ), f"lang: {lang}, cand ids: {len(lang_candidate_ids)}, candidate pool: {int(lang_candidate_pool.shape[0])}"

        candidate_pool[lang] = lang_candidate_pool
        candidate_ids[lang] = lang_candidate_ids

    return candidate_pool, candidate_ids


def load_or_generate_candidate_encoding(reranker, candidate_pool_reprs, params, logger):

    candidate_encoding = {}
    for lang in candidate_pool_reprs:
        lang_candidate_encoding = encode_candidate(
            reranker,
            candidate_pool_reprs[lang],
            params["encode_batch_size"],
            silent=params["silent"],
            logger=logger,
        )
        assert (
            candidate_pool_reprs[lang].shape[0] == lang_candidate_encoding.shape[0]
        ), f"cand pool: {candidate_pool_reprs[lang].shape[0]}, cand encoding: {lang_candidate_encoding.shape[0]}"
        candidate_encoding[lang] = lang_candidate_encoding

    return candidate_encoding


def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer

    logger.info(f"Generating candidate pool.")
    candidate_pool_reprs, candidate_pool_ids = load_or_generate_candidate_pool(tokenizer, params, logger)

    logger.info(f"Generating candidate encoding.")
    candidate_encoding = load_or_generate_candidate_encoding(reranker, candidate_pool_reprs, params, logger)

    test_samples = utils.read_dataset(params["mode"], params["data_path"])
    logger.info("Read %d test samples." % len(test_samples))

    _, test_tensor_data = data.process_lang_mention_data(
        test_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )

    new_data = None

    for lang in test_tensor_data:

        logger.info(f"extracting topk candidates for samples from language: {lang}")
        lang_test_tensor_data = test_tensor_data[lang]

        test_sampler = SequentialSampler(lang_test_tensor_data)
        test_dataloader = DataLoader(
            lang_test_tensor_data, sampler=test_sampler, batch_size=params["eval_batch_size"]
        )

        lang_new_data = nnquery.get_topk_predictions(
            reranker,
            test_dataloader,
            candidate_pool_ids[lang],
            candidate_encoding[lang],
            params["silent"],
            logger,
            params["top_k"],
            save_predictions=False,
        )

        logger.info(f"extracted topk predictions for {len(lang_new_data['label_ids'])} input contexts")

        if new_data is None:
            new_data = lang_new_data
        else:
            for _key in new_data:
                new_data[_key] = torch.cat((new_data[_key], lang_new_data[_key]), 0)

    logger.info(f"extracted topk predictions for a total of {len(new_data['label_ids'])} input contexts")

    save_results = params.get("save_topk_result")
    if save_results:
        save_data_dir = os.path.join(
            params["output_path"],
            "top%d_candidates" % params["top_k"],
        )
        if not os.path.exists(save_data_dir):
            os.makedirs(save_data_dir)
        save_data_path = os.path.join(save_data_dir, "%s.t7" % params["mode"])
        torch.save(new_data, save_data_path)


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

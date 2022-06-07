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

from torch.utils.data import DataLoader, SequentialSampler

from blink.biencoder.biencoder import BiEncoderRanker
import blink.biencoder.data_process as data
import blink.biencoder.nn_prediction as nnquery
import blink.candidate_ranking.utils as utils
from blink.common.params import BlinkParser


def load_event_dict(logger, params):

    path = params.get("event_dict_path", None)
    assert path is not None, "Error! event_dict_path is empty."

    event_list = []
    logger.info("Loading event descriptions from path: " + path)
    with open(path, "rt") as f:
        for line in f:
            sample = json.loads(line.rstrip())
            title = sample["label_title"]
            desc = sample["label_desc"]
            if desc != desc:
                desc = ""
            event_id = int(sample["label_id"])
            event_list.append((event_id, title, desc))
            if params["debug"] and len(event_list) > 200:
                break

    logger.info(f"loaded {len(event_list)} events from the dictionary")

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


def load_or_generate_candidate_pool(
    tokenizer,
    params,
    logger,
    cand_pool_path,
    cand_pool_id_path,
):
    candidate_pool = None
    if cand_pool_path is not None:
        # try to load candidate pool from file
        try:
            logger.info("Loading pre-generated candidate pool from: ")
            logger.info(cand_pool_path)
            candidate_pool = torch.load(cand_pool_path)
        except:
            logger.info("Loading failed. Generating candidate pool")

    if candidate_pool is None:
        # compute candidate pool from event list
        event_desc_list = load_event_dict(logger, params)
        candidate_pool, candidate_ids = get_candidate_pool_tensor(
            event_desc_list,
            tokenizer,
            params["max_cand_length"],
            logger,
        )

        if cand_pool_path is not None:
            logger.info("Saving candidate pool.")
            torch.save(candidate_pool, cand_pool_path)
            with open(cand_pool_id_path, "w") as wf:
                wf.write("\n".join(candidate_ids))

    assert int(candidate_pool.shape[0]) == len(
        candidate_ids
    ), f"cand ids: {len(candidate_ids)}, candidate pool: {int(candidate_pool.shape[0])}"

    return candidate_pool, candidate_ids


def main(params):
    output_path = params["output_path"]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = BiEncoderRanker(params)
    tokenizer = reranker.tokenizer

    cand_encode_path = params.get("cand_encode_path", None)

    # candidate encoding is not pre-computed.
    # load/generate candidate pool to compute candidate encoding.
    cand_pool_path = params.get("cand_pool_path", None)
    cand_pool_id_path = params.get("cand_pool_id_path", None)
    candidate_pool_reprs, candidate_pool_ids = load_or_generate_candidate_pool(
        tokenizer, params, logger, cand_pool_path, cand_pool_id_path
    )

    candidate_encoding = None
    if cand_encode_path is not None:
        # try to load candidate encoding from path
        # if success, avoid computing candidate encoding
        try:
            logger.info("Loading pre-generated candidate encode path.")
            candidate_encoding = torch.load(cand_encode_path)
        except:
            logger.info("Loading failed. Generating candidate encoding.")

    if candidate_encoding is None:
        candidate_encoding = encode_candidate(
            reranker,
            candidate_pool_reprs,
            params["encode_batch_size"],
            silent=params["silent"],
            logger=logger,
        )

        if cand_encode_path is not None:
            # Save candidate encoding to avoid re-compute
            logger.info("Saving candidate encoding to file " + cand_encode_path)
            torch.save(candidate_encoding, cand_encode_path)

    assert (
        candidate_pool_reprs.shape[0] == candidate_encoding.shape[0]
    ), f"cand pool: {candidate_pool_reprs.shape[0]}, cand encoding: {candidate_encoding.shape[0]}"
    test_samples = utils.read_dataset(params["mode"], params["data_path"])
    logger.info("Read %d test samples." % len(test_samples))

    test_data, test_tensor_data = data.process_mention_data(
        test_samples,
        tokenizer,
        params["max_context_length"],
        params["max_cand_length"],
        context_key=params["context_key"],
        silent=params["silent"],
        logger=logger,
        debug=params["debug"],
    )
    test_sampler = SequentialSampler(test_tensor_data)
    test_dataloader = DataLoader(test_tensor_data, sampler=test_sampler, batch_size=params["eval_batch_size"])

    save_results = params.get("save_topk_result")
    new_data = nnquery.get_topk_predictions(
        reranker,
        test_dataloader,
        candidate_pool_ids,
        candidate_encoding,
        params["silent"],
        logger,
        params["top_k"],
        save_results,
    )
    logger.info(f"extracted topk predictions for {len(new_data['label_ids'])} input contexts")
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

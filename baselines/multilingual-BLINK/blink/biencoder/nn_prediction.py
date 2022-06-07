# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from tqdm import tqdm

from blink.biencoder.zeshel_utils import Stats


def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool_ids,
    cand_encode_list,
    silent,
    logger,
    top_k=10,
    save_predictions=False,
):
    reranker.model.eval()
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    label_ids = []  # gold label ids of the context
    topk_candidate_ids = []  # label ids of the nn candidates
    data_idx = []  # index of the samples in the training data

    stats = Stats(top_k)

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        batch_context_input, _, batch_label_ids, batch_data_idx = batch
        scores = reranker.score_candidate(batch_context_input, None, cand_encs=cand_encode_list.to(device))
        values, indicies = scores.topk(top_k)

        for i in range(batch_context_input.size(0)):
            inds = indicies[i]
            batch_topk_cand_ids = [candidate_pool_ids[x] for x in inds]

            pointer = -1
            for j in range(top_k):
                if batch_topk_cand_ids[j] == batch_label_ids[i].item():
                    pointer = j
                    break
            stats.add(pointer)

            # if pointer == -1:
            #     continue

            label_ids.append(batch_label_ids[i].item())
            topk_candidate_ids.append(batch_topk_cand_ids)
            data_idx.append(batch_data_idx[i].item())

    res = Stats(top_k)
    res.extend(stats)
    logger.info(res.output())

    label_ids = torch.LongTensor(label_ids)
    topk_candidate_ids = torch.LongTensor(topk_candidate_ids)
    data_idx = torch.LongTensor(data_idx)
    nn_data = {"label_ids": label_ids, "topk_candidate_ids": topk_candidate_ids, "data_idx": data_idx}

    return nn_data

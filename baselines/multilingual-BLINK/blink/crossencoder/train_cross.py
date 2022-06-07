# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import torch
import random
import time
import numpy as np
import json

from tqdm import tqdm, trange

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

from transformers import get_linear_schedule_with_warmup

from blink.crossencoder.crossencoder import CrossEncoderRanker
import blink.crossencoder.data_process as data

import blink.candidate_ranking.utils as utils
from blink.common.optimizer import get_bert_optimizer
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


def evaluate(reranker, eval_dataloader, device, logger, context_length, silent=True):
    reranker.model.eval()
    if silent:
        iter_ = eval_dataloader
    else:
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

    results = {}

    eval_accuracy = 0.0
    nb_eval_examples = 0

    all_logits = []

    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        context_input = batch[0]
        label_input = batch[1]
        with torch.no_grad():
            eval_loss, logits = reranker(context_input, context_length, label_input)

        logits = logits.detach().cpu().numpy()
        all_logits.extend(logits)

        label_ids = label_input.cpu().numpy()
        batch_eval_accuracy, _ = utils.accuracy(logits, label_ids)

        eval_accuracy += batch_eval_accuracy
        nb_eval_examples += context_input.size(0)

    normalized_eval_accuracy = -1
    if nb_eval_examples > 0:
        normalized_eval_accuracy = eval_accuracy / nb_eval_examples
    if logger:
        logger.info("Eval accuracy: %.5f" % normalized_eval_accuracy)

    results["normalized_accuracy"] = normalized_eval_accuracy
    results["logits"] = all_logits
    return results


def get_optimizer(model, params):
    return get_bert_optimizer(
        [model],
        params["type_optimization"],
        params["learning_rate"],
        fp16=params.get("fp16"),
    )


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params["train_batch_size"]
    grad_acc = params["gradient_accumulation_steps"]
    epochs = params["num_train_epochs"]

    num_train_steps = int(len_train_data / batch_size / grad_acc) * epochs
    num_warmup_steps = int(num_train_steps * params["warmup_proportion"])

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_steps,
    )
    logger.info(" Num optimization steps = %d" % num_train_steps)
    logger.info(" Num warmup steps = %d", num_warmup_steps)
    return scheduler


def main(params):
    model_output_path = params["output_path"]
    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
    logger = utils.get_logger(params["output_path"])

    # Init model
    reranker = CrossEncoderRanker(params)
    tokenizer = reranker.tokenizer
    model = reranker.model

    # use [SEP] token to distingush context and candidate for XLM-RoBERTa
    # use type_token_ids for mBERT
    add_sep_token = model.config.type_vocab_size == 1

    # utils.save_model(model, tokenizer, model_output_path)

    device = reranker.device
    n_gpu = reranker.n_gpu

    if params["gradient_accumulation_steps"] < 1:
        raise ValueError(
            "Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                params["gradient_accumulation_steps"]
            )
        )

    # An effective batch size of `x`, when we are accumulating the gradient accross `y` batches will be achieved by having a batch size of `z = x / y`
    # args.gradient_accumulation_steps = args.gradient_accumulation_steps // n_gpu
    params["train_batch_size"] = params["train_batch_size"] // params["gradient_accumulation_steps"]
    grad_acc_steps = params["gradient_accumulation_steps"]

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if reranker.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)

    max_seq_length = params["max_seq_length"]
    context_length = params["max_context_length"]

    # use the event dictionary to load a mapping from label id to its title and description
    id2title, id2desc = load_event_dict(logger, params)

    """ Prepare train data """
    # load label ids and nn candidate label ids from biencoder predictions (train split)
    mode = "train"
    fname = os.path.join(params["topk_path"], f"{mode}.t7")
    nn_data = torch.load(fname)
    idx2label, idx2nns = {}, {}
    for idx, data_idx in enumerate(nn_data["data_idx"].tolist()):
        idx2label[data_idx] = nn_data["label_ids"][idx]
        idx2nns[data_idx] = nn_data["topk_candidate_ids"][idx]

    # load train contexts
    data_samples = utils.read_dataset(mode, params["data_path"])
    logger.info(f"Read {len(data_samples)} {mode} samples")

    # prepare data for crossencoder training
    context_input, candidate_input, label_input = data.prepare_crossencoder_data(
        tokenizer=tokenizer,
        samples=data_samples,
        idx2label=idx2label,
        idx2nns=idx2nns,
        id2title=id2title,
        id2text=id2desc,
        max_context_length=params["max_context_length"],
        max_cand_length=params["max_cand_length"],
        logger=logger,
        debug=params["debug"],
        topk=params["top_k"],
    )
    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]
    # concatenate context and candidate inputs to pass as a single input to crossencoder
    context_input = modify(
        context_input,
        candidate_input,
        max_seq_length,
        add_sep_token=add_sep_token,
        sep_token_id=tokenizer.sep_token_id,
    )
    train_tensor_data = TensorDataset(context_input, label_input)
    train_sampler = RandomSampler(train_tensor_data)
    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=params["train_batch_size"]
    )

    """ Prepare dev data """
    # load label ids and nn candidate label ids from biencoder predictions (dev split)
    mode = "dev"
    fname = os.path.join(params["topk_path"], f"{mode}.t7")
    nn_data = torch.load(fname)
    idx2label, idx2nns = {}, {}
    for idx, data_idx in enumerate(nn_data["data_idx"].tolist()):
        idx2label[data_idx] = nn_data["label_ids"][idx]
        idx2nns[data_idx] = nn_data["topk_candidate_ids"][idx]

    # load dev contexts
    data_samples = utils.read_dataset(mode, params["data_path"])
    logger.info(f"Read {len(data_samples)} {mode} samples")

    # prepare data for crossencoder eval
    context_input, candidate_input, label_input = data.prepare_crossencoder_data(
        tokenizer=tokenizer,
        samples=data_samples,
        idx2label=idx2label,
        idx2nns=idx2nns,
        id2title=id2title,
        id2text=id2desc,
        max_context_length=params["max_context_length"],
        max_cand_length=params["max_cand_length"],
        logger=logger,
        debug=params["debug"],
        topk=params["top_k"],
    )
    if params["debug"]:
        max_n = 200
        context_input = context_input[:max_n]
        candidate_input = candidate_input[:max_n]
        label_input = label_input[:max_n]
    # concatenate context and candidate inputs to pass as a single input to crossencoder
    context_input = modify(
        context_input,
        candidate_input,
        max_seq_length,
        add_sep_token=add_sep_token,
        sep_token_id=tokenizer.sep_token_id,
    )
    valid_tensor_data = TensorDataset(context_input, label_input)
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=params["eval_batch_size"]
    )

    # evaluate before training
    results = evaluate(
        reranker,
        valid_dataloader,
        device=device,
        logger=logger,
        context_length=context_length,
        silent=params["silent"],
    )

    number_of_samples_per_dataset = {}

    time_start = time.time()

    utils.write_to_file(os.path.join(model_output_path, "training_params.txt"), str(params))

    logger.info("Starting training")
    logger.info("device: {} n_gpu: {}, distributed training: {}".format(device, n_gpu, False))

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

    model.train()

    best_epoch_idx = -1
    best_score = -1

    num_train_epochs = params["num_train_epochs"]

    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        results = None

        if params["silent"]:
            iter_ = train_dataloader
        else:
            iter_ = tqdm(train_dataloader, desc="Batch")

        part = 0
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            context_input = batch[0]
            label_input = batch[1]
            loss, _ = reranker(context_input, context_length, label_input)

            # if n_gpu > 1:
            #     loss = loss.mean() # mean() to average on multi-gpu.

            if grad_acc_steps > 1:
                loss = loss / grad_acc_steps

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"] * grad_acc_steps) == 0:
                logger.info(
                    "Step {} - epoch {} average loss: {}\n".format(
                        step,
                        epoch_idx,
                        tr_loss / (params["print_interval"] * grad_acc_steps),
                    )
                )
                tr_loss = 0

            loss.backward()

            if (step + 1) % grad_acc_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), params["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # if (step + 1) % (params["eval_interval"] * grad_acc_steps) == 0:
            #     logger.info("Evaluation on the development dataset")
            #     evaluate(
            #         reranker,
            #         valid_dataloader,
            #         device=device,
            #         logger=logger,
            #         context_length=context_length,
            #         silent=params["silent"],
            #     )
            #     logger.info("***** Saving fine - tuned model *****")
            #     epoch_output_folder_path = os.path.join(
            #         model_output_path, "epoch_{}_{}".format(epoch_idx, part)
            #     )
            #     part += 1
            #     utils.save_model(model, tokenizer, epoch_output_folder_path)
            #     model.train()
            #     logger.info("\n")

        logger.info("***** Saving fine - tuned model *****")
        epoch_output_folder_path = os.path.join(model_output_path, "epoch_{}".format(epoch_idx))
        utils.save_model(model, tokenizer, epoch_output_folder_path)
        # reranker.save(epoch_output_folder_path)

        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            reranker,
            valid_dataloader,
            device=device,
            logger=logger,
            context_length=context_length,
            silent=params["silent"],
        )

        ls = [best_score, results["normalized_accuracy"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmax(ls)]
        best_epoch_idx = li[np.argmax(ls)]
        logger.info("\n")

    execution_time = (time.time() - time_start) / 60
    utils.write_to_file(
        os.path.join(model_output_path, "training_time.txt"),
        "The training took {} minutes\n".format(execution_time),
    )
    logger.info("The training took {} minutes\n".format(execution_time))

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["path_to_model"] = os.path.join(model_output_path, "epoch_{}".format(best_epoch_idx))


if __name__ == "__main__":
    parser = BlinkParser(add_model_args=True)
    parser.add_training_args()
    parser.add_eval_args()

    # args = argparse.Namespace(**params)
    args = parser.parse_args()
    print(args)

    params = args.__dict__
    main(params)

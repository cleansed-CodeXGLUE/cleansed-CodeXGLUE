# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import
import os
import sys
import pickle
import torch
import json
import random
import logging
import argparse
import numpy as np
from io import open
from itertools import cycle
import torch.nn as nn
from tqdm import tqdm, trange
from bleu import _bleu
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (WEIGHTS_NAME, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from model_lstm_enc import LSTMEncoder, Seq2Seq
from torch.optim import AdamW

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer)}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


# def read_examples(filename):
#     """Read examples from filename."""
#     examples=[]
#     with open(filename,encoding="utf-8") as f:
#         for idx,js in enumerate(json.load(f)):
#             source=' '.join(js['old_comment_tokens'])
#             target=' '.join(js['new_comment_tokens'])
#             examples.append(
#                 Example(
#                         idx = idx,
#                         source=source,
#                         target=target,
#                         )
#             )
#     return examples
def read_examples(filename):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
    return examples


def save_as_json(data, save_path):
    data_json = json.dumps(data, indent=4)
    with open(save_path, "w") as file:
        file.write(data_json)


def load_from_json(load_path, by_line=False):
    if by_line:
        data = []
        for line in open(load_path, "r", encoding="utf8"):
            data.append(json.loads(line))
        return data
    with open(load_path, "r") as f:
        data = json.load(f)
    return data


class InputFeatures(object):
    """A single training/test features for a example."""

    def __init__(self,
                 example_id,
                 source_ids,
                 target_ids,
                 source_mask,
                 target_mask,

                 ):
        self.example_id = example_id
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.source_mask = source_mask
        self.target_mask = target_mask


def convert_examples_to_features(examples, tokenizer, args, stage=None):
    features = []
    for example_index, example in enumerate(examples):
        # source
        source_tokens = tokenizer.tokenize(example.source)[
            :args.max_source_length - 2]
        source_tokens = [tokenizer.cls_token] + \
            source_tokens + [tokenizer.sep_token]
        source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
        source_mask = [1] * (len(source_tokens))
        padding_length = args.max_source_length - len(source_ids)
        source_ids += [tokenizer.pad_token_id] * padding_length
        source_mask += [0] * padding_length

        # target
        if stage == "test":
            target_tokens = tokenizer.tokenize("None")
        else:
            target_tokens = tokenizer.tokenize(example.target)[
                :args.max_target_length - 2]
        target_tokens = [tokenizer.cls_token] + \
            target_tokens + [tokenizer.sep_token]
        target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
        target_mask = [1] * len(target_ids)
        padding_length = args.max_target_length - len(target_ids)
        target_ids += [tokenizer.pad_token_id] * padding_length
        target_mask += [0] * padding_length

        if example_index < 5:
            if stage == 'train':
                logger.info("*** Example ***")
                logger.info("idx: {}".format(example.mapped_idx))

                logger.info("source_tokens: {}".format(
                    [x.replace('\u0120', '_') for x in source_tokens]))
                logger.info("source_ids: {}".format(
                    ' '.join(map(str, source_ids))))
                logger.info("source_mask: {}".format(
                    ' '.join(map(str, source_mask))))

                logger.info("target_tokens: {}".format(
                    [x.replace('\u0120', '_') for x in target_tokens]))
                logger.info("target_ids: {}".format(
                    ' '.join(map(str, target_ids))))
                logger.info("target_mask: {}".format(
                    ' '.join(map(str, target_mask))))

        features.append(
            InputFeatures(
                example_index,
                source_ids,
                target_ids,
                source_mask,
                target_mask,
            )
        )
    return features


def _truncate_seq_pair(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type: e.g. roberta")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--tokenizer_name", default="", required=True,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. (source and target files).")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. (source and target files).")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--lstm_hidden_size', type=int, default=768,
                    help="Hidden size of the LSTM encoder.")
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help="Number of layers in the LSTM encoder.")
    parser.add_argument('--lstm_dropout', type=float, default=0.1,
                        help="Dropout rate for the LSTM encoder.")
    # print arguments
    args = parser.parse_args()
    logger.info(args)

        # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name, do_lower_case=args.do_lower_case)

    # budild model
    encoder = LSTMEncoder(
        vocab_size=config.vocab_size,
        embedding_dim=config.hidden_size,
        hidden_size=int(args.lstm_hidden_size/2),
        num_layers=args.lstm_num_layers,
        dropout=args.lstm_dropout
    )
    decoder_layer = nn.TransformerDecoderLayer(
        d_model=args.lstm_hidden_size, nhead=config.num_attention_heads)
    decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
                    beam_size=args.beam_size, max_length=args.max_target_length,
                    sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
    

    if args.load_model_path is not None:
        logger.info("reload model from {}".format(args.load_model_path))
        model.load_state_dict(torch.load(args.load_model_path))

    model.to(device)

    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # multi-gpu training
        model = torch.nn.DataParallel(model)

    #####
    inf_res = []

    dev_dataset = {}

    eval_examples = read_examples(args.dev_filename)
    eval_features = convert_examples_to_features(
        eval_examples, tokenizer, args, stage='dev')
    all_source_ids = torch.tensor(
        [f.source_ids for f in eval_features], dtype=torch.long)
    all_source_mask = torch.tensor(
        [f.source_mask for f in eval_features], dtype=torch.long)
    all_target_ids = torch.tensor(
        [f.target_ids for f in eval_features], dtype=torch.long)
    all_target_mask = torch.tensor(
        [f.target_mask for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(
        all_source_ids, all_source_mask, all_target_ids, all_target_mask)
    dev_dataset['dev_loss'] = eval_examples, eval_data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, shuffle=False)

    eval_dataloader_bs1 = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=1, shuffle=False)

    logger.info("\n***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    # Start Evaling model
    model.eval()
    p = []
    eval_loss, tokens_num = 0, 0
    print(len(eval_dataloader))
    for i, batch in enumerate(tqdm(eval_dataloader_bs1, desc="get-loss")):
        res = {}
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask, target_ids, target_mask = batch

        with torch.no_grad():
            _, loss, num = model(source_ids=source_ids, source_mask=source_mask,
                                 target_ids=target_ids, target_mask=target_mask)

        eval_loss = loss.sum().item()
        tokens_num += num.sum().item()

        assert i == eval_examples[i].idx
        res["source"] = eval_examples[i].source
        res["target"] = eval_examples[i].target
        res["loss"] = eval_loss
        inf_res.append(res)

    for i, batch in enumerate(tqdm(eval_dataloader, desc="get-inf")):
        res = {}
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask, target_ids, target_mask = batch

        with torch.no_grad():
            preds = model(source_ids=source_ids, source_mask=source_mask)
            for pred in preds:
                t = pred[0].cpu().numpy()
                t = list(t)
                if 0 in t:
                    t = t[:t.index(0)]
                text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
                p.append(text)

    save_as_json(inf_res, os.path.join(args.output_dir, "dev_loss.json"))
    with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
            os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
        for ref, gold in zip(p, eval_examples):
            f.write(ref + '\n')
            f1.write(gold.target + '\n')

    # Pring loss of dev dataset

    # if 'dev_bleu' in dev_dataset:
    #     eval_examples, eval_data = dev_dataset['dev_bleu']
    # else:
    #     eval_examples = read_examples(args.dev_filename)
    #     eval_examples = random.sample(eval_examples, min(1000, len(eval_examples)))
    #     eval_features = convert_examples_to_features(eval_examples, tokenizer, args, stage='test')
    #     all_source_ids = torch.tensor([f.source_ids for f in eval_features], dtype=torch.long)
    #     all_source_mask = torch.tensor([f.source_mask for f in eval_features], dtype=torch.long)
    #     eval_data = TensorDataset(all_source_ids, all_source_mask)
    #     dev_dataset['dev_bleu'] = eval_examples, eval_data

    # eval_sampler = SequentialSampler(eval_data)
    # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # model.eval()
    # p = []
    # for batch in eval_dataloader:
    #     batch = tuple(t.to(device) for t in batch)
    #     source_ids, source_mask = batch
    #     with torch.no_grad():
    #         preds = model(source_ids=source_ids, source_mask=source_mask)
    #         for pred in preds:
    #             t = pred[0].cpu().numpy()
    #             t = list(t)
    #             if 0 in t:
    #                 t = t[:t.index(0)]
    #             text = tokenizer.decode(t, clean_up_tokenization_spaces=False)
    #             p.append(text)
    # model.train()
    # predictions = []
    # accs = []
    # with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
    #         os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
    #     for ref, gold in zip(p, eval_examples):
    #         predictions.append(str(gold.mapped_idx) + '\t' + ref)
    #         f.write(ref + '\n')
    #         f1.write(gold.target + '\n')
    #         accs.append(ref == gold.target)

    # dev_bleu = round(
    #     _bleu(os.path.join(args.output_dir, "dev.gold"), os.path.join(args.output_dir, "dev.output")), 2)
    # logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
    # logger.info("  %s = %s " % ("xMatch", str(round(np.mean(accs) * 100, 4))))
    # logger.info("  " + "*" * 20)


if __name__ == "__main__":
    main()
'''
python inference.py --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --config_name roberta-base --tokenizer_name roberta-base --train_filename ../data/small/train.buggy-fixed.buggy,../data/small/train.buggy-fixed.fixed --dev_filename ../data/small/valid.buggy-fixed.buggy,../data/small/valid.buggy-fixed.fixed --load_model_path ./saved_models/checkpoint-best-bleu/pytorch_model.bin --output_dir ./saved_models --max_source_length 256 --max_target_length 256 --beam_size 5 --train_batch_size 16 --eval_batch_size 1 --learning_rate 5e-5 --train_steps 100000 --eval_steps 5000
'''
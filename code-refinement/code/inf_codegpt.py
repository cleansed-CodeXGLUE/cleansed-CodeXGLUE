
import argparse
import logging
from os import name
import os
import random
from numpy import source
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model, get_linear_schedule_with_warmup
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.optim import AdamW
from bleu import _bleu
import numpy as np
from itertools import cycle
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def read_examples(filename, data_type='train'):
    """Read examples from filename."""
    source_list = []
    target_list = []
    idx_list = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            if data_type == 'train':
                source_list.append(
                    f"<buggy-s>{line1.strip()}<buggy-e><fixed-s>{line2.strip()}<fixed-e>")
            else:
                source_list.append(
                    f"<buggy-s>{line1.strip()}<buggy-e><fixed-s>")
                target_list.append(line2.strip())

            idx_list.append(idx)
            idx += 1
    if data_type == 'train':
        return idx_list, source_list
    else:
        return idx_list, source_list, target_list


def set_seed(args):
    """set random seed."""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument("--do_eval", action='store_true')
    parser.add_argument("--train_filename", default=None, type=str)
    parser.add_argument("--dev_filename", default=None, type=str)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--load_model_path", default=None, type=str)
    parser.add_argument("--max_source_length", default=256, type=int)
    parser.add_argument("--max_target_length", default=256, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=8, type=int)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument("--train_steps", default=-1, type=int)
    parser.add_argument("--eval_steps", default=-1, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--beam_size", default=10, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    logger.info(args)

    device = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    args.device = device
    # Set seed
    set_seed(args)
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)

    # model_name = "microsoft/CodeGPT-small-java-adaptedGPT2"
    # model = GPT2LMHeadModel.from_pretrained(model_name)
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    tokenizer = GPT2Tokenizer.from_pretrained(
        "microsoft/CodeGPT-small-java-adaptedGPT2")
    model = GPT2LMHeadModel.from_pretrained(
        "microsoft/CodeGPT-small-java-adaptedGPT2")
    special_tokens_dict = {'additional_special_tokens': [
        '<buggy-s>', '<buggy-e>', '<fixed-s>', '<fixed-e>']}
    tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(torch.load(args.load_model_path))
    logger.info("reload model from {}".format(args.load_model_path))

    # GET EVAL DATA
    eval_idx_list, eval_source_list, eval_target_list = read_examples(
        args.dev_filename, data_type='test')

    eval_source_res = tokenizer(
        eval_source_list, return_tensors="pt", padding="max_length", truncation=True, max_length=args.max_target_length)

    eval_data = TensorDataset(
        eval_source_res.input_ids, eval_source_res.attention_mask)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, shuffle=False)

    ##

    logger.info("\n***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_idx_list))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.to(device)
    model.eval()
    p = []
    for batch in tqdm(eval_dataloader, desc="get-inf"):
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask = batch
        with torch.no_grad():
            generated_ids = model.generate(input_ids=source_ids, attention_mask=source_mask, eos_token_id=tokenizer.convert_tokens_to_ids('<fixed-e>'),
                                           pad_token_id=model.config.eos_token_id, num_return_sequences=1, max_length=args.max_target_length*3+10)

            for cnt, generated_id in enumerate(generated_ids):
                # find <fixed-s> and <fixed-e> to get the code
                fixed_s_index = (generated_id == tokenizer.convert_tokens_to_ids(
                    '<fixed-s>')).nonzero()[0].item()
                fixed_e_index = (generated_id == tokenizer.convert_tokens_to_ids(
                    '<fixed-e>')).nonzero()
                if fixed_e_index.numel() == 0:
                    generated_id = generated_id[fixed_s_index + 1:]
                else:
                    fixed_e_index[0].item()
                    generated_id = generated_id[fixed_s_index +
                                                1: fixed_e_index]
                generated_code = tokenizer.decode(
                    generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                p.append(generated_code)
    with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
            os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
        for ref, gold in zip(p, eval_target_list):
            f.write(ref + '\n')
            f1.write(gold + '\n')

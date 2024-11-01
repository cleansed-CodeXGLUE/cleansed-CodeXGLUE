
import argparse
import logging
from os import name
import os
import random
from numpy import source
from tqdm import tqdm
from transformers import RobertaTokenizer,T5ForConditionalGeneration, get_linear_schedule_with_warmup
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


def read_examples(filename):
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
            source_list.append(line1.strip())
            target_list.append(line2.strip())
            idx_list.append(idx)
            idx += 1
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
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--train_filename", default=None, type=str)
    parser.add_argument("--dev_filename", default=None, type=str)
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
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # model_name = "microsoft/CodeGPT-small-java-adaptedGPT2"
    # model = GPT2LMHeadModel.from_pretrained(model_name)
    # tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
    # model.load_state_dict(torch.load("./test/checkpoint-last/pytorch_model.bin"))
    print(model)
    idx_list, source_list, target_list = read_examples(args.train_filename)

    source_res = tokenizer(source_list, return_tensors="pt", padding="max_length",
                           max_length=args.max_source_length, truncation=True)
    target_res = tokenizer(target_list, return_tensors="pt", padding="max_length",
                           max_length=args.max_target_length, truncation=True)
    print(source_res.input_ids.shape)

    train_data = TensorDataset(source_res.input_ids, source_res.attention_mask,
                               target_res.input_ids, target_res.attention_mask)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=args.train_batch_size // args.gradient_accumulation_steps)

    # GET EVAL DATA
    eval_idx_list, eval_source_list, eval_target_list = read_examples(
        args.dev_filename)

    indexs = list(range(len(eval_idx_list)))
    random.shuffle(indexs)
    indexs = indexs[:1000]
    eval_idx_list = [eval_idx_list[i] for i in indexs]
    eval_source_list = [eval_source_list[i] for i in indexs]
    eval_target_list = [eval_target_list[i] for i in indexs]

    eval_source_res = tokenizer(eval_source_list, return_tensors="pt", padding="max_length",
                                max_length=args.max_source_length, truncation=True)

    eval_data = TensorDataset(
        eval_source_res.input_ids, eval_source_res.attention_mask)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    ##

    num_train_optimization_steps = args.train_steps

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(idx_list))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num epoch = %d", num_train_optimization_steps *
                args.train_batch_size // len(idx_list))
    model.to(device)
    model.train()
    dev_dataset = {}
    nb_tr_examples, nb_tr_steps, tr_loss, global_step, best_bleu, best_loss, best_acc = 0, 0, 0, 0, 0, 1e6, 0
    bar = range(num_train_optimization_steps)
    train_dataloader = cycle(train_dataloader)
    eval_flag = True
    for step in bar:
        batch = next(train_dataloader)
        batch = tuple(t.to(device) for t in batch)
        source_ids, source_mask, target_ids, target_mask = batch

        output = model(input_ids=source_ids,
                       attention_mask=source_mask, labels=target_ids)
        loss = output.loss
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        tr_loss += loss.item()
        train_loss = round(
            tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
        if (global_step + 1) % 100 == 0:
            logger.info("  step {} loss {}".format(
                global_step + 1, train_loss))
        nb_tr_examples += source_ids.size(0)
        nb_tr_steps += 1
        loss.backward()

        if (nb_tr_steps + 1) % args.gradient_accumulation_steps == 0:
            # Update parameters
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            eval_flag = True

        if args.do_eval and ((global_step + 1) % args.eval_steps == 0):
            last_output_dir = os.path.join(
                args.output_dir, 'checkpoint-last')
            if not os.path.exists(last_output_dir):
                os.makedirs(last_output_dir)
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                last_output_dir, "pytorch_model.bin")
            torch.save(model_to_save.state_dict(), output_model_file)

            model.eval()
            p = []
            for batch in eval_dataloader:
                batch = tuple(t.to(device) for t in batch)
                source_ids, source_mask = batch
                with torch.no_grad():
                    generated_ids = model.generate(input_ids=source_ids, attention_mask=source_mask, #pad_token_id=model.config.eos_token_id,
                                                    max_length=args.max_target_length+10, num_beams=args.beam_size, early_stopping=True)
                    for generated_id in generated_ids:
                        generated_code = tokenizer.decode(
                            generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        p.append(generated_code)
            model.train()
            predictions = []
            accs = []
            with open(os.path.join(args.output_dir, "dev.output"), 'w') as f, open(
                    os.path.join(args.output_dir, "dev.gold"), 'w') as f1:
                cnt = 0
                for ref, gold in zip(p, eval_target_list):
                    predictions.append(str(eval_idx_list[cnt]) + '\t' + ref)
                    f.write(ref + '\n')
                    f1.write(gold + '\n')
                    accs.append(ref == gold)
                    cnt += 1
            dev_bleu = round(_bleu(os.path.join(args.output_dir, "dev.gold"), os.path.join(
                args.output_dir, "dev.output")), 2)
            logger.info("  %s = %s " % ("bleu-4", str(dev_bleu)))
            logger.info("  %s = %s " %
                        ("xMatch", str(round(np.mean(accs) * 100, 4))))
            dev_accs = round(np.mean(accs) * 100, 4)
            logger.info("  " + "*" * 20)
            if dev_bleu > best_bleu:
                logger.info("  Best bleu:%s", dev_bleu)
                logger.info("  " + "*" * 20)
                best_bleu = dev_bleu
                # Save best checkpoint for best bleu
                output_dir = os.path.join(
                    args.output_dir, 'checkpoint-best-bleu')
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = model.module if hasattr(
                    model, 'module') else model  # Only save the model it-self
                output_model_file = os.path.join(
                    output_dir, "pytorch_model.bin")
                torch.save(model_to_save.state_dict(), output_model_file)
            step_output_dir = os.path.join(
                args.output_dir, 'checkpoint-steps')
            if not os.path.exists(step_output_dir):
                os.makedirs(step_output_dir)
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                step_output_dir, f"pytorch_model-{global_step}.bin")
            torch.save(model_to_save.state_dict(), output_model_file)
            with open(os.path.join(step_output_dir, f"step-{global_step}.txt"), 'w') as f:
                f.write(
                    f"bleu:{dev_bleu}\naccs:{str(round(np.mean(accs) * 100, 4))}\n")

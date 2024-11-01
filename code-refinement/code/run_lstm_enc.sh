#!/bin/sh

# train small dirty ; val-on dirty
# python run_lstm_enc.py --do_train --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --config_name roberta-base --tokenizer_name roberta-base --train_filename ../data/small/train.buggy-fixed.buggy,../data/small/train.buggy-fixed.fixed --dev_filename ../data/small/valid.buggy-fixed.buggy,../data/small/valid.buggy-fixed.fixed --output_dir ./lstm-small-dirty --max_source_length 256 --max_target_length 256 --beam_size 5 --train_batch_size 16 --eval_batch_size 16 --learning_rate 5e-5 --train_steps 100000 --eval_steps 5000

# train small clean ; val-on clean
python run_lstm_enc.py --do_train --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --config_name roberta-base --tokenizer_name roberta-base --train_filename ../data/small/train-cleaned/train.buggy,../data/small/train-cleaned/train.fixed --dev_filename ../data/small/valid-cleaned/valid.buggy,../data/small/valid-cleaned/valid.fixed --output_dir ./lstm-small-clean --max_source_length 256 --max_target_length 256 --beam_size 5 --train_batch_size 16 --eval_batch_size 16 --learning_rate 5e-5 --train_steps 100000 --eval_steps 5000


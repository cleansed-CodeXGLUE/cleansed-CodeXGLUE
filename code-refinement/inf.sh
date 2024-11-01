#!/bin/sh

# codebert(dirty) small valid(clean)
# python ./code/inf_codebert.py --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --config_name roberta-base --tokenizer_name roberta-base --dev_filename ./data/small/valid-cleaned/valid.buggy,./data/small/valid-cleaned/valid.fixed --load_model_path ./code/codebert-small-dirty/checkpoint-last/pytorch_model.bin --output_dir ./data/small/valid-cleaned/inf-codebert --max_source_length 256 --max_target_length 256 --beam_size 5 --eval_batch_size 16

# codebert(dirty) small valid(dirty)
# python ./code/inf_codebert.py --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --config_name roberta-base --tokenizer_name roberta-base --dev_filename ./data/small/valid-org/valid.buggy,./data/small/valid-org/valid.fixed --load_model_path ./code/codebert-small-dirty/checkpoint-last/pytorch_model.bin --output_dir ./data/small/valid-org/inf-codebert --max_source_length 256 --max_target_length 256 --beam_size 5 --eval_batch_size 16

# # codebert(clean) small valid(dirty)
python ./code/inf_codebert.py --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --config_name roberta-base --tokenizer_name roberta-base --dev_filename ./data/small/valid-org/valid.buggy,./data/small/valid-org/valid.fixed --load_model_path ./code/codebert-small-clean/checkpoint-best-xMatch/pytorch_model.bin --output_dir ./data/small/valid-org/inf-codebert-clean-bxmatch --max_source_length 256 --max_target_length 256 --beam_size 5 --eval_batch_size 16

# # codebert(clean) small valid(clean)
# python ./code/inf_codebert.py --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --config_name roberta-base --tokenizer_name roberta-base --dev_filename ./data/small/valid-cleaned/valid.buggy,./data/small/valid-cleaned/valid.fixed --load_model_path ./code/codebert-small-clean/checkpoint-best-xMatch/pytorch_model.bin --output_dir ./data/small/valid-cleaned/inf-codebert-clean-bxmatch --max_source_length 256 --max_target_length 256 --beam_size 5 --eval_batch_size 16


# lstm(dirty) small valid(clean)
# python ./code/inf_lstm.py --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --config_name roberta-base --tokenizer_name roberta-base --dev_filename ./data/small/valid-cleaned/valid.buggy,./data/small/valid-cleaned/valid.fixed --load_model_path ./code/lstm-small-dirty/checkpoint-last/pytorch_model.bin --output_dir ./data/small/valid-clean/inf-lstm --max_source_length 256 --max_target_length 256 --beam_size 5 --eval_batch_size 16

# lstm(dirty) small valid(dirty)
# python ./code/inf_lstm.py --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --config_name roberta-base --tokenizer_name roberta-base --dev_filename ./data/small/valid-org/valid.buggy,./data/small/valid-org/valid.fixed --load_model_path ./code/lstm-small-dirty/checkpoint-last/pytorch_model.bin --output_dir ./data/small/valid-org/inf-lstm --max_source_length 256 --max_target_length 256 --beam_size 5 --eval_batch_size 16

# lstm(clean) small valid(dirty)
# python ./code/inf_lstm.py --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --config_name roberta-base --tokenizer_name roberta-base --dev_filename ./data/small/valid-org/valid.buggy,./data/small/valid-org/valid.fixed --load_model_path ./code/lstm-small-clean/checkpoint-last/pytorch_model.bin --output_dir ./data/small/valid-org/inf-lstm --max_source_length 256 --max_target_length 256 --beam_size 5 --eval_batch_size 16

# lstm(clean) small valid(clean)
# python ./code/inf_lstm.py --do_eval --model_type roberta --model_name_or_path microsoft/codebert-base --config_name roberta-base --tokenizer_name roberta-base --dev_filename ./data/small/valid-cleaned/valid.buggy,./data/small/valid-cleaned/valid.fixed --load_model_path ./code/lstm-small-clean/checkpoint-last/pytorch_model.bin --output_dir ./data/small/valid-cleaned/inf-lstm --max_source_length 256 --max_target_length 256 --beam_size 5 --eval_batch_size 16

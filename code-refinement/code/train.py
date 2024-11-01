import os
import subprocess
from unittest import result


def train(model, data_size, train_on, val_on):
    _model = {
        "codebert": "run.py",
        "lstm": "run_lstm_enc.py",
        "codegpt": "run_codegpt.py",
        "codet5": "run_codet5.py"
    }

    _train_filename = {
        "dirty": f"../data/{data_size}/train.buggy-fixed.buggy,../data/{data_size}/train.buggy-fixed.fixed",
        "clean": f"../data/{data_size}/train-cleaned/train.buggy,../data/{data_size}/train-cleaned/train.fixed",
        "control": f"../data/{data_size}/train-control/train.buggy,../data/{data_size}/train-control/train.fixed"
    }

    _dev_filename = {
        "dirty": f"../data/{data_size}/valid.buggy-fixed.buggy,../data/{data_size}/valid.buggy-fixed.fixed",
        "clean": f"../data/{data_size}/valid-cleaned/valid.buggy,../data/{data_size}/valid-cleaned/valid.fixed",
    }

    command = [
        "python", _model[model],
        "--do_train",
        "--do_eval",
        # "--model_type", "roberta",
        # "--model_name_or_path", "microsoft/codebert-base",
        # "--config_name", "roberta-base",
        # "--tokenizer_name", "roberta-base",
        "--train_filename", _train_filename[train_on],
        "--dev_filename", _dev_filename[val_on],
        "--output_dir", f"../../autodl-tmp/{model}-{data_size}-{train_on}",
        "--max_source_length", "256",
        "--max_target_length", "256",
        "--beam_size", "5",
        "--train_batch_size", "8" if model == "codegpt" else "16",
        "--eval_batch_size", "16",
        "--learning_rate", "5e-5",
        "--train_steps", "200000" if model == "codegpt" else "100000",
        "--eval_steps", "10000" if model == "codegpt" else "5000",
    ]
    info = f"\n\n==================\nTraining {model}-{data_size}-{train_on} \non " + \
        f"{_train_filename[train_on]}\nwith train_val on {_dev_filename[val_on]}" + \
        f"\nsave as ../../autodl-tmp/{model}-{data_size}-{train_on}\n======================\n\n"
    print(info)
    subprocess.run(command, capture_output=False, text=True)

    return f"../../autodl-tmp/{model}-{data_size}-{train_on}"


def inf(model, inf_on, data_size, train_on):
    _inf_model = {
        "codebert": "inf_codebert.py",
        "lstm": "inf_lstm.py",
        "codegpt": "inf_codegpt.py",
        "codet5": "inf_codet5.py"
    }
    _output_dir = {
        "dirty": f"../data/{data_size}/valid-org/inf-{model}-{train_on}",
        "clean": f"../data/{data_size}/valid-cleaned/inf-{model}-{train_on}",
    }

    _inf_filename = {
        "dirty": f"../data/{data_size}/valid-org/valid.buggy,../data/{data_size}/valid-org/valid.fixed",
        "clean": f"../data/{data_size}/valid-cleaned/valid.buggy,../data/{data_size}/valid-cleaned/valid.fixed"
    }
    
    # fix old typo bug
    train_on = "dirty" if (model == "lstm" and train_on == "control") else train_on
    train_on = "control" if (model == "lstm" and train_on == "dirty") else train_on
    
    command = [
        "python", _inf_model[model],
        "--do_eval",
        # "--model_type", "roberta",
        # "--model_name_or_path", "microsoft/codebert-base",
        # "--config_name", "roberta-base",
        # "--tokenizer_name", "roberta-base",
        "--dev_filename", _inf_filename[inf_on],
        "--load_model_path", f"../../autodl-tmp/{model}-{data_size}-{train_on}/checkpoint-last/pytorch_model.bin",
        "--output_dir", _output_dir[inf_on],
        "--max_source_length", "256",
        "--max_target_length", "256",
        "--beam_size", "5",
        "--eval_batch_size", "16",
    ]

    info = f"\n\n==================\nInference {model}-{data_size}-{train_on} \non " + \
        f"{_inf_filename[inf_on]}\n" + \
        f"save as {_output_dir[inf_on]}\n======================\n\n"

    subprocess.run(command, capture_output=False, text=True)

    return _output_dir[inf_on]


def inf_defects4j(model, inf_on, data_size, train_on):
    _inf_model = {
        "codebert": "inf_codebert.py",
        "lstm": "inf_lstm.py",
        "codegpt": "inf_codegpt.py",
        "codet5": "inf_codet5.py"
    }
    _output_dir = {
        "dirty": f"../data/{data_size}/valid-org/inf-defects4j-{model}-{train_on}",
        "clean": f"../data/{data_size}/valid-cleaned/inf-defects4j-{model}-{train_on}",
    }

    _inf_filename = f"../data/defects4j/valid.buggy,../data/defects4j/valid.fixed"

    command = [
        "python", _inf_model[model],
        "--do_eval",
        "--model_type", "roberta",
        "--model_name_or_path", "microsoft/codebert-base",
        "--config_name", "roberta-base",
        "--tokenizer_name", "roberta-base",
        "--dev_filename", _inf_filename,
        "--load_model_path", f"../../autodl-tmp/{model}-{data_size}-{train_on}/checkpoint-last/pytorch_model.bin",
        "--output_dir", _output_dir[inf_on],
        "--max_source_length", "256",
        "--max_target_length", "256",
        "--beam_size", "5",
        "--eval_batch_size", "16",
    ]

    info = f"\n\n==================\nInference {model}-{data_size}-{train_on} \non " + \
        f"{_inf_filename}\n" + \
        f"save as {_output_dir[inf_on]}\n======================\n\n"
    print(info)

    subprocess.run(command, capture_output=False, text=True)

    return _output_dir[inf_on]


def eva(data_size, model, train_on, inf_on):
    _eva_filename = {
        "dirty": {"ref": f"../data/{data_size}/valid-org/inf-{model}-{train_on}/dev.gold",
                  "pre": f"../data/{data_size}/valid-org/inf-{model}-{train_on}/dev.output"},
        "clean": {"ref": f"../data/{data_size}/valid-cleaned/inf-{model}-{train_on}/dev.gold",
                  "pre": f"../data/{data_size}/valid-cleaned/inf-{model}-{train_on}/dev.output"},
    }

    command1 = [
        "python", "../evaluator/evaluator.py",
        "-ref", _eva_filename[inf_on]["ref"],
        "-pre", _eva_filename[inf_on]["pre"]
    ]

    command2 = [
        "python", "../evaluator/CodeBLEU/calc_code_bleu.py",
        "--refs", _eva_filename[inf_on]["ref"],
        "--hyp", _eva_filename[inf_on]["pre"],
        "--lang", "java",
        "--params", "0.25,0.25,0.25,0.25"
    ]

    result1 = subprocess.run(command1, capture_output=True, text=True)
    print(result1.stdout)
    print(result1.stderr)

    result2 = subprocess.run(command2, capture_output=True, text=True)
    print(result2.stdout)
    print(result2.stderr)

    save_eva_dir = os.path.dirname(_eva_filename[inf_on]["pre"])
    print(save_eva_dir)
    with open(f"{save_eva_dir}/eva.txt", "w") as file:
        file.write(result1.stdout + result2.stdout)
        
def eva_defects4j(data_size, model, train_on, inf_on):
    _eva_filename = {
        "dirty": {"ref": f"../data/{data_size}/valid-org/inf-defects4j-{model}-{train_on}/dev.gold",
                  "pre": f"../data/{data_size}/valid-org/inf-defects4j-{model}-{train_on}/dev.output"},
        "clean": {"ref": f"../data/{data_size}/valid-cleaned/inf-defects4j-{model}-{train_on}/dev.gold",
                  "pre": f"../data/{data_size}/valid-cleaned/inf-defects4j-{model}-{train_on}/dev.output"},
    }

    command1 = [
        "python", "../evaluator/evaluator.py",
        "-ref", _eva_filename[inf_on]["ref"],
        "-pre", _eva_filename[inf_on]["pre"]
    ]

    command2 = [
        "python", "../evaluator/CodeBLEU/calc_code_bleu.py",
        "--refs", _eva_filename[inf_on]["ref"],
        "--hyp", _eva_filename[inf_on]["pre"],
        "--lang", "java",
        "--params", "0.25,0.25,0.25,0.25"
    ]

    result1 = subprocess.run(command1, capture_output=True, text=True)
    print(result1.stdout)
    print(result1.stderr)

    result2 = subprocess.run(command2, capture_output=True, text=True)
    print(result2.stdout)
    print(result2.stderr)

    save_eva_dir = os.path.dirname(_eva_filename[inf_on]["pre"])
    print(save_eva_dir)
    with open(f"{save_eva_dir}/eva.txt", "w") as file:
        file.write(result1.stdout + result2.stdout)


if __name__ == "__main__":

    #########################################################
    model = "lstm"
    data_size = "small"
    train_on = "control"
    inf_on_list = ["dirty"]
    #########################################################

    # save1 = train(model, data_size, train_on, val_on="dirty")
    save2 = []
    for inf_on in inf_on_list:
        # save_temp = inf(model, inf_on, data_size, train_on)
        save_defects4j = inf_defects4j(model, inf_on, data_size, train_on)
        # save2.append(save_temp)
        # eva(data_size, model, train_on, inf_on)
        eva_defects4j(data_size, model, train_on, inf_on)


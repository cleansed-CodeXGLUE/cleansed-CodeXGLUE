import os
import random


random.seed(1216)
data_size = "small"

org_train_buggy_path = f"../data/{data_size}/train.buggy-fixed.buggy"
org_train_fixed_path = f"../data/{data_size}/train.buggy-fixed.fixed"

# read line
with open(org_train_buggy_path, "r") as f:
    buggy_lines = f.readlines()

with open(org_train_fixed_path, "r") as f:
    fixed_lines = f.readlines()


print(f"org num: {len(buggy_lines)}")
assert len(buggy_lines) == len(fixed_lines)

index = list(range(len(buggy_lines)))
random.shuffle(index)

# cleaned data
clean_train_data_path = f"../data/{data_size}/train-cleaned/train.buggy"

with open(clean_train_data_path, "r") as f:
    cleaned_lines = f.readlines()
print(f"cleaned num: {len(cleaned_lines)}")

index = index[:len(cleaned_lines)]
assert len(index) == len(cleaned_lines)

save_path = f"../data/{data_size}/train-control"
os.makedirs(save_path)
save_buggy_path = f"{save_path}/train.buggy"
save_fixed_path = f"{save_path}/train.fixed"

with open(save_buggy_path, "w") as f:
    for idx in index:
        f.write(buggy_lines[idx])

with open(save_fixed_path, "w") as f:
    for idx in index:
        f.write(fixed_lines[idx])

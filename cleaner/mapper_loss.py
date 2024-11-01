import pandas as pd
from tqdm import tqdm
from utils.tools import *

data_root = '../code-refinement/data/'
raw_root = '//home/shweng/raw/'
data_size = 'small'
data_type = 'valid'
have_loss = False
cur_datas_path = None
if have_loss:
    cur_datas_path = f"{data_root}inference/dev_res_sorted.json"
mapped_cur_datas_save_path = f"{data_root}inference/dev_res_sorted_mapped.json"
mapped_datas_save_path = f"{data_root}{data_size}/valid_mapped.json"


raw_type = "50" if data_size == "small" else "50-100"
# load file
with open(f"{data_root}{data_size}/{data_type}.buggy-fixed.buggy", 'r') as f:
    buggy = f.readlines()
with open(f"{data_root}{data_size}/{data_type}.buggy-fixed.fixed", 'r') as f:
    fixed = f.readlines()
assert len(buggy) == len(fixed)

with open(f"{raw_root}datasets/{raw_type}/buggy_all.txt", 'r') as f:
    all_buggy = f.readlines()
with open(f"{raw_root}datasets/{raw_type}/fixed_all.txt", 'r') as f:
    all_fixed = f.readlines()

# find the idx of buggy in all_buggy
idx_buggy = []
for i, b in enumerate(buggy):
    idx_buggy.append(all_buggy.index(b))

# find the idx of fixed in all_fixed
idx_fixed = []
for i, f in enumerate(fixed):
    idx_fixed.append(all_fixed.index(f))

# read_csv
mapper = pd.read_csv(f"{raw_root}datasets/{raw_type}/bugfixes.key.csv")
source_code_root = f"{raw_root}sciclone/data10/mtufano/deepLearningMutants/out/changes/code"
# get source_code and operators
org_idx_based_dict = {}
for org_idx, mapped_idx in enumerate(tqdm(idx_buggy, desc='loading mapper...')):
    assert mapped_idx == mapper.loc[mapped_idx]['idx'] - 1
    repo = mapper.loc[mapped_idx]['repo']
    f1 = mapper.loc[mapped_idx]['f1']
    f2 = mapper.loc[mapped_idx]['f2']
    buggy_source_code = open(f"{source_code_root}/{repo}/{f1}/{f2}/before.java", 'r').read()
    fixed_source_code = open(f"{source_code_root}/{repo}/{f1}/{f2}/after.java", 'r').read()
    operations = open(f"{source_code_root}/{repo}/{f1}/{f2}/operations.txt", 'r').read()
    signature = open(f"{source_code_root}/{repo}/{f1}/{f2}/signature.txt", 'r').read()

    # test
    # print(all_buggy[mapped_idx])
    # print(all_fixed[mapped_idx])
    # print(buggy_source_code)
    # print(fixed_source_code)
    # if mapped_idx == 10:
    #     break

    org_idx_based_dict[str(org_idx)] = {
        'buggy': all_buggy[mapped_idx],
        'fixed': all_fixed[mapped_idx],
        'buggy_source_code': buggy_source_code,
        'fixed_source_code': fixed_source_code,
        'operations': operations,
        'signature': signature
    }
save_as_json(org_idx_based_dict, mapped_datas_save_path)

if have_loss:
    cur_datas = load_from_json(cur_datas_path)

    for cur_data in tqdm(cur_datas, desc='mapping...'):
        cur_idx = cur_data['id']
        mapped_dict_item = org_idx_based_dict[cur_idx]
        cur_data['buggy_source_code'] = mapped_dict_item['buggy_source_code']
        cur_data['fixed_source_code'] = mapped_dict_item['fixed_source_code']
        cur_data['operations'] = mapped_dict_item['operations']
        cur_data['signature'] = mapped_dict_item['signature']

    save_as_json(cur_datas, mapped_cur_datas_save_path)






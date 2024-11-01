from calendar import c
import copy
import random
import time
from turtle import title
from vae_cleaner import VaeCleaner
from rules import RulesCleaner
from utils import *
import pandas as pd


from temp import save_res_temp

random.seed(1216)


def save_res(filt_ids, full_datas, save_path, sample=False, orgs=None):
    res = {}
    res2 = {}
    N = len(filt_ids)
    n = (N*1.96*1.96*0.5*(1-0.5))/((N-1)*0.05*0.05+1.96*1.96*0.5*(1-0.5))
    print(f"Total size (N) : {N} Sample size (n) : {int(n+1)}")
    random.shuffle(filt_ids)
    filt_ids = filt_ids[:int(n+1)]
    for idx in filt_ids:
        res[idx] = full_datas[idx]
        if orgs is not None:
            res2[idx] = orgs[idx]

    save_as_json(res, save_path+".json")
    if orgs is not None:
        save_as_json(res2, save_path+"-org.json")
    # save as csv
    df = pd.DataFrame(res)
    df = df.T
    # save
    df.to_csv(save_path+".csv")
    if orgs is not None:
        df2 = pd.DataFrame(res2)
        df2 = df2.T
        df2.to_csv(save_path+"-org.csv")


if __name__ == "__main__":
    # args

    data_type = "train"
    data_size = 'small'
    datas = load_from_json(
        f"../code-refinement/data/{data_size}/{data_type}_mapped.json")
    have_loss = False
    is_save = False
    rule_cleaner = RulesCleaner()
    vae_cleaner = VaeCleaner()

    acts_path = f"../code-refinement/data/{data_size}-addMain/{data_type}-action.json"
    ept_buggy_acts_path = f"../code-refinement/data/{data_size}-addMain/{data_type}-empty-buggy-action.json"
    ept_fixed_acts_path = f"../code-refinement/data/{data_size}-addMain/{data_type}-empty-fixed-action.json"

    cleaned_output_file = f"../code-refinement/data/{data_size}/{data_type}-cleaned"

    org_buggy_dir = f"../code-refinement/data/{data_size}-addMain/org-{data_type}-buggy-javafile"
    org_fixed_dir = f"../code-refinement/data/{data_size}-addMain/org-{data_type}-fixed-javafile"

    if have_loss:
        # TODO: do not use this way
        loss_path = f"../code-refinement/data/inference/dev_res_sorted_mapped.json"

    # load data

    acts = load_from_json(acts_path)
    ept_buggy_acts = load_from_json(ept_buggy_acts_path)
    ept_fixed_acts = load_from_json(ept_fixed_acts_path)

    filterd_idx = set()

    # main process
    start_time = time.time()
    ast_start_time = time.time()
    ast_edit_filtered = rule_cleaner.rule_AST_edit_distance(
        datas, acts, ept_buggy_acts, ept_fixed_acts)
    ast_end_time = time.time()
    print(f"[ast] Time: {ast_end_time - ast_start_time}")
    filterd_idx.update(ast_edit_filtered)
    if is_save:
        save_res(ast_edit_filtered, datas,
                f"./filtered/{data_size}-{data_type}-ast_edit", True)

    ept_start_time = time.time()
    ept_filtered_idx = rule_cleaner.rule_empty_method(datas)
    ept_end_time = time.time()
    print(f"[ept] Time: {ept_end_time - ept_start_time}")
    filterd_idx.update(ept_filtered_idx)
    if is_save:
        save_res(ept_filtered_idx, datas,
                f"./filtered/{data_size}-{data_type}-ept", True)

    pmd_start_time = time.time()
    pmd_filtered_idx = rule_cleaner.rule_pmd_check(
        org_buggy_dir, org_fixed_dir)
    pmd_end_time = time.time()
    print(f"[pmd] Time: {pmd_end_time - pmd_start_time}")
    filterd_idx.update(pmd_filtered_idx)
    if is_save:
        save_res(pmd_filtered_idx, datas,
                f"./filtered/{data_size}-{data_type}-pmd", True)
    vae_start_time = time.time()
    vae_filtered_idx = vae_cleaner(datas, acts)
    vae_end_time = time.time()
    print(f"[vae] Time: {vae_end_time - vae_start_time}")
    filterd_idx.update(vae_filtered_idx)
    if is_save:
        save_res(vae_filtered_idx, datas,
                f"./filtered/{data_size}-{data_type}-vae", True)

    # datas changed
    # deep copy
    org_data = copy.deepcopy(datas)
    urp_start_time = time.time()
    urp_changed_idx, datas = rule_cleaner.rule_unreplaceable_change(
        datas, acts)
    urp_end_time = time.time()
    print(f"[urp] Time: {urp_end_time - urp_start_time}")
    if is_save:
        save_res(urp_changed_idx, datas,
                f"./filtered/{data_size}-{data_type}-urp", True, org_data)

    end_time = time.time()

    print(f"[ours] Time: {end_time - start_time}")

    ### temp
    
    filterd_idx.update(urp_changed_idx)
    filterd_idx = list(filterd_idx)
    # save_res_temp(filterd_idx, datas, f"./filtered/{data_size}-{data_type}-500sample", org_data)
    
    ####
    
    
    # delete these datas
    datas = del_datas(datas, list(filterd_idx))

    venn_plt(ast_edit_distance=ast_edit_filtered,
             empty_method=ept_filtered_idx,
             pmd_check=pmd_filtered_idx,
             vae=vae_filtered_idx,
             unreplaced=urp_changed_idx, save_path=f"./figures/{data_size}-{data_type}-venn.pdf")
    # if have_loss:
    #     loss_avg(ast_edit_distance=ast_edit_filtered,
    #              empty_method=ept_filtered_idx,
    #              pmd_check=pmd_filtered_idx,
    #              vae=vae_filtered_idx,
    #              unreplaced=urp_changed_idx, loss_path=loss_path)

    # write_to_file(datas=datas, file_path=cleaned_output_file, data_type=data_type)

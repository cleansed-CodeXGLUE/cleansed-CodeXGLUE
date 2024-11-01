import json
import os


def load_from_json(load_path, by_line=False):
    if by_line:
        data = []
        for line in open(load_path, "r", encoding="utf8"):
            data.append(json.loads(line))
        return data
    with open(load_path, "r") as f:
        data = json.load(f)
    return data


def save_as_json(data, save_path):
    data_json = json.dumps(data, indent=4)
    with open(save_path, "w") as file:
        file.write(data_json)


def del_datas(datas: dict, filterd_idx: set) -> dict:
    """
    Function to delete the data by the index
    """
    originial_len = len(datas.keys())
    for idx in filterd_idx:
        datas.pop(idx)
    assert len(datas.keys()) == originial_len - len(filterd_idx)
    print(
        f"[ours] filtered: {originial_len} -> {len(datas)} ({len(filterd_idx)})")
    return datas


def write_to_file(datas, file_path, data_type):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    save_as_json(datas, f"{file_path}/{data_type}-datas.json")

    buggy_text = f"{file_path}/{data_type}.buggy"
    fixed_text = f"{file_path}/{data_type}.fixed"
    idx_text = f"{file_path}/{data_type}.idx"

    idx_list = datas.keys()

    with open(buggy_text, "w") as f:
        for idx in idx_list:
            f.write(datas[idx]["buggy"])

    with open(fixed_text, "w") as f:
        for idx in idx_list:
            f.write(datas[idx]["fixed"])

    with open(idx_text, "w") as f:
        for idx in idx_list:
            f.write(f"{idx}\n")


def loss_avg(**kwargs):
    """
    Function to calculate the average loss
    """
    loss_list = load_from_json(kwargs["loss_path"])
    loss_dict = {loss_item["id"]: loss_item["loss"] for loss_item in loss_list}
    cl_name_list = [k for k, _ in kwargs.items() if k != 'loss_path']
    
    base_loss = 0
    for idx in loss_dict.keys():
        base_loss += loss_dict[idx]
    base_loss /= len(loss_dict)
    
    for cl_name in cl_name_list:
        total_loss = 0
        cl_idx_list = kwargs[cl_name]
        for idx in cl_idx_list:
            total_loss += loss_dict[idx]
        avg_loss = total_loss / len(cl_idx_list)
        print(f"[{cl_name}] avg_loss: {avg_loss:.4f} (base: {base_loss}, num: {len(cl_idx_list)})")

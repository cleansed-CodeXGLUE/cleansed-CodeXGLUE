

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

data_size = "small"
data_type = "train"
root_dir = "../code-refinement/data"
datas_path = f"{root_dir}/{data_size}/{data_type}_mapped.json"
datas_mapped = load_from_json(datas_path)

if not os.path.exists(f"{root_dir}/{data_size}-addMain/org-{data_type}-buggy-javafile"):
    os.makedirs(f"{root_dir}/{data_size}-addMain/org-{data_type}-buggy-javafile")
    
if not os.path.exists(f"{root_dir}/{data_size}-addMain/org-{data_type}-fixed-javafile"):
    os.makedirs(f"{root_dir}/{data_size}-addMain/org-{data_type}-fixed-javafile")


for idx in datas_mapped.keys():
    source_code = "public class Main {\n" + datas_mapped[idx]["buggy_source_code"] + "\n}"
    target_code = "public class Main {\n" + datas_mapped[idx]["fixed_source_code"] + "\n}"
    # write to file
    with open(f"{root_dir}/{data_size}-addMain/org-{data_type}-buggy-javafile/{idx}.java", "w") as f:
        f.write(source_code)
    with open(f"{root_dir}/{data_size}-addMain/org-{data_type}-fixed-javafile/{idx}.java", "w") as f:
        f.write(target_code)

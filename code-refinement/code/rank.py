import json


def save_as_json(data, save_path):
    data_json = json.dumps(data, indent=4)
    with open(save_path, "w") as file:
        file.write(data_json)


def load_from_json(load_path, by_line=False):
    if by_line:
        data = []
        for line in open(load_path, "r", encoding="utf8"):
            data.append(json.loads(line))
        return data
    with open(load_path, "r") as f:
        data = json.load(f)
    return data


data = load_from_json("../data/inference/dev_res.json")
ids = data.keys()

for id in ids:
    data[id]["id"] = id
print(len(ids))
data_list = [data[id] for id in ids]

# sort by loss reverse
data_list.sort(key=lambda x: x["loss"], reverse=True)
save_as_json(data_list, "../data/inference/dev_res_sorted.json")


'''md
## action embedding:
0. none
1. insert
2. update
3. delete
4. move

example: [[2],[1],[3],[4]]

## 修改语句类型

example: [[32,43],[54,43],[23],[3,42]]

## position embedding:
相对位置 0-1

example: [[0.2,0.3],[0.4,0.5],[0.8],[0.9,0.1]]

## 语句类型 length embedding:
长度

example: [[12,21],[32,43,21,3],[2],[3,42]]

## 函数签名是否变化

## 前后长度
'''

import re
from utils.tools import *
from utils.fea_map import FeatureMap
from tqdm import tqdm
import numpy as np
from vae_attention.models.vae import VAE
from pyod.models.vae import VAE

class VaeCleaner:
    # def _action_embedding(action_dict):
    def __init__(self):
        self.fea_map = FeatureMap()
        self.pattern1 = r"\n(\w+): (\S+) \[(\d+),(\d+)\]\n"
        self.pattern2 = r"\n(\w+) \[(\d+),(\d+)\]\n"
        self.datas: dict = None

    def __call__(self, datas: dict, acts: dict, **kwds) -> dict:
        self.datas = datas
        self._state_extract(acts)
        max_len_dim0 = 0
        max_len_dim1 = 0
        cnt = 0
        fea_dict = {}
        for idx in tqdm(acts.keys(), desc="fea_extract..."):
            action = acts[idx]
            act_type_list = self._get_action_type_list(action)
            state_infos = self._get_state_infos_irrmatrixs(action)
            fea = [a + b + c + d for a, b, c, d in zip(
                act_type_list, state_infos["state_types"], state_infos["state_pos"], state_infos["state_len"])]
            cnt += 1
            max_len_dim0 = max(max_len_dim0, len(fea))
            max_len_dim1 = max(max_len_dim1, max([len(x) for x in fea]))
            fea_dict[idx] = fea
        print(f"max_len_dim0: {max_len_dim0}, max_len_dim1: {max_len_dim1}")
        fea_dict = self._padding(fea_dict, max_len_dim0, max_len_dim1)
        
        fea_list = []
        idx_list = []

        for idx in fea_dict.keys():
            fea_list.append(fea_dict[idx])
            idx_list.append(idx)

        vae = VAE()
        fea_list = np.array(fea_list)
        fea_list = fea_list.reshape(
            fea_list.shape[0], fea_list.shape[1] * fea_list.shape[2])

        vae.fit(fea_list)

        labels = vae.labels_
        
        filtered_idx = [idx_list[i] for i in range(len(labels)) if labels[i] == 1]
        
        print(f"[vae_based_cleaner] filtered-idx-num: {len(filtered_idx)}")
        
        return filtered_idx

    def _padding(self, fea_dict, max_len_dim0, max_len_dim1):
        for idx in fea_dict.keys():
            fea = fea_dict[idx]
            for i in range(max_len_dim0 - len(fea)):
                fea.append([0])
            for i in range(len(fea)):
                for j in range(max_len_dim1 - len(fea[i])):
                    fea[i].append(0)
            assert len(fea) == max_len_dim0
            for i in range(len(fea)):
                assert len(fea[i]) == max_len_dim1
        return fea_dict

    def _get_action_type_list(self, action) -> list:
        act_type_list = []
        for opt in action["opts_org"]:
            act_type_list.append([self.fea_map.get_action_type_idx(opt)])
        return act_type_list

    def _get_state_infos_irrmatrixs(self, action) -> dict:
        state_types_irrmatrix = []
        state_pos_irrmatrix = []
        state_len_irrmatrix = []
        for full_item in action["fulls_org"]:
            matches1 = re.finditer(self.pattern1, full_item)
            matches2 = re.finditer(self.pattern2, full_item)
            pos_list = []
            mat_item_list = []
            for item in matches1:
                mat_item_list.append(item)
                pos_list.append(item.start())
            for item in matches2:
                mat_item_list.append(item)
                pos_list.append(item.start())
            # sort mat_item_list by pos_list
            mat_item_list = [x for _, x in sorted(
                zip(pos_list, mat_item_list))]
            state_types_list = self._get_state_types_list(mat_item_list)
            state_pos_list = self._get_state_pos_list(
                mat_item_list, action["idx"])
            state_len_list = self._get_state_len_list(mat_item_list)

            state_types_irrmatrix.append(state_types_list)
            state_pos_irrmatrix.append(state_pos_list)
            state_len_irrmatrix.append(state_len_list)

        return {
            "state_types": state_types_irrmatrix,
            "state_pos": state_pos_irrmatrix,
            "state_len": state_len_irrmatrix
        }

    def _get_state_types_list(self, mat_item_list: list) -> list:
        state_types = []
        for item in mat_item_list:
            state_types.append(self.fea_map.get_state_type_idx(item.group(1)))
        return state_types

    def _get_state_pos_list(self, mat_item_list: list, idx: str) -> list:
        state_pos_list = []
        # beacause of the addMain
        org_len = len(self.datas[idx]["buggy"]) + 22
        for item in mat_item_list:
            item_num = len(item.groups())
            assert item_num == 4 or item_num == 3
            state_pos = item.group(3) if item_num == 4 else item.group(2)
            rel_pos = int(state_pos) / org_len
            state_pos_list.append(rel_pos)

        return state_pos_list

    def _get_state_len_list(self, mat_item_list: list) -> list:
        state_len_list = []
        for item in mat_item_list:
            item_num = len(item.groups())
            assert item_num == 4 or item_num == 3
            state_start = item.group(3) if item_num == 4 else item.group(2)
            state_end = item.group(4) if item_num == 4 else item.group(3)
            state_len = int(state_end) - int(state_start)
            state_len_list.append(state_len)

        return state_len_list

    def _state_extract(self, acts: dict):
        set_state_type = set()
        for idx in acts.keys():
            action = acts[idx]
            for full_item in action["fulls_org"]:
                matches1 = re.findall(self.pattern1, full_item)
                matches2 = re.findall(self.pattern2, full_item)
                for item in matches1:
                    set_state_type.add(item[0])
                for item in matches2:
                    set_state_type.add(item[0])
        self.fea_map.set_state_type_dict(set_state_type)


if __name__ == "__main__":

    data_type = "valid"
    vae_cleaner = VaeCleaner()
    action_path = f"../code-refinement/data/small-addMain/{data_type}-action.json"
    datas_path = f"../code-refinement/data/small/{data_type}_mapped.json"
    acts = load_from_json(action_path)
    datas = load_from_json(datas_path)
    vae_cleaner(acts=acts, datas=datas)

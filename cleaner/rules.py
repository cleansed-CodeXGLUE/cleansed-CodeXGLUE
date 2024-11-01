from ast import List
import json
import re


from networkx import ra_index_soundarajan_hopcroft
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from regex import R
from scipy.stats import kendalltau, spearmanr
import seaborn as sns
from torch import le
import tree_sitter
from utils.tools import *
from tree_sitter import Language, Parser
import tree_sitter_java as tsj
import subprocess
import xml.etree.ElementTree as ET


class RulesCleaner:

    def rule_AST_edit_distance(self, datas, acts, ept_buggy_acts, ept_fixed_acts, theshold=0.4):

        score_res = {x: {} for x in datas.keys()}

        for idx in datas.keys():
            total_dist = len(
                ept_buggy_acts[idx]["opts"])+len(ept_fixed_acts[idx]["opts"])
            dist = 0
            for action in acts[idx]["opts"]:
                if "update" in action:
                    continue
                dist += 1
            score_res[idx]["AST_edit"] = dist/total_dist

        filtered_idx = [
            x for x in datas.keys() if score_res[x]["AST_edit"] > theshold]

        print(
            f"[rule_AST_edit_distance] filtered-idx-num: {len(filtered_idx)}")

        return filtered_idx

    def rule_empty_method(self, datas):
        filtered_idx = []

        for idx in datas.keys():
            buggy = datas[idx]["buggy"]
            fixed = datas[idx]["fixed"]
            if self._check_empty_method(buggy.encode()) or self._check_empty_method(fixed.encode()):
                filtered_idx.append(idx)

        print(f"[rule_empty_method] filtered-idx-num: {len(filtered_idx)}")
        return filtered_idx

    def rule_unreplaceable_change(self, datas: dict, acts: dict):
        idx_list = datas.keys()
        changed_idx = []
        rep_cnt = 0
        for idx in idx_list:
            # for buggy
            opts_list = acts[idx]["opts"]
            up_idxs = np.where(np.isin(np.array(opts_list), "update-node"))[0]
            full_opts_list = acts[idx]["fulls"]
            input_str = datas[idx]["buggy"]
            is_rep, _cnt, buggy_str = self._unreplaceable_change(
                idx, up_idxs, full_opts_list, input_str)
            rep_cnt += _cnt
            if is_rep:
                datas[idx]["buggy"] = buggy_str

            # for fixed
            opts_list_rev = acts[idx]["opts_rev"]
            up_idxs_rev = np.where(
                np.isin(np.array(opts_list_rev), "update-node"))[0]
            full_opts_list_rev = acts[idx]["fulls_rev"]
            input_str = datas[idx]["fixed"]
            is_rep_rev, _cnt_rev, fixed_str = self._unreplaceable_change(
                idx, up_idxs_rev, full_opts_list_rev, input_str)
            rep_cnt += _cnt_rev
            if is_rep_rev:
                datas[idx]["fixed"] = fixed_str

            if is_rep or is_rep_rev:
                changed_idx.append(idx)

        print(
            f"[rule_unreplaceable_change] changed-idx-num: {len(changed_idx)}, rep-cnt: {rep_cnt}")

        return changed_idx, datas

    def rule_pmd_check(self, org_buggy_dir, org_fixed_dir):
        pmd_buggy_xml_path = f"{os.path.dirname(org_buggy_dir)}/{os.path.basename(org_buggy_dir)}.xml"
        pmd_fixed_xml_path = f"{os.path.dirname(org_fixed_dir)}/{os.path.basename(org_fixed_dir)}.xml"

        subprocess.run(["./pmd-bin-7.3.0/bin/pmd", "check", "-d", org_buggy_dir, "-R",
                        "./pmd-bin-7.3.0/quickstart.xml", "-f", "xml", "-r", pmd_buggy_xml_path])

        subprocess.run(["./pmd-bin-7.3.0/bin/pmd", "check", "-d", org_fixed_dir, "-R",
                        "./pmd-bin-7.3.0/quickstart.xml", "-f", "xml", "-r", pmd_fixed_xml_path])

        violation_buggy = self._parse_pmd_xml(pmd_buggy_xml_path)

        violation_fixed = self._parse_pmd_xml(pmd_fixed_xml_path)

        # no more new bugs
        filtered_idx = []
        idx_list = violation_buggy.keys()
        for idx in idx_list:
            if (violation_fixed[idx]["priority"].count('1') > violation_buggy[idx]["priority"].count('1')) or (violation_fixed[idx]["priority"].count('2') > violation_buggy[idx]["priority"].count('2')):
                filtered_idx.append(idx)
            # if len(violation_fixed[idx]["priority"]) > len(violation_buggy[idx]["priority"]):
            #     filtered_idx.append(idx)
            # if (violation_fixed[idx]["priority"].count('1') > 0):
            #     filtered_idx.append(idx)
        # print(filtered_idx)

        print(f"[rule_pmd_check] filtered-idx-num: {len(filtered_idx)}")
        
        return filtered_idx

    def _parse_pmd_xml(self, xml_path) -> dict:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        namespace = {'pmd': 'http://pmd.sourceforge.net/report/2.0.0'}
        res = {}
        for file_elem in root.findall('pmd:file', namespace):
            file_name = file_elem.get('name')
            file_number = file_name.split('/')[-1].split('.')[0]
            res[str(file_number)] = {"priority": []}
            for violation_elem in file_elem.findall('pmd:violation', namespace):
                priority = violation_elem.get('priority')
                # rule = violation_elem.get('rule')
                # res[str(file_number)]["rule"].append(rule)
                res[str(file_number)]["priority"].append(priority)
        return res

    def _check_empty_method(self, source_code):
        JAVA_LANGUAGE = Language(tsj.language())
        parser = Parser(JAVA_LANGUAGE)
        tree = parser.parse(source_code)
        # Helper function to check if a node is an empty method

        def is_empty_method(node):
            for child in node.children:
                if child.type == 'block':
                    # Check if the block is empty (only contains one child, the `{` and `}` token)
                    if len(child.children) <= 2:
                        return True
                    # Check if the block only contains "return 0;"
                    elif len(child.children) == 3 and child.children[1].type == 'return_statement':
                        return_statement = child.children[1]
                        if return_statement.children[1].type == 'decimal_integer_literal':
                            return True

            return False

        # Get the root node of the parsed tree
        root_node = tree.root_node

        # Find all method declarations
        methods = [n for n in root_node.children if n.type ==
                   'method_declaration']

        # Check each method to see if it is empty
        for method in methods:
            if is_empty_method(method):
                return True

    def _unreplaceable_change(self, idx, up_idxs, full_opts_list, input_str):
        is_rep = False
        rep_cnt = 0
        rep_name_list = []
        rep_pos_list = []
        for up_idx in up_idxs:
            full_opt = full_opts_list[up_idx]
            re_text = r'===\nupdate-node\n---\nSimpleName: (\w+) \[(\d+),(\d+)\]\nreplace (\w+) by (\w+)'
            up_macher = re.fullmatch(re_text, full_opt)
            if up_macher is None:
                continue
            buggy_name = up_macher.group(1)
            start_pos = int(up_macher.group(2)) - 20  # beacuse of the addMain
            end_pos = int(up_macher.group(3)) - 20
            replaced_name = up_macher.group(4)
            fixed_name = up_macher.group(5)
            assert buggy_name == replaced_name, print(
                f"error: {buggy_name}!={replaced_name}")
            abs_re_text = "([A-Z]+_\d+)"
            abs_macher_buggy = re.fullmatch(abs_re_text, buggy_name)
            abs_macher_fixed = re.fullmatch(abs_re_text, fixed_name)
            if (abs_macher_buggy is None) and (abs_macher_fixed is not None):
                rep_name = abs_macher_fixed.group(1)
                rep_name_list.append(rep_name)
                rep_pos_list.append((start_pos, end_pos))
        # reverse
        rep_name_list = rep_name_list[::-1]
        rep_pos_list = rep_pos_list[::-1]
        for rep_name, rep_pos in zip(rep_name_list, rep_pos_list):
            # print(idx)
            # print(input_str)
            input_str = f"{input_str[:rep_pos[0]]}{rep_name}{input_str[rep_pos[1]:]}"
            # print(input_str)
            is_rep = True
            rep_cnt += 1
        return is_rep, rep_cnt, input_str


if __name__ == "__main__":
    data_type = "valid"
    data_size = 'small'
    datas = load_from_json(
        f"../code-refinement/data/{data_size}/{data_type}_mapped.json")
    idx_list = datas.keys()

    rule_cleaner = RulesCleaner()

    acts_path = f"../code-refinement/data/{data_size}-addMain/{data_type}-action.json"
    ept_buggy_acts_path = f"../code-refinement/data/{data_size}-addMain/{data_type}-empty-buggy-action.json"
    ept_fixed_acts_path = f"../code-refinement/data/{data_size}-addMain/{data_type}-empty-fixed-action.json"

    acts = load_from_json(acts_path)
    ept_buggy_acts = load_from_json(ept_buggy_acts_path)
    ept_fixed_acts = load_from_json(ept_fixed_acts_path)

    ast_edit_filtered = rule_cleaner.rule_AST_edit_distance(
        datas, acts, ept_buggy_acts, ept_fixed_acts)

    ept_filtered_idx = rule_cleaner.rule_empty_method(datas)

    urp_changed_idx, new_datas = rule_cleaner.rule_unreplaceable_change(
        datas, acts)

    # empty_method_filtered_idx = rule_empty_method(datas)
    # print(len(empty_method_filtered_idx))

    # rule_unreplaceable_change(datas)

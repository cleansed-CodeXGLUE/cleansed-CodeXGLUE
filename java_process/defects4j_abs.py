
import subprocess

from tqdm import tqdm

import os
def abs_process():
    defects4j_buggy_dir = "./before/"
    defects4j_fixed_dir = "./after/"
    # list all the projects
    filenames = os.listdir(defects4j_buggy_dir)

    for filename in tqdm(filenames):
        command = [
            "java", "-jar", "/home/shweng/code_data_clean/java_process/src2abs/target/src2abs-0.1-jar-with-dependencies.jar",
            "pair", "method", f"/home/shweng/code_data_clean/java_process/before/{filename}",
            f"/home/shweng/code_data_clean/java_process/after/{filename}",
            f"/home/shweng/code_data_clean/code-refinement/data/defects4j/buggy/{filename}",
            f"/home/shweng/code_data_clean/code-refinement/data/defects4j/fixed/{filename}",
            f"/home/shweng/code_data_clean/java_process/src2abs/idioms/idioms-review.csv",
        ]
        subprocess.run(command, capture_output=False, text=True)

def limit_tokens():
    defects4j_fixed_dir = "/home/shweng/code_data_clean/code-refinement/data/defects4j/fixed/"
    defects4j_buggy_dir = "/home/shweng/code_data_clean/code-refinement/data/defects4j/buggy/"
    filenames = os.listdir(defects4j_fixed_dir)
    for filename in tqdm(filenames):
        with open(f"{defects4j_buggy_dir}{filename}", "r") as f:
            buggy_lines = f.readlines()
        with open(f"{defects4j_fixed_dir}{filename}", "r") as f:
            fixed_lines = f.readlines()
        buggy_line = ""
        for line in buggy_lines:
            buggy_line += line
        fixed_line = ""
        for line in fixed_lines:
            fixed_line += line
        if len(buggy_line.split()) > 50 or len(fixed_line.split()) > 50:
            continue
        # wirte to the end of the text file
        with open(f"/home/shweng/code_data_clean/code-refinement/data/defects4j/valid.buggy", "a") as f:
            f.write(buggy_line+"\n")
        with open(f"/home/shweng/code_data_clean/code-refinement/data/defects4j/valid.fixed", "a") as f:
            f.write(fixed_line+"\n")
        

limit_tokens()
        

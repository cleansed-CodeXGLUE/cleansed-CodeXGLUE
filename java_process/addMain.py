import os


class Example(object):
    """A single training/test example."""

    def __init__(self,
                 idx,
                 source,
                 target,
                 ):
        self.idx = idx
        self.source = source
        self.target = target


def read_examples(filename):
    """Read examples from filename."""
    examples = []
    assert len(filename.split(',')) == 2
    src_filename = filename.split(',')[0]
    trg_filename = filename.split(',')[1]
    idx = 0
    with open(src_filename) as f1, open(trg_filename) as f2:
        for line1, line2 in zip(f1, f2):
            examples.append(
                Example(
                    idx=idx,
                    source=line1.strip(),
                    target=line2.strip(),
                )
            )
            idx += 1
    return examples


# args
data_size = "small"
data_type = "train"
root_dir = "../code-refinement/data"
examples_path = f"{root_dir}/{data_size}/{data_type}.buggy-fixed.buggy,{root_dir}/{data_size}/{data_type}.buggy-fixed.fixed"

if not os.path.exists(f"{root_dir}/{data_size}-addMain/{data_type}-buggy-javafile"):
    os.makedirs(f"{root_dir}/{data_size}-addMain/{data_type}-buggy-javafile")

if not os.path.exists(f"{root_dir}/{data_size}-addMain/{data_type}-fixed-javafile"):
    os.makedirs(f"{root_dir}/{data_size}-addMain/{data_type}-fixed-javafile")

examples = read_examples(examples_path)
for example in examples:
    # add public class Main { to the beginning of the code and add } to the end of the code
    source_code = "public class Main { " + example.source + " }"
    target_code = "public class Main { " + example.target + " }"
    # write to file
    with open(f"{root_dir}/{data_size}-addMain/{data_type}-buggy-javafile/{example.idx}.java", "w") as f:
        f.write(source_code)
    with open(f"{root_dir}/{data_size}-addMain/{data_type}-fixed-javafile/{example.idx}.java", "w") as f:
        f.write(target_code)

print("Done!")

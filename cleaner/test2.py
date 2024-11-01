import re
from turtle import pos
import matplotlib.pyplot as plt
from matplotlib_venn import venn3, venn2
import torch


def find_node_type(text):
    pattern1 = r"(\w+): (\S+) \[(\d+),(\d+)\]"
    pattern2 = r"\n(\w+) \[(\d+),(\d+)\]\n"
    matches1 = re.finditer(pattern1, text)
    matches2 = re.finditer(pattern2, text)
    match_list = []
    for match in matches1:
        match_list.append(match)
        print(match.group(2))
    for match in matches2:
        match_list.append(match)
        print(len(match.groups()))
    print(match_list[0].group(1))

    list1 = [[2], [1], [3], [4]]
    list2 = [[32, 43], [54, 43, 21, 43], [23], [3, 42]]
    list3 = [[0.2, 0.3], [0.4, 0.5, 0.6, 0.7], [0.8], [0.9, 0.1]]
    merged_list = [a + b + c for a, b, c in zip(list1, list2, list3)]
    print(merged_list)
    # 测试代码
    text = "QualifiedName: org.bukkit.scheduler.BukkitRunnable [105,140]"
    # 输出：[('INFIX_EXPRESSION_OPERATOR: > [120,121]', ': >')]
    print(find_node_type(text))

    # text = "NumberLiteral: 0 [100,101]"
    # print(find_node_type(text))  # 输出：[('Block [126,174]', '')]


def venn_fig():
    # 示例数据
    list1 = [1, 2, 3, 4, 5]
    list2 = [4, 5, 6, 7, 8]
    list3 = [5, 7, 8, 9, 10]

    # 将列表转换为集合
    set1 = set(list1)
    set2 = set(list2)
    set3 = set(list3)

    # 绘制两个集合的Venn图
    plt.figure(figsize=(8, 4))
    venn2([set1, set2], ('List 1', 'List 2'))
    plt.title("Venn Diagram for List 1 and List 2")
    plt.show()

    # 绘制三个集合的Venn图
    plt.figure(figsize=(8, 8))
    venn3([set1, set2, set3], ('List 1', 'List 2', 'List 3'))
    plt.title("Venn Diagram for List 1, List 2 and List 3")
    plt.show()
    plt.savefig('test.png')


def pop_dict():
    datas = {"123": 444, "234": 555, "345": 666}
    filterd_idx = {"123", "345"}
    for idx in filterd_idx:
        datas.pop(idx)
    print(datas)

def testequal():
    generated_id = torch.tensor([1, 2, 3, 4, 5])
    fixed_e_index = (generated_id == 38).nonzero().numel()
    print(fixed_e_index)


if __name__ == '__main__':
    testequal()


from math import nan
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
# read csv
df = pd.read_csv('./filtered/small-train-vae-checked.csv',
                 encoding='ISO-8859-1')
print(len(df["check1"]), len(df["check2"]))
cnt1 = 0
cnt2 = 0
cnt = 0
for ck1, ck2 in zip(df["check1"], df["check2"]):
    if (ck1 + ck2)/2 <= -1:
        cnt += 1
    if ck1 <= -1:
        cnt1 += 1
    if ck2 <= -1:
        cnt2 += 1
print(cnt1, cnt2, cnt)
n = len(df["check1"])
p = cnt/n
print(f"Accuracy: {p}")
SE = ((p*(1-p))/n)**0.5
left = p - 1.96*SE
right = p + 1.96*SE
print(f"[{left}, {right}]")
scores_person1 = np.array(df["check1"].to_list())
scores_person2 = np.array(df["check2"].to_list())
print(scores_person1, scores_person2)
# calculate Pearson's correlation
correlation, _ = pearsonr(scores_person1, scores_person2)

print(f"Pearson correlation: {correlation}")

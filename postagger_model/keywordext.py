from head_tail_lstm import analysis
from tqdm import tqdm

import pandas as pd
import numpy as np

df = pd.read_csv("dataset.csv",encoding="utf-8")
dfs = df.fillna(" ")

values = dfs.loc[:,["sentence"]].values
hts = []
# print(len(values))
# exit()
for i in tqdm(range(0,len(values),200)):
    v = np.reshape(values[i:i+200],(-1)).tolist()
    ht = analysis(v)
    hts = hts + ht

df["head-tail"] = hts
df.to_csv("head-tail.csv",encoding="utf-8")
# synthetic data

import pandas as pd
from mdl_rank_est import calc_MDL
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import NMF

data = np.random.gamma(1, 2, (15, 200))
data = (np.random.gamma(5, 3, (100, 15))).dot(data)
np.save('./toy_data/syn_data.npy', data)

res = [[], [], [], [], [], []]  # res, lw, lw0, lh, lh0, le
ari = []

for rank in tqdm(range(1, min(data.shape)+1)):
    res_tmp = [[], [], [], [], [], []]  # res, lw, lw0, lh, lh0, le
    ari_tmp = []
#     print("rank:",rank)
    for j in range(3):
        nmf = NMF(n_components=rank)
        W = nmf.fit_transform(data)
        H = nmf.components_

        MDL_s = calc_MDL(data, np.array(W), np.array(H))
        res_tmp[0].append(sum(MDL_s))
        for i in range(len(MDL_s)):
            res_tmp[i+1].append(MDL_s[i])
        # print(MDL)

    for i in range(len(res)):
        res[i].append(res_tmp[i])

pd.DataFrame(np.array(res[0])).to_csv('./toy_data/MDL_def.csv')
pd.DataFrame(np.array(res[1])).to_csv('./toy_data/w_def.csv')
pd.DataFrame(np.array(res[2])).to_csv('./toy_data/w0_def.csv')
pd.DataFrame(np.array(res[3])).to_csv('./toy_data/h_def.csv')
pd.DataFrame(np.array(res[4])).to_csv('./toy_data/h0_def.csv')
pd.DataFrame(np.array(res[5])).to_csv('./toy_data/e_def.csv')

res = pd.read_csv('./toy_data/MDL_def.csv', index_col=0)
w0 = pd.read_csv('./toy_data/w0_def.csv', index_col=0)
w = pd.read_csv('./toy_data/w_def.csv', index_col=0)
h = pd.read_csv('./toy_data/h_def.csv', index_col=0)
h0 = pd.read_csv('./toy_data/h0_def.csv', index_col=0)
e = pd.read_csv('./toy_data/e_def.csv', index_col=0)

fig, ax = plt.subplots(figsize=(15, 10))

ax.errorbar(x=[i for i in range(1, res.shape[0]+1)],
            y=list(res.mean(axis=1).values),
            yerr=[[j-i for i, j in zip(res.min(axis=1), res.mean(axis=1))],
                  [i-j for i, j in zip(res.max(axis=1), res.mean(axis=1))]],
            capsize=3,
            fmt='o',
            ecolor='black',
            color='w',
            markeredgecolor="black",
            label='Ltot')

ax.errorbar(x=[i for i in range(1, res.shape[0]+1)],
            y=list(w0.mean(axis=1).values),
            yerr=[[j - i for i,
                   j in zip(w0.min(axis=1),
                            w0.mean(axis=1))],
                  [i - j for i,
                   j in zip(w0.max(axis=1),
                            w0.mean(axis=1))]],
            capsize=3,
            fmt='o',
            ecolor='red',
            color='w',
            markeredgecolor="red",
            label='LW0')


ax.errorbar(x=[i for i in range(1, res.shape[0]+1)],
            y=list(w.mean(axis=1).values),
            yerr=[[j - i for i,
                   j in zip(w.min(axis=1),
                            w.mean(axis=1))],
                  [i - j for i,
                   j in zip(w.max(axis=1),
                            w.mean(axis=1))]],
            capsize=3,
            fmt='o',
            ecolor='g',
            color='w',
            markeredgecolor="g",
            label='LW+')


ax.errorbar(x=[i for i in range(1, res.shape[0]+1)],
            y=list(h.mean(axis=1).values),
            yerr=[[j - i for i,
                   j in zip(h.min(axis=1),
                            h.mean(axis=1))],
                  [i - j for i,
                   j in zip(h.max(axis=1),
                            h.mean(axis=1))]],
            capsize=3,
            fmt='o',
            ecolor='c',
            color='w',
            markeredgecolor="c",
            label='LH+')

ax.errorbar(x=[i for i in range(1, res.shape[0]+1)],
            y=list(h0.mean(axis=1).values),
            yerr=[[j - i for i,
                   j in zip(h0.min(axis=1),
                            h0.mean(axis=1))],
                  [i - j for i,
                   j in zip(h0.max(axis=1),
                            h0.mean(axis=1))]],
            capsize=3,
            fmt='o',
            ecolor='m',
            color='w',
            markeredgecolor="m",
            label='LH0')

ax.errorbar(x=[i for i in range(1, res.shape[0]+1)],
            y=list(e.mean(axis=1).values),
            yerr=[[j - i for i,
                   j in zip(e.min(axis=1),
                            e.mean(axis=1))],
                  [i - j for i,
                   j in zip(e.max(axis=1),
                            e.mean(axis=1))]],
            capsize=3,
            fmt='o',
            ecolor='y',
            color='w',
            markeredgecolor="y",
            label='LE')
ax.legend()

ax.set_ylabel('MDL')
ax.set_xlabel('rank')
plt.savefig('./toy_data/syn_result.png')

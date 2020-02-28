import pandas as pd
import numpy as np
from scipy.stats import gamma, norm
from sklearn.preprocessing import Normalizer

import sys


def calc_MDL(data, mat_w, mat_h, d_D=1e-5, threshold_search_num=10):
    # Sturges' rule
    # d_D = (max(mat_w.max(), mat_h.max()) - min(mat_w.min(), mat_h.min())) / (round(np.log2(min(mat_w.size, mat_h.size)) + 1))
    # d_D = (max(mat_w.max(), mat_h.max()) - min(mat_w.min(), mat_h.min())) / 1e3
    # d_D = (data.max()-data.min())/(round(np.log2(data.size))+1)

    threshold = 0.
    min_res = sys.float_info.max
    res = []
    while threshold < d_D:
        LW, LW0 = __calc_factorized_MDL(mat_w, threshold, d_D)
        LH, LH0 = __calc_factorized_MDL(mat_h, threshold, d_D)
        LE = __calc_error_MDL(data - mat_w.dot(mat_h), d_D)
        if LW < 0 or LH < 0:
            break
        if min_res > LW + LW0 + LH + LH0 + LE:
            res = [LW, LW0, LH, LH0, LE]
            min_res = sum(res)
        threshold += d_D / threshold_search_num
    # print(LW,LW0,LH,LH0,LE)
    if len(res) == 0:
        print("res is none")
        sys.exit()
    # print(res)
    return res


def __calc_factorized_MDL(mat, threshold, d_D):
    L = -1
    L0 = 0
    if mat[mat > threshold].size == 0:
        return -1, -1

    a_hat, loc_hat, scale_hat = gamma.fit(mat[mat > threshold])
    # P = gamma.pdf(np.arange(mat.min(), mat.max(), d_D) + d_D / 2,a_hat,loc=loc_hat,scale=scale_hat) * d_D
    P = gamma.pdf(mat[mat > threshold], a_hat,
                  loc=loc_hat, scale=scale_hat) * d_D
    P[P > 1] = 1.
    P[P == 0.] = sys.float_info.min
    L = -np.log2(P).sum()
    if L < 0:
        sys.exit()
    n0 = mat[mat <= threshold].size
    if n0 != 0:
        nt = mat.size
        L0 = max(-n0 * np.log2(n0 / nt) - (nt - n0)
                 * np.log2((nt - n0) / nt), 0.)

    return L, L0


def __calc_error_MDL(mat, d_D):
    # d_D = (mat.max() - mat.min()) / round(np.log2(mat.size) + 1)
    # d_D = (mat.max() - mat.min()) / 1e3
    loc_hat, scale_hat = norm.fit(mat.flatten())
    # P = norm.pdf(np.arange(mat.min(), mat.max(), d_D) + d_D / 2,loc=loc_hat,scale=scale_hat) * d_D
    P = norm.pdf(mat.flatten(), loc=loc_hat, scale=scale_hat) * d_D
    # print("loc_hat: {0}, scale_hat: {1}".format(loc_hat, scale_hat))
    P[P == 0.] = sys.float_info.min
    L = -np.log2(P).sum()
    return L

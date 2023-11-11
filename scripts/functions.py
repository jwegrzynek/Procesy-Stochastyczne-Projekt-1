import numpy as np
from statsmodels.sandbox.stats.runs import runstest_1samp
import statsmodels.tsa.stattools as ts
from itertools import islice
import random


def interpret_WW(pvalue):
    if pvalue < 0.05:
        return "sekwencja nielosowa"
    else:
        return "sekwencja losowa"


def interpret_ADF(pvalue):
    if pvalue < 0.05:
        return "szereg stacjonarny"
    else:
        return "szereg niestacjonarny"


def chunk_list(lst, n):
    it = iter(lst)
    return iter(lambda: tuple(islice(it, n)), ())


def statistics(data):
    data = data.dropna()
    WW_pvalue = runstest_1samp(data, cutoff="median")[1]
    ADF_pvalue = ts.adfuller(data)[1]
    stats = [
        round(np.mean(data),3),
        round(np.std(data),3),
        round(np.min(data),3),
        round(np.max(data),3),
        round(WW_pvalue, 3),
        interpret_WW(WW_pvalue),
        round(ADF_pvalue, 3),
        interpret_ADF(ADF_pvalue)
    ]
    return stats


def generate_random_color():
    color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color
    


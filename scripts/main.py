# %% Imports

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
# import statsmodels.api as sm
from statsmodels.sandbox.stats.runs import runstest_1samp
import statsmodels.tsa.stattools as ts
from itertools import islice
from time import time
from arch.unitroot import ADF
from itertools import product

# %% Setting working directory and creating directories
script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(os.path.dirname(script_path)))
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)
patients_dir = os.path.join(os.getcwd(), "patients")
plots_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plots_dir, exist_ok=True)


# %% Loading file
files_list = os.listdir(data_dir)
df = pd.read_csv(os.path.join(data_dir, files_list[0]), sep='\t', names=['RR Interval', 'Index'])

# %% (A) Completeness of the series
all_indexes = np.arange(df['Index'][0], df['Index'][len(df['Index']) - 1] + 1)
missing_indexes = np.setdiff1d(all_indexes, df['Index'])

print("Brakujące indeksy:")
print(missing_indexes)

# %% Filling missing values
missing_indexes_df = pd.DataFrame({'Index': missing_indexes})
df = pd.concat([df, missing_indexes_df], ignore_index=True)
df = df.sort_values(by='Index')
#df['RR Interval'].interpolate(method="linear", inplace=True)

# %% Visualise RR Intervals
plt.plot(df['Index'], df['RR Interval'])
plt.show()

# %% Statistics
RR = df['RR Interval']
stats = [np.mean(RR), np.std(RR), np.min(RR), np.max(RR), np.quantile(RR, 0.25), np.quantile(RR, 0.5),
         np.quantile(RR, 0.75)]
print(stats)

# %% Tests
runstest_1samp(RR, cutoff="median")
ts.adfuller(RR)

# %% (C) Windows

def chunk(lst, n):
    it = iter(lst)
    return iter(lambda: tuple(islice(it, n)), ())


chunks = [batch for batch in list(chunk(RR, 20)) if len(batch) == 20]


#%% (C)

start = time()
for chunk in chunks:
    chunk = pd.Series(chunk)
    max_lags = int(np.sqrt(chunk.shape[0]))
    ADF(chunk, trend="n", max_lags=max_lags)
    
stop = time()

means_of_windows = [np.mean(chunk) for chunk in chunks]

mean_RR = np.mean(means_of_windows)
min_RR = np.min(means_of_windows)
max_RR = np.max(means_of_windows)
var_RR = np.std(means_of_windows)


print(stop-start)

# %% (B)

#plt.plot(ts_diff)
#plt.show()

delta_RR = RR.diff().dropna()

series_sumbolization = np.zeros(len(delta_RR), dtype=object)

for i, diff in enumerate(delta_RR):
    if 0 < diff < 40:
        series_sumbolization[i] = "a"
    elif -40 < diff < 0:
        series_sumbolization[i] = "d"
    elif 40 <= diff:
        series_sumbolization[i] = "A"
    elif diff <= -40:
        series_sumbolization[i] = "D"
    else:
        series_sumbolization[i] = "z"
    
pairs_in_series_symbolization = [series_sumbolization[i] + series_sumbolization[i + 1] for i in range(len(series_sumbolization) - 1)]

original_list = ["z", "a", "A", "d", "D"]

combinations_list = ["".join(pair) for pair in product(original_list, repeat=2)]

print(combinations_list)




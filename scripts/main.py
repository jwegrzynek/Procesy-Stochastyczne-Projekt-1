# %% Imports

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.stats.runs import runstest_1samp
import statsmodels.tsa.stattools as ts

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
df['RR Interval'].interpolate(method="linear", inplace=True)

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

from itertools import islice

def chunk(lst, n):
    it = iter(lst)
    return iter(lambda: tuple(islice(it, n)), ())


chunks = [batch for batch in list(chunk(RR, 20)) if len(batch) == 20]


#%%
from time import time
from arch.unitroot import ADF

start = time()
for chunk in chunks:
    chunk = pd.Series(chunk)
    max_lags = int(np.sqrt(chunk.shape[0]))
    print(ADF(chunk, trend="n", max_lags=max_lags))
stop = time()

print(stop-start)

#%%

import pandas as pd
import numpy as np

y = [3126.0, 3321.0, 3514.0, 3690.0, 3906.0, 4065.0, 4287.0, 
     4409.0, 4641.0, 4812.0, 4901.0, 5028.0, 5035.0, 5083.0,
     5183.0, 5377.0, 5428.0, 5601.0, 5705.0, 5895.0, 6234.0,
     6542.0, 6839.0]
y = pd.Series(y)

max_lags = int(np.sqrt(y.shape[0]))
ADF(y, trend="ct", max_lags=max_lags).summary()






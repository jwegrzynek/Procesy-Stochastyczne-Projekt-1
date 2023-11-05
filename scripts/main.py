# %% Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.sandbox.stats.runs import runstest_1samp


# %% Setting working directory and creating directories
script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(os.path.dirname(script_path)))
data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)
plots_dir = os.path.join(os.getcwd(), "plots")
os.makedirs(plots_dir, exist_ok=True)

# %% Loading file
files_list = os.listdir(data_dir)
df = pd.read_csv(os.path.join(data_dir, files_list[0]), sep='\t', names=['RR Interval', 'Index'])

# %% Completeness of the series
all_indexes = np.arange(df['Index'][0], df['Index'][len(df['Index']) - 1] + 1)
missing_indexes = np.setdiff1d(all_indexes, df['Index'])

print("Brakujące indeksy:")
print(missing_indexes)

# %% Visualise RR Intervals
plt.plot(df['Index'], df['RR Interval'])
plt.show()

# %% Statistics
RR = df['RR Interval']
stats = [np.mean(RR), np.std(RR), np.min(RR), np.max(RR), np.quantile(RR, 0.25), np.quantile(RR, 0.5),
         np.quantile(RR, 0.75)]
print(stats)

#Wald test danusie pojebało chyba?



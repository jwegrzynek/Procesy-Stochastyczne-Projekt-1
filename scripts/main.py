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

# %% (B) Differentiation

# Create a DataFrame from your data
data = {'Time Between Heart Beats': [712, 696, 680, 728, 728, 728, 728, 728, 656, 656, 640, 632, 624, 624],
        'Indexes': []}
dft = pd.DataFrame(data)

# Define the range of missing indexes
missing_index_range = range(100194, 100204)  # Adjust as needed

# Create a new DataFrame with the desired index range
missing_indexes_df = pd.DataFrame({'Indexes': missing_index_range})

# Merge the original DataFrame with the one containing missing indexes
merged_df = pd.concat([dft, missing_indexes_df], ignore_index=True)

# Sort the DataFrame by 'Indexes' if needed
merged_df = merged_df.sort_values(by='Indexes')

print(merged_df)


# %% pchip, slinear, linear

import pandas as pd

# Create a DataFrame with your data, including NaN values
data = {'Time Between Heart Beats': [712.0, 696.0, 680.0, 728.0, None, 728.0, 728.0, 728.0, None, None, None, None, None, None, None, None, None, None, None, 656.0, 656.0, 640.0, 632.0, 624.0, 624.0],
        'Indexes': [100186, 100187, 100188, 100189, 100190, 100191, 100192, 100193, 100194, 100195, 100196, 100197, 100198, 100199, 100200, 100201, 100202, 100203, 100204, 100204, 100205, 100206, 100207, 100208, 100209]}

df = pd.DataFrame(data)

# Use linear interpolation to fill NaN values
df['Time Between Heart Beats'].interpolate(method="linear", inplace=True)

# Display the updated DataFrame
print(df)



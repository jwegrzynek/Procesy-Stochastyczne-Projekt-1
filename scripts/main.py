# %% Imports
import os
script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(os.path.dirname(script_path))) #setting working directory

# Standard packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.sandbox.stats.runs import runstest_1samp
from arch.unitroot import ADF
from itertools import product

# My modules
from bubbleChart import BubbleChart
from functions import interpret_ADF, interpret_WW, chunk_list, statistics, generate_random_color

# %% Setting working directory and creating directories

data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)
patients_dir = os.path.join(os.getcwd(), "patients")
os.makedirs(patients_dir, exist_ok=True)

files_list = os.listdir(data_dir)

for file in files_list:
    file_name, file_extension = os.path.splitext(file)
    os.makedirs(os.path.join(patients_dir, ), exist_ok=True)

# %% Loading files

df = pd.read_csv(os.path.join(data_dir, files_list[0]), sep='\t', names=['RR Interval', 'Index'])


# %% (A) Completeness of the series
all_indexes = np.arange(df['Index'][0], df['Index'][len(df['Index']) - 1] + 1)
missing_indexes = np.setdiff1d(all_indexes, df['Index'])


# %% Filling missing values
missing_indexes_df = pd.DataFrame({'Index': missing_indexes})
df = pd.concat([df, missing_indexes_df], ignore_index=True)
df = df.sort_values(by='Index')
# df['RR Interval'].interpolate(method="linear", inplace=True)

# %% Visualise RR Intervals

fig, ax = plt.subplots()
fig.set_size_inches(45, 10)
ax.plot(df['Index'], df['RR Interval'])
ax.set_title('RR Intervals - {}'.format(files_list[0]))
ax.set_ylabel('RR Interval [ms]')
ax.set_xlabel('Index')
plt.savefig(os.path.join(plots_dir, 'RR Intervals Plot.pdf'.format(files_list[0])))
plt.show()

def plot(x, y, file_name, title, xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(45, 10)
    ax.plot(x, y)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(os.path.join(plots_dir, 'RR Intervals Plot.pdf'.format(file_name)))
    plt.show()


# %% Statistics

RR = df['RR Interval']
print(statistics(RR))


# %% (B)

RR = df['RR Interval']

for k in range(1, 21):
    print(statistics(RR.diff(k)))


# %% (C) Windows

for k in range(20, 201, 10):
    chunks = [batch for batch in list(chunk_list(RR.dropna(), k)) if len(batch) == k]
    
    WW_results = []
    ADF_results = []
    
    for chunk in chunks:
        chunk_ts = pd.Series(chunk)
        max_lags = int(np.sqrt(chunk_ts.shape[0]))
        ADF_pvalue = ADF(chunk_ts, trend="c", max_lags=max_lags).pvalue
        ADF_results.append(interpret_ADF(ADF_pvalue))
        WW_pvalue = runstest_1samp(chunk_ts.dropna(), cutoff="median")[1]
        WW_results.append(interpret_WW(WW_pvalue))
        
    means_of_windows = np.array([np.mean(chunk) for chunk in chunks])
    
    mean_RR = np.nanmean(means_of_windows)
    min_RR = np.nanmin(means_of_windows)
    max_RR = np.nanmax(means_of_windows)
    var_RR = np.nanstd(means_of_windows)
    random_sequences = WW_results.count("sekwencja losowa") / len(WW_results)
    nonstationary_serieses = ADF_results.count("szereg niestacjonarny") / len(ADF_results)
    stats = [
        k, 
        round(mean_RR,2), 
        round(min_RR,2), 
        round(max_RR,2), 
        round(var_RR,3), 
        round(random_sequences,3), 
        round(nonstationary_serieses,3)
    ]
    print(stats)


# %% (D)

delta_RR = RR.diff().dropna()
series_symbolization = list(np.zeros(len(delta_RR), dtype=object))

for i, diff in enumerate(delta_RR):
    if 0 < diff < 40:
        series_symbolization[i] = "d"
    elif -40 < diff < 0:
        series_symbolization[i] = "a"
    elif 40 <= diff:
        series_symbolization[i] = "D"
    elif diff <= -40:
        series_symbolization[i] = "A"
    else:
        series_symbolization[i] = "z"

pairs_in_series_symbolization = [series_symbolization[i] + series_symbolization[i + 1] for i in
                                 range(len(series_symbolization) - 1)]
tripples_in_series_symbolization = [series_symbolization[i] + series_symbolization[i + 1] + series_symbolization[i+2] for i in
                                 range(len(series_symbolization) - 2)]

single_events = ["z", "a", "A", "d", "D"]
double_events = ["".join(pair) for pair in product(single_events, repeat=2)]
tripple_events = ["".join(pair) for pair in product(single_events, repeat=3)]

single_events_occurrences = [series_symbolization.count(event) for event in single_events]
double_events_occurrences = [pairs_in_series_symbolization.count(event) for event in double_events]
tripple_events_occurrences = [tripples_in_series_symbolization.count(event) for event in tripple_events]


#%% Bubble chart

color_list = [generate_random_color() for _ in range(125)]

browser_market_share = {
    'browsers': tripple_events,
    'market_share': tripple_events_occurrences,
    'color': color_list
}

bubble_chart = BubbleChart(area=browser_market_share['market_share'],
                           bubble_spacing=0.1)

bubble_chart.collapse()

fig, ax = plt.subplots(subplot_kw=dict(aspect="equal"))
bubble_chart.plot(
    ax, browser_market_share['browsers'], browser_market_share['color'])
ax.axis("off")
ax.relim()
ax.autoscale_view()
ax.set_title('Browser market share')

plt.show()

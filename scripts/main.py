# %% Imports
import os

script_path = os.path.abspath(__file__)
os.chdir(os.path.dirname(script_path))  # setting working directory for modules import

# My modules
from functions import read_file, statistics, windows_statistics, rr_intervals_events, plot, hist, bar_plot, \
    stacked_plot, stacked_hist, merge_files

os.chdir(os.path.dirname(os.path.dirname(script_path)))

# Standard packages
import numpy as np
import pandas as pd

# %% Setting working directory and creating directories

data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)
patients_dir = os.path.join(os.getcwd(), "patients")
os.makedirs(patients_dir, exist_ok=True)
os.makedirs(os.path.join(patients_dir, "Group Statistics"), exist_ok=True)

files_list = os.listdir(data_dir)

for file in files_list:
    file_name, file_extension = os.path.splitext(file)
    os.makedirs(os.path.join(patients_dir, file_name, "plots"), exist_ok=True)
    os.makedirs(os.path.join(patients_dir, file_name, "results"), exist_ok=True)

# %% (A)

for file in files_list:
    file_name, file_extension = os.path.splitext(file)

    df = read_file(data_dir, file)
    RR = df['RR Interval']
    idx = df['Index']

    columns = ['śr', 'var', 'min', 'max', 'ww p-val', 'ww', 'adf p-val', 'adf', 'braki']
    data = np.array([statistics(RR) + ['{}/{}'.format(sum(RR.isna()), len(RR))]])

    df_A = pd.DataFrame(data, columns=columns)
    df_A.to_csv(os.path.join(patients_dir, file_name, "results", "A.txt"), sep='\t', index=False)

    plot(idx, RR, file_name, patients_dir, "Time Series")
    hist(RR, file_name, patients_dir, "Histogram RR")

# %% (B)

for file in files_list:
    file_name, file_extension = os.path.splitext(file)

    df = read_file(data_dir, file)
    RR = df['RR Interval']

    columns = ['k', 'śr', 'var', 'min', 'max', 'ww p-val', 'ww', 'adf p-val', 'adf']
    results_B = []

    for k in range(1, 21):
        results_B.append([k] + statistics(RR.diff(k)))

    df_B = pd.DataFrame(np.array(results_B), columns=columns)
    df_B.to_csv(os.path.join(patients_dir, file_name, "results", "B.txt"), sep='\t', index=False)

    stacked_plot(RR, file_name, patients_dir)
    stacked_hist(RR, file_name, patients_dir)

# %% (C)

for file in files_list:
    file_name, file_extension = os.path.splitext(file)

    df = read_file(data_dir, file)
    RR = df['RR Interval']

    columns = ['okno', 'śr', 'var', 'min', 'max', '% niezal', '% niestac']
    results_C = []

    for k in range(20, 201, 10):
        results_C.append(windows_statistics(RR, k))

    df_C = pd.DataFrame(np.array(results_C), columns=columns)
    df_C.to_csv(os.path.join(patients_dir, file_name, "results", "C.txt"), sep='\t', index=False)

# %% (D)

for file in files_list:
    file_name, file_extension = os.path.splitext(file)

    df = read_file(data_dir, file)
    RR = df['RR Interval']

    results = rr_intervals_events(RR)
    events = results[0] + results[2] + results[4]
    no_events = results[1] + results[3] + results[5]

    columns = ['event', 'occurrences']
    results_D = [[events[i], no_events[i]] for i in range(len(events))]

    df_D = pd.DataFrame(np.array(results_D), columns=columns)
    df_D.to_csv(os.path.join(patients_dir, file_name, "results", "D.txt"), sep='\t', index=False)

    bar_plot(results[0], results[1], file_name, patients_dir, "Single")
    bar_plot(results[2], results[3], file_name, patients_dir, "Double", hight=6, length=10, n=25)
    bar_plot(results[4], results[5], file_name, patients_dir, "Tripple", hight=6, length=12, n=25)

# %%

men = []
women = []

for file in files_list:
    file_name, file_extension = os.path.splitext(file)

    if file_name[0] == "f":
        women.append(file_name)
    else:
        men.append(file_name)

# %% Group (A)
men_file_paths = [os.path.join(patients_dir, file, "results", "A.txt") for file in men]
women_file_paths = [os.path.join(patients_dir, file, "results", "A.txt") for file in women]
selected_columns = ['śr', 'var', 'min', 'max', 'ww', 'adf']

men_df_A = merge_files(men_file_paths, selected_columns)
women_df_A = merge_files(women_file_paths, selected_columns)

men_df_A.columns = men_df_A.columns.str.replace("ww", "% niezal")
men_df_A.columns = men_df_A.columns.str.replace("adf", "% niestac")
women_df_A.columns = women_df_A.columns.str.replace("ww", "% niezal")
women_df_A.columns = women_df_A.columns.str.replace("adf", "% niestac")

men_df_A.to_csv(os.path.join(patients_dir, "Group Statistics", "A_males.txt"), index=False, sep='\t')
women_df_A.to_csv(os.path.join(patients_dir, "Group Statistics", "A_females.txt"), index=False, sep='\t')

# %% Group (B)
men_file_paths = [os.path.join(patients_dir, file, "results", "B.txt") for file in men]
women_file_paths = [os.path.join(patients_dir, file, "results", "B.txt") for file in women]
selected_columns = ['k', 'śr', 'var', 'min', 'max', 'ww', 'adf']

men_df_B = merge_files(men_file_paths, selected_columns)
women_df_B = merge_files(women_file_paths, selected_columns)

men_df_B.columns = men_df_B.columns.str.replace("ww", "% niezal")
men_df_B.columns = men_df_B.columns.str.replace("adf", "% niestac")
women_df_B.columns = women_df_B.columns.str.replace("ww", "% niezal")
women_df_B.columns = women_df_B.columns.str.replace("adf", "% niestac")

men_df_B.to_csv(os.path.join(patients_dir, "Group Statistics", "B_males.txt"), index=False, sep='\t')
women_df_B.to_csv(os.path.join(patients_dir, "Group Statistics", "B_females.txt"), index=False, sep='\t')

# %% Group (C)
men_file_paths = [os.path.join(patients_dir, file, "results", "C.txt") for file in men]
women_file_paths = [os.path.join(patients_dir, file, "results", "C.txt") for file in women]
selected_columns = ['okno', 'śr', 'var', 'min', 'max', '% niezal', '% niestac']

men_df_C = merge_files(men_file_paths, selected_columns)
women_df_C = merge_files(women_file_paths, selected_columns)

men_df_C.to_csv(os.path.join(patients_dir, "Group Statistics", "C_males.txt"), index=False, sep='\t')
women_df_C.to_csv(os.path.join(patients_dir, "Group Statistics", "C_females.txt"), index=False, sep='\t')

# %% Group (D)

men_file_paths = [os.path.join(patients_dir, file, "results", "D.txt") for file in men]
women_file_paths = [os.path.join(patients_dir, file, "results", "D.txt") for file in women]
selected_columns = ['occurrences']

events = pd.read_csv(men_file_paths[0], sep='\t', usecols=['event'])

men_df_D = merge_files(men_file_paths, selected_columns)
women_df_D = merge_files(women_file_paths, selected_columns)

columns = ['event', 'occurrences']

data_m = {'event': np.array(events['event']), 'occurences': np.array(men_df_D['occurrences'])}
data_w = {'event': np.array(events['event']), 'occurences': np.array(women_df_D['occurrences'])}

men_df_D = pd.DataFrame(data_m)
women_df_D = pd.DataFrame(data_w)

men_df_D.to_csv(os.path.join(patients_dir, "Group Statistics", "D_males.txt"), index=False, sep='\t')
women_df_D.to_csv(os.path.join(patients_dir, "Group Statistics", "D_females.txt"), index=False, sep='\t')

r0 = men_df_D.iloc[:5, 0].tolist()
r1 = men_df_D.iloc[:5, 1].tolist()
r2 = men_df_D.iloc[5:30, 0].tolist()
r3 = men_df_D.iloc[5:30, 1].tolist()
r4 = men_df_D.iloc[30:155, 0].tolist()
r5 = men_df_D.iloc[30:155, 1].tolist()

bar_plot(r0, r1, "Males", patients_dir, "Single", d=True)
bar_plot(r2, r3, "Males", patients_dir, "Double", hight=6, length=10, n=25, d=True)
bar_plot(r4, r5, "Males", patients_dir, "Tripple", hight=6, length=12, n=25, d=True)

r0 = women_df_D.iloc[:5, 0].tolist()
r1 = women_df_D.iloc[:5, 1].tolist()
r2 = women_df_D.iloc[5:30, 0].tolist()
r3 = women_df_D.iloc[5:30, 1].tolist()
r4 = women_df_D.iloc[30:155, 0].tolist()
r5 = women_df_D.iloc[30:155, 1].tolist()

bar_plot(r0, r1, "Females", patients_dir, "Single", d=True)
bar_plot(r2, r3, "Females", patients_dir, "Double", hight=6, length=10, n=25, d=True)
bar_plot(r4, r5, "Females", patients_dir, "Tripple", hight=6, length=12, n=25, d=True)

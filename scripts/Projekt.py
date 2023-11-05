# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 20:58:02 2022

@author: Jakub Węgrzynek
"""
# %% Importy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import csv

pd.options.mode.chained_assignment = None

# %% Ustawianie katalogu pracy
project_dir = os.path.join(os.getcwd(), "hypertension_project")
data_dir = os.path.join(project_dir, "data")
plot_dir = os.path.join(project_dir, "plots")
os.makedirs(data_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# %% Uzyskiwanie nazw plików oraz grupowanie ich
files_list = os.listdir(data_dir)

healthy = np.array([])
sick = np.array([])

for file_name in files_list:
    if file_name[4] == '1':
        healthy = np.append(healthy, file_name)
    elif file_name[4] == '2':
        sick = np.append(sick, file_name)

# %% WYBÓR PLIKU DO ANALIZY

string = ""
m = 1

for k, l in enumerate([healthy, sick]):
    if k == 0:
        string += "\nPacjenci BEZ nadciśnienia:\n"
        for i in range(len(l)):
            string += "\t{}. {}{}".format(i + 1, l[i], "\n")
            m += 1
    if k == 1:
        string += "\nPacjenci Z nadciśnieniem:\n"
        for i in range(len(l)):
            string += "\t{}. {}{}".format(m + i, l[i], "\n")

message = "\nWybierz plik wpisując jego numer: "

users_input = int(input("{}{}".format(string, message))) - 1

file = [*healthy, *sick][users_input]

print("{}---> Wybrany plik: {}".format("\n", file))

# Ładowanie danych (gdy w pliku pomiar jest rozpoczynany wiele razy, pomijam te przypadkowe)

try:
    df = pd.read_csv(os.path.join(data_dir, file), skiprows=[0, 1, 2, 3, 4], usecols=[0, 1, 2, 3],
                     names=['Time', 'R peak', 'Chest pos', 'SBP'], low_memory=False, dtype='float', sep="\t",
                     header=None)
except ValueError:
    df = pd.read_csv(os.path.join(data_dir, file), skiprows=[0, 1, 2, 3, 4], usecols=[0, 1, 2, 3],
                     names=['Time', 'R peak', 'Chest pos', 'SBP'], low_memory=False, sep="\t", header=None,
                     encoding='unicode_escape')
    li = np.array(df.iloc[:, 0])
    i, = np.where(li == "Interval=")
    df = pd.read_csv(os.path.join(data_dir, file), skiprows=(i[len(i) - 1] + 10), usecols=[0, 1, 2, 3],
                     names=['Time', 'R peak', 'Chest pos', 'SBP'], low_memory=False, dtype='float', sep="\t",
                     header=None, encoding='unicode_escape')

# %% Informacje o danych
print("---> Informacje o typach danych: ")
print(df.info())

print("\n---> Ilość pustych wierszy:")
print(df.isna().sum())

# %% Ładowanie danych do zmiennych

df = df.dropna()
times = df.iloc[:, 0]
R_peaks = df.iloc[:, 1]
chest = df.iloc[:, 2]
sbp = df.iloc[:, 3]
RR_data_raw = df.iloc[:, 0:2]

# %% Obróbka danych
# Utworzenie wektora zawierającego wartosci 1 - unoszenie klatki, -1 - opadanie klatki
n = len(chest) - 1
l = np.ones(n)

for i in range(n):
    if chest.iloc[i] < chest.iloc[i + 1]:
        l[i] = 1
    elif chest.iloc[i] > chest.iloc[i + 1]:
        l[i] = -1

# Sprawdzanie czasu trwania czasu wdechu i wydechu
n = len(l) - 1
since_change = 0
duration = []
for i in range(n):
    if l[i] == l[i + 1]:
        since_change += 1
    elif l[i] != l[i + 1]:
        duration.append([i, since_change])
        since_change = 0

# Usuwanie szumu (fałszywych miejsc gdzie wdech zmienia się na wydech i na odwrót)
for i in range(len(duration) - 1):
    if duration[i][1] < 750:
        for j in range(duration[i - 1][0], duration[i][0] + 1):
            l[j] = l[duration[i - 1][0]]

# Sprawdzanie czasu trwania wdechu i wydechu
n = len(l) - 1
since_change = 0
duration = []
for i in range(n):
    if l[i] == l[i + 1]:
        since_change += 1
    elif l[i] != l[i + 1]:
        duration.append([i, since_change])
        since_change = 0

# Tworzenie puntków gdzie klatka była najwyzej i najnizej
t = np.array(times)
c = np.array(chest)
peak_times = [t[dur[0]] for dur in duration]
peak_chest = [c[dur[0]] for dur in duration]

# RR
RR_data = RR_data_raw[RR_data_raw['R peak'] != 0]
RR_data['Time'] = RR_data['Time'] * 1000
R_time = np.array(RR_data.iloc[:, 0])
rr = np.array([int(abs(R_time[i] - R_time[i + 1])) for i in range(len(R_time) - 1)])

# Dane Poincare
rrn = rr[:len(rr) - 1]
rrn1 = rr[1:]

# %% Wykres (fala oddechowa)
fig, ax = plt.subplots()
fig.set_size_inches(45, 10)
ax.plot(times, chest)
ax.scatter(peak_times, peak_chest)
ax.set_title('Fala oddechowa {}'.format(file))
ax.set_xlabel('Czas')
ax.set_ylabel('Pozycja klatki piersiowej')
plt.savefig(os.path.join(plot_dir, 'Fala oddechowa {}.pdf'.format(file)))

# %% Wykres Poicare
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax.scatter(rrn, rrn1)
ax.set_title('Wykres Poincare {}'.format(file))
ax.set_xlabel('$RR_{n}$')
ax.set_ylabel('$RR_{n+1} [ms]$')

plt.savefig(os.path.join(plot_dir, 'Wykres Poincare {}.pdf'.format(file)))

# %% Histogram ciśnienia
fig, ax = plt.subplots()
fig.set_size_inches(10, 10)
ax.hist(sbp[sbp != 0])
ax.set_title('Histogram ciśnienia {}'.format(file))
ax.set_xlabel('Ciśnienie')
plt.savefig(os.path.join(plot_dir, 'Histogram ciśnienia {}.pdf'.format(file)))
# %% Wyznaczenie czasowych własności sygnału interwałów RR

# SDNN
sdnn = np.std(rr)
print("---> SDNN = {}".format(round(sdnn, 2)))

# RMSSD
n = len(rr)
s = np.sum(np.array([(rr[i + 1] - rr[i]) ** 2 for i in range(n - 1)]))
rmssd = np.sqrt(1 / n * s)
print("---> RMSSD = {}".format(round(rmssd, 2)))

# pNN50
nn50 = np.array([1 if abs(rr[i + 1] - rr[i]) > 50 else 0 for i in range(n - 1)])
pnn50 = np.sum(nn50) / len(nn50)
print("---> pNN50 = {}".format(round(pnn50, 2)))

# pNN20
nn20 = np.array([1 if abs(rr[i + 1] - rr[i]) > 20 else 0 for i in range(n - 1)])
pnn20 = np.sum(nn20) / len(nn20)
print("---> pNN20 = {}".format(round(pnn20, 2)))

# %% Wyznaczenie czasowych własności sygnału ciśnienia krwi
data_sbp = np.array(sbp)
data_sbp = data_sbp[data_sbp != 0]
mean_sbp = np.mean(data_sbp)

print("---> Średnie ciśnienie krwi: {}".format(round(mean_sbp, 2)))

# %% Ilość wystapień pików R przy wdechu i wydechu
R_peaks_map = l * np.array(R_peaks[1::])
unique, counts = np.unique(R_peaks_map[R_peaks_map != 0], return_counts=True)
dict(zip(unique, counts))

# %% Ilość przyspieszeń "a" i zwolnień "d" rytmu serca przy wdechu i wydechu
ac_dc = np.array([])

for i in range(len(rr) - 1):
    if rr[i] < rr[i + 1]:
        ac_dc = np.append(ac_dc, 'a')
    elif rr[i] > rr[i + 1]:
        ac_dc = np.append(ac_dc, 'd')
    else:
        ac_dc = np.append(ac_dc, 0)

R_peak_breath = R_peaks_map[R_peaks_map != 0]
R_peak_breath = R_peak_breath[2::]

RR_breathe = {"a b-in": 0,
              "a b-out": 0,
              "d b-in": 0,
              "d b-out": 0}

for i in range(len(ac_dc)):
    if ac_dc[i] == 'a' and R_peak_breath[i] == 1:
        RR_breathe["a b-in"] += 1
    elif ac_dc[i] == 'a' and R_peak_breath[i] == -1:
        RR_breathe["a b-out"] += 1
    elif ac_dc[i] == 'd' and R_peak_breath[i] == 1:
        RR_breathe["d b-in"] += 1
    elif ac_dc[i] == 'd' and R_peak_breath[i] == -1:
        RR_breathe["d b-out"] += 1

print("---> Przyspieszenia rytmu serca przy wdechu: {} ({}%)".format(RR_breathe["a b-in"], round(
    100 * RR_breathe["a b-in"] / sum(RR_breathe.values()), 2)))
print("---> Przyspieszenia rytmu serca przy wydechu: {} ({}%)".format(RR_breathe["a b-out"], round(
    100 * RR_breathe["a b-out"] / sum(RR_breathe.values()), 2)))
print("---> Zwolnienia rytmu serca przy wdechu: {} ({}%)".format(RR_breathe["d b-in"], round(
    100 * RR_breathe["d b-in"] / sum(RR_breathe.values()), 2)))
print("---> Zwolnienia rytmu serca przy wydechu: {} ({}%)".format(RR_breathe["d b-out"], round(
    100 * RR_breathe["d b-out"] / sum(RR_breathe.values()), 2)))

# %% Ilość wzrostów i spadkow ciśnienia skurczowego SBP przy wdechu i wydechu
sbp_breath_map = l * np.array([1 if bp != 0 else 0 for bp in sbp])[1::]
sbp_breath_map = sbp_breath_map[sbp_breath_map != 0][1::]

sbp_values = np.array(sbp[sbp != 0])

sbp_changes = np.array([])

for i in range(len(sbp_values) - 1):
    if sbp_values[i] < sbp_values[i + 1]:
        sbp_changes = np.append(sbp_changes, "i")
    elif sbp_values[i] > sbp_values[i + 1]:
        sbp_changes = np.append(sbp_changes, "d")
    else:
        sbp_changes = np.append(sbp_changes, 0)

sbp_breathe = {"i b-in": 0,
               "i b-out": 0,
               "d b-in": 0,
               "d b-out": 0}

for i in range(len(sbp_changes)):
    if sbp_changes[i] == 'i' and sbp_breath_map[i] == 1:
        sbp_breathe["i b-in"] += 1
    elif sbp_changes[i] == 'i' and sbp_breath_map[i] == -1:
        sbp_breathe["i b-out"] += 1
    elif sbp_changes[i] == 'd' and sbp_breath_map[i] == 1:
        sbp_breathe["d b-in"] += 1
    elif sbp_changes[i] == 'd' and sbp_breath_map[i] == -1:
        sbp_breathe["d b-out"] += 1

print("---> Wzrosty ciśnienia przy wdechu: {} ({}%)".format(sbp_breathe["i b-in"], round(
    100 * sbp_breathe["i b-in"] / sum(sbp_breathe.values()), 2)))
print("---> Wzrosty ciśnienia przy wydechu: {} ({}%)".format(sbp_breathe["i b-out"], round(
    100 * sbp_breathe["i b-out"] / sum(sbp_breathe.values()), 2)))
print("---> Spadki ciśnienia przy wdechu: {} ({}%)".format(sbp_breathe["d b-in"], round(
    100 * sbp_breathe["d b-in"] / sum(sbp_breathe.values()), 2)))
print("---> Spadki ciśnienia przy wydechu: {} ({}%)".format(sbp_breathe["d b-out"], round(
    100 * sbp_breathe["d b-out"] / sum(sbp_breathe.values()), 2)))

# %% WYLICZANIE STATYSTYKI GRUPOWEJ

files_list = os.listdir(data_dir)

healthy = np.array([])
sick = np.array([])

# Zmienne - zdrowi
sdnn_h = np.array([])
rmssd_h = np.array([])
pnn50_h = np.array([])
pnn20_h = np.array([])

sbp_breathe_h = {"i b-in": 0,
                 "i b-out": 0,
                 "d b-in": 0,
                 "d b-out": 0}

RR_breathe_h = {"a b-in": 0,
                "a b-out": 0,
                "d b-in": 0,
                "d b-out": 0}

# Zmienne - chorzy
sdnn_s = np.array([])
rmssd_s = np.array([])
pnn50_s = np.array([])
pnn20_s = np.array([])

sbp_breathe_s = {"i b-in": 0,
                 "i b-out": 0,
                 "d b-in": 0,
                 "d b-out": 0}

RR_breathe_s = {"a b-in": 0,
                "a b-out": 0,
                "d b-in": 0,
                "d b-out": 0}

# Podział plików na zdrowych i chorych
for file_name in files_list:
    if file_name[4] == '1':
        healthy = np.append(healthy, file_name)
    elif file_name[4] == '2':
        sick = np.append(sick, file_name)

# %% Program - zdrowi
for patient in healthy:
    try:
        df = pd.read_csv(os.path.join(data_dir, patient), skiprows=[0, 1, 2, 3, 4], usecols=[0, 1, 2, 3],
                         names=['Time', 'R peak', 'Chest pos', 'SBP'], low_memory=False, dtype='float', sep="\t",
                         header=None)
    except ValueError:
        df = pd.read_csv(os.path.join(data_dir, patient), skiprows=[0, 1, 2, 3, 4], usecols=[0, 1, 2, 3],
                         names=['Time', 'R peak', 'Chest pos', 'SBP'], low_memory=False, sep="\t", header=None,
                         encoding='unicode_escape')
        li = np.array(df.iloc[:, 0])
        i, = np.where(li == "Interval=")
        df = pd.read_csv(os.path.join(data_dir, patient), skiprows=(i[len(i) - 1] + 10), usecols=[0, 1, 2, 3],
                         names=['Time', 'R peak', 'Chest pos', 'SBP'], low_memory=False, dtype='float', sep="\t",
                         header=None, encoding='unicode_escape')

    df = df.dropna()  # usuwamy puste wiersze
    times = df.iloc[:, 0]
    R_peaks = df.iloc[:, 1]
    chest = df.iloc[:, 2]
    sbp = df.iloc[:, 3]
    RR_data_raw = df.iloc[:, 0:2]

    n = len(chest) - 1
    l = np.ones(n)

    for i in range(n):
        if chest.iloc[i] < chest.iloc[i + 1]:
            l[i] = 1
        elif chest.iloc[i] > chest.iloc[i + 1]:
            l[i] = -1

    # Sprawdzanie czasu trwania czasu wdechu i wydechu
    n = len(l) - 1
    since_change = 0
    duration = []
    for i in range(n):
        if l[i] == l[i + 1]:
            since_change += 1
        elif l[i] != l[i + 1]:
            duration.append([i, since_change])
            since_change = 0

    # Usuwanie szumu (falszywych miejsc gdzie wdech zmienia się na wydech i na odwrót)
    for i in range(len(duration) - 1):
        if duration[i][1] < 750:
            for j in range(duration[i - 1][0], duration[i][0] + 1):
                l[j] = l[duration[i - 1][0]]

    # Sprawdzanie czasu trwania wdechu i wydechu
    n = len(l) - 1
    since_change = 0
    duration = []
    for i in range(n):
        if l[i] == l[i + 1]:
            since_change += 1
        elif l[i] != l[i + 1]:
            duration.append([i, since_change])
            since_change = 0

    # Wyznaczenie czasowych wlasnosci sygnalu interwalow RR
    # RR
    RR_data = RR_data_raw[RR_data_raw['R peak'] != 0]
    RR_data['Time'] = RR_data['Time'] * 1000
    R_time = np.array(RR_data.iloc[:, 0])

    rr = np.array([int(abs(R_time[i] - R_time[i + 1])) for i in range(len(R_time) - 1)])

    # SDNN
    sdnn_h = np.append(sdnn_h, np.std(rr))

    # RMSSD
    n = len(rr)
    s = np.sum(np.array([(rr[i + 1] - rr[i]) ** 2 for i in range(n - 1)]))
    rmssd_h = np.append(rmssd_h, np.sqrt(1 / n * s))

    # pNN50
    nn50 = np.array([1 if abs(rr[i + 1] - rr[i]) > 50 else 0 for i in range(n - 1)])
    pnn50_h = np.append(pnn50_h, np.sum(nn50) / len(nn50))

    # pNN20
    nn20 = np.array([1 if abs(rr[i + 1] - rr[i]) > 20 else 0 for i in range(n - 1)])
    pnn20_h = np.append(pnn20_h, np.sum(nn20) / len(nn20))

    # Wyznaczenie czasowych własnosci sygnalu cisnienia krwi
    data_sbp_h = np.array(sbp)
    data_sbp_h = data_sbp_h[data_sbp_h != 0]
    mean_sbp_h = np.mean(data_sbp_h)

    # Dane Poincare
    rrn = rr[:len(rr) - 1]
    rrn1 = rr[1:]

    # Poszukiwanie wzorcow
    # Ilosc wystapien pikow R przy wdechu i wydechu
    R_peaks_map = l * np.array(R_peaks[1::])
    unique, counts = np.unique(R_peaks_map[R_peaks_map != 0], return_counts=True)
    dict(zip(unique, counts))

    # Ilosc przyspieszen a i zwolnien d rytmu serca przy wdechu i wydechu
    ac_dc = np.array([])

    for i in range(len(rr) - 1):
        if rr[i] < rr[i + 1]:
            ac_dc = np.append(ac_dc, 'a')
        elif rr[i] > rr[i + 1]:
            ac_dc = np.append(ac_dc, 'd')
        else:
            ac_dc = np.append(ac_dc, 0)

    R_peak_breath = R_peaks_map[R_peaks_map != 0]
    R_peak_breath = R_peak_breath[2::]

    for i in range(len(ac_dc)):
        if ac_dc[i] == 'a' and R_peak_breath[i] == 1:
            RR_breathe_h["a b-in"] += 1
        elif ac_dc[i] == 'a' and R_peak_breath[i] == -1:
            RR_breathe_h["a b-out"] += 1
        elif ac_dc[i] == 'd' and R_peak_breath[i] == 1:
            RR_breathe_h["d b-in"] += 1
        elif ac_dc[i] == 'd' and R_peak_breath[i] == -1:
            RR_breathe_h["d b-out"] += 1

    # Ilosc wzrostow i spadkow cisnienia skurczowego SBP przy wdechu i wydechu
    sbp_breath_map = l * np.array([1 if bp != 0 else 0 for bp in sbp])[1::]
    sbp_breath_map = sbp_breath_map[sbp_breath_map != 0][1::]

    sbp_values = np.array(sbp[sbp != 0])
    sbp_changes = np.array([])

    for i in range(len(sbp_values) - 1):
        if sbp_values[i] < sbp_values[i + 1]:
            sbp_changes = np.append(sbp_changes, "i")
        elif sbp_values[i] > sbp_values[i + 1]:
            sbp_changes = np.append(sbp_changes, "d")
        else:
            sbp_changes = np.append(sbp_changes, 0)

    for i in range(len(sbp_changes)):
        if sbp_changes[i] == 'i' and sbp_breath_map[i] == 1:
            sbp_breathe_h["i b-in"] += 1
        elif sbp_changes[i] == 'i' and sbp_breath_map[i] == -1:
            sbp_breathe_h["i b-out"] += 1
        elif sbp_changes[i] == 'd' and sbp_breath_map[i] == 1:
            sbp_breathe_h["d b-in"] += 1
        elif sbp_changes[i] == 'd' and sbp_breath_map[i] == -1:
            sbp_breathe_h["d b-out"] += 1

    print("Postęp: {}/{}".format(np.where(healthy == patient)[0][0] + 1, len(healthy)))

# %% Program - chorzy
for patient in sick:
    try:
        df = pd.read_csv(os.path.join(data_dir, patient), skiprows=[0, 1, 2, 3, 4], usecols=[0, 1, 2, 3],
                         names=['Time', 'R peak', 'Chest pos', 'SBP'], low_memory=False, dtype='float', sep="\t",
                         header=None)
    except ValueError:
        df = pd.read_csv(os.path.join(data_dir, patient), skiprows=[0, 1, 2, 3, 4], usecols=[0, 1, 2, 3],
                         names=['Time', 'R peak', 'Chest pos', 'SBP'], low_memory=False, sep="\t", header=None,
                         encoding='unicode_escape')
        li = np.array(df.iloc[:, 0])
        i, = np.where(li == "Interval=")
        df = pd.read_csv(os.path.join(data_dir, patient), skiprows=(i[len(i) - 1] + 10), usecols=[0, 1, 2, 3],
                         names=['Time', 'R peak', 'Chest pos', 'SBP'], low_memory=False, dtype='float', sep="\t",
                         header=None, encoding='unicode_escape')

    df = df.dropna()  # usuwamy puste wiersze
    times = df.iloc[:, 0]
    R_peaks = df.iloc[:, 1]
    chest = df.iloc[:, 2]
    sbp = df.iloc[:, 3]
    RR_data_raw = df.iloc[:, 0:2]

    n = len(chest) - 1
    l = np.ones(n)

    for i in range(n):
        if chest.iloc[i] < chest.iloc[i + 1]:
            l[i] = 1
        elif chest.iloc[i] > chest.iloc[i + 1]:
            l[i] = -1

    # Sprawdzanie czasu trwania czasu wdechu i wydechu
    n = len(l) - 1
    since_change = 0
    duration = []
    for i in range(n):
        if l[i] == l[i + 1]:
            since_change += 1
        elif l[i] != l[i + 1]:
            duration.append([i, since_change])
            since_change = 0

    # Usuwanie szumu (falszywych miejsc gdzie wdech zmienia się na wydech i na odwrót)
    for i in range(len(duration) - 1):
        if duration[i][1] < 750:
            for j in range(duration[i - 1][0], duration[i][0] + 1):
                l[j] = l[duration[i - 1][0]]

    # Sprawdzanie czasu trwania wdechu i wydechu
    n = len(l) - 1
    since_change = 0
    duration = []
    for i in range(n):
        if l[i] == l[i + 1]:
            since_change += 1
        elif l[i] != l[i + 1]:
            duration.append([i, since_change])
            since_change = 0

    # Wyznaczenie czasowych wlasnosci sygnalu interwalow RR
    # RR
    RR_data = RR_data_raw[RR_data_raw['R peak'] != 0]
    RR_data['Time'] = RR_data['Time'] * 1000
    R_time = np.array(RR_data.iloc[:, 0])

    rr = np.array([int(abs(R_time[i] - R_time[i + 1])) for i in range(len(R_time) - 1)])

    # SDNN
    sdnn_s = np.append(sdnn_s, np.std(rr))

    # RMSSD
    n = len(rr)
    s = np.sum(np.array([(rr[i + 1] - rr[i]) ** 2 for i in range(n - 1)]))
    rmssd_s = np.append(rmssd_s, np.sqrt(1 / n * s))

    # pNN50
    nn50 = np.array([1 if abs(rr[i + 1] - rr[i]) > 50 else 0 for i in range(n - 1)])
    pnn50_s = np.append(pnn50_s, np.sum(nn50) / len(nn50))

    # pNN20
    nn20 = np.array([1 if abs(rr[i + 1] - rr[i]) > 20 else 0 for i in range(n - 1)])
    pnn20_s = np.append(pnn20_s, np.sum(nn20) / len(nn20))

    # Dane Poincare
    rrn = rr[:len(rr) - 1]
    rrn1 = rr[1:]

    # Wyznaczenie czasowych własnosci sygnalu cisnienia krwi
    data_sbp_s = np.array(sbp)
    data_sbp_s = data_sbp_s[data_sbp_s != 0]
    mean_sbp_s = np.mean(data_sbp_s)

    # Poszukiwanie wzorcow
    # Ilosc wystapien pikow R przy wdechu i wydechu
    R_peaks_map = l * np.array(R_peaks[1::])
    unique, counts = np.unique(R_peaks_map[R_peaks_map != 0], return_counts=True)
    dict(zip(unique, counts))

    # Ilosc przyspieszen a i zwolnien d rytmu serca przy wdechu i wydechu
    ac_dc = np.array([])

    for i in range(len(rr) - 1):
        if rr[i] < rr[i + 1]:
            ac_dc = np.append(ac_dc, 'a')
        elif rr[i] > rr[i + 1]:
            ac_dc = np.append(ac_dc, 'd')
        else:
            ac_dc = np.append(ac_dc, 0)

    R_peak_breath = R_peaks_map[R_peaks_map != 0]
    R_peak_breath = R_peak_breath[2::]

    for i in range(len(ac_dc)):
        if ac_dc[i] == 'a' and R_peak_breath[i] == 1:
            RR_breathe_s["a b-in"] += 1
        elif ac_dc[i] == 'a' and R_peak_breath[i] == -1:
            RR_breathe_s["a b-out"] += 1
        elif ac_dc[i] == 'd' and R_peak_breath[i] == 1:
            RR_breathe_s["d b-in"] += 1
        elif ac_dc[i] == 'd' and R_peak_breath[i] == -1:
            RR_breathe_s["d b-out"] += 1

    # Ilosc wzrostow i spadkow cisnienia skurczowego SBP przy wdechu i wydechu
    sbp_breath_map = l * np.array([1 if bp != 0 else 0 for bp in sbp])[1::]
    sbp_breath_map = sbp_breath_map[sbp_breath_map != 0][1::]

    sbp_values = np.array(sbp[sbp != 0])
    sbp_changes = np.array([])

    for i in range(len(sbp_values) - 1):
        if sbp_values[i] < sbp_values[i + 1]:
            sbp_changes = np.append(sbp_changes, "i")
        elif sbp_values[i] > sbp_values[i + 1]:
            sbp_changes = np.append(sbp_changes, "d")
        else:
            sbp_changes = np.append(sbp_changes, 0)

    for i in range(len(sbp_changes)):
        if sbp_changes[i] == 'i' and sbp_breath_map[i] == 1:
            sbp_breathe_s["i b-in"] += 1
        elif sbp_changes[i] == 'i' and sbp_breath_map[i] == -1:
            sbp_breathe_s["i b-out"] += 1
        elif sbp_changes[i] == 'd' and sbp_breath_map[i] == 1:
            sbp_breathe_s["d b-in"] += 1
        elif sbp_changes[i] == 'd' and sbp_breath_map[i] == -1:
            sbp_breathe_s["d b-out"] += 1

    print("Postęp: {}/{}".format(np.where(sick == patient)[0][0] + 1, len(sick)))

# %% Podsumowanie zdrowi
print("PODSUMOWANIE - OSOBY Z ZDROWE")
print("---> Średnia wartość SDNN = {}".format(round(np.mean(sdnn_h), 2)))
print("---> Średnia wartość RMSSD = {}".format(round(np.mean(rmssd_h), 2)))
print("---> Średnia wartość pNN50 = {}".format(round(np.mean(pnn50_h), 2)))
print("---> Średnia wartość pNN20 = {}".format(round(np.mean(pnn20_h), 2)))

print("---> Średnie ciśnienie krwi: {}".format(round(np.mean(mean_sbp_h), 2)))

print("---> Przyspieszenia rytmu serca przy wdechu: {} ({}%)".format(RR_breathe_h["a b-in"], round(
    100 * RR_breathe_h["a b-in"] / sum(RR_breathe_h.values()), 2)))
print("---> Przyspieszenia rytmu serca przy wydechu: {} ({}%)".format(RR_breathe_h["a b-out"], round(
    100 * RR_breathe_h["a b-out"] / sum(RR_breathe_h.values()), 2)))
print("---> Zwolnienia rytmu serca przy wdechu: {} ({}%)".format(RR_breathe_h["d b-in"], round(
    100 * RR_breathe_h["d b-in"] / sum(RR_breathe_h.values()), 2)))
print("---> Zwolnienia rytmu serca przy wydechu: {} ({}%)".format(RR_breathe_h["d b-out"], round(
    100 * RR_breathe_h["d b-out"] / sum(RR_breathe_h.values()), 2)))

print("---> Wzrosty ciśnienia przy wdechu: {} ({}%)".format(sbp_breathe_h["i b-in"], round(
    100 * sbp_breathe_h["i b-in"] / sum(sbp_breathe_h.values()), 2)))
print("---> Wzrosty ciśnienia przy wydechu: {} ({}%)".format(sbp_breathe_h["i b-out"], round(
    100 * sbp_breathe_h["i b-out"] / sum(sbp_breathe_h.values()), 2)))
print("---> Spadki ciśnienia przy wdechu: {} ({}%)".format(sbp_breathe_h["d b-in"], round(
    100 * sbp_breathe_h["d b-in"] / sum(sbp_breathe_h.values()), 2)))
print("---> Spadki ciśnienia przy wydechu: {} ({}%)".format(sbp_breathe_h["d b-out"], round(
    100 * sbp_breathe_h["d b-out"] / sum(sbp_breathe_h.values()), 2)))

# %% Podsumowanie chorzy
print("PODSUMOWANIE - OSOBY Z NADCIŚNIENIEM")
print("---> Średnia wartość SDNN = {}".format(round(np.mean(sdnn_s), 2)))
print("---> Średnia wartość RMSSD = {}".format(round(np.mean(rmssd_s), 2)))
print("---> Średnia wartość pNN50 = {}".format(round(np.mean(pnn50_s), 2)))
print("---> Średnia wartość pNN20 = {}".format(round(np.mean(pnn20_s), 2)))

print("---> Średnie ciśnienie krwi: {}".format(round(np.mean(mean_sbp_s), 2)))

print("---> Przyspieszenia rytmu serca przy wdechu: {} ({}%)".format(RR_breathe_s["a b-in"], round(
    100 * RR_breathe_s["a b-in"] / sum(RR_breathe_s.values()), 2)))
print("---> Przyspieszenia rytmu serca przy wydechu: {} ({}%)".format(RR_breathe_s["a b-out"], round(
    100 * RR_breathe_s["a b-out"] / sum(RR_breathe_s.values()), 2)))
print("---> Zwolnienia rytmu serca przy wdechu: {} ({}%)".format(RR_breathe_s["d b-in"], round(
    100 * RR_breathe_s["d b-in"] / sum(RR_breathe_s.values()), 2)))
print("---> Zwolnienia rytmu serca przy wydechu: {} ({}%)".format(RR_breathe_s["d b-out"], round(
    100 * RR_breathe_s["d b-out"] / sum(RR_breathe_s.values()), 2)))

print("---> Wzrosty ciśnienia przy wdechu: {} ({}%)".format(sbp_breathe_s["i b-in"], round(
    100 * sbp_breathe_s["i b-in"] / sum(sbp_breathe_s.values()), 2)))
print("---> Wzrosty ciśnienia przy wydechu: {} ({}%)".format(sbp_breathe_s["i b-out"], round(
    100 * sbp_breathe_s["i b-out"] / sum(sbp_breathe_s.values()), 2)))
print("---> Spadki ciśnienia przy wdechu: {} ({}%)".format(sbp_breathe_s["d b-in"], round(
    100 * sbp_breathe_s["d b-in"] / sum(sbp_breathe_s.values()), 2)))
print("---> Spadki ciśnienia przy wydechu: {} ({}%)".format(sbp_breathe_s["d b-out"], round(
    100 * sbp_breathe_s["d b-out"] / sum(sbp_breathe_s.values()), 2)))

# %% Tworzenie pliku csv z danymi

header = [' ', 'Zdrowi', 'Chorzy']
data = [['Średnia wartość SDNN', round(np.mean(sdnn_h), 2), round(np.mean(sdnn_s), 2)],
        ['Średnia wartość RMSSD', round(np.mean(rmssd_h), 2), round(np.mean(rmssd_s), 2)],
        ['Średnia wartość pNN50', round(np.mean(pnn50_h), 2), round(np.mean(pnn50_s), 2)],
        ['Średnia wartość pNN20', round(np.mean(pnn20_h), 2), round(np.mean(pnn20_s), 2)],
        ['Średnie ciśnienie krwi', round(np.mean(mean_sbp_h), 2), round(np.mean(mean_sbp_s), 2)],
        ['Przyspieszenia rytmu serca przy wdechu',
         str(round(100 * RR_breathe_h["a b-in"] / sum(RR_breathe_h.values()), 2)) + '%',
         str(round(100 * RR_breathe_s["a b-in"] / sum(RR_breathe_s.values()), 2)) + '%'],
        ['Przyspieszenia rytmu serca przy wydechu',
         str(round(100 * RR_breathe_h["a b-out"] / sum(RR_breathe_h.values()), 2)) + '%',
         str(round(100 * RR_breathe_s["a b-out"] / sum(RR_breathe_s.values()), 2)) + '%'],
        ['Zwolnienia rytmu serca przy wdechu',
         str(round(100 * RR_breathe_h["d b-in"] / sum(RR_breathe_h.values()), 2)) + '%',
         str(round(100 * RR_breathe_s["d b-in"] / sum(RR_breathe_s.values()), 2)) + '%'],
        ['Zwolnienia rytmu serca przy wydechu',
         str(round(100 * RR_breathe_h["d b-out"] / sum(RR_breathe_h.values()), 2)) + '%',
         str(round(100 * RR_breathe_s["d b-out"] / sum(RR_breathe_s.values()), 2)) + '%'],
        ['Wzrosty ciśnienia przy wdechu',
         str(round(100 * sbp_breathe_h["i b-in"] / sum(sbp_breathe_h.values()), 2)) + '%',
         str(round(100 * sbp_breathe_s["i b-in"] / sum(sbp_breathe_s.values()), 2)) + '%'],
        ['Wzrosty ciśnienia przy wydechu',
         str(round(100 * sbp_breathe_h["i b-out"] / sum(sbp_breathe_h.values()), 2)) + '%',
         str(round(100 * sbp_breathe_s["i b-out"] / sum(sbp_breathe_s.values()), 2)) + '%'],
        ['Spadki ciśnienia przy wdechu',
         str(round(100 * sbp_breathe_h["d b-in"] / sum(sbp_breathe_h.values()), 2)) + '%',
         str(round(100 * sbp_breathe_s["d b-in"] / sum(sbp_breathe_s.values()), 2)) + '%'],
        ['Spadki ciśnienia przy wydechu',
         str(round(100 * sbp_breathe_h["d b-out"] / sum(sbp_breathe_h.values()), 2)) + '%',
         str(round(100 * sbp_breathe_s["d b-out"] / sum(sbp_breathe_s.values()), 2)) + '%']]

with open(os.path.join(project_dir, 'Statystyka zbiorcza.csv'), 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)

dane_zbiorcze = pd.read_csv(os.path.join(project_dir, 'Statystyka zbiorcza.csv'))

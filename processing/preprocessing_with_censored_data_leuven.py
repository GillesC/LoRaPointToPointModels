"""
    ____  ____      _    __  __  ____ ___
   |  _ \|  _ \    / \  |  \/  |/ ___/ _ \
   | | | | |_) |  / _ \ | |\/| | |  | | | |
   | |_| |  _ <  / ___ \| |  | | |__| |_| |
   |____/|_| \_\/_/   \_\_|  |_|\____\___/
                             research group
                               dramco.be/

    KU Leuven - Technology Campus Gent,
    Gebroeders De Smetstraat 1,
    B-9000 Gent, Belgium

           File: preprocessing.py
        Created: 2018-10-30
         Author: Gilles Callebaut
    Description: Preprocessing raw data coming from Arduino Receiver
                 - concatenates different raw csv files
                 - filters data (e.g., values withouth GPS)
                 - removes unneeded data
                 - sorts by time
                 - distance to transmitters (CENTER)
                 - PL
                Stores a pickle file in ../result
"""
import glob
import json
import os
import pickle
import datetime

import numpy as np

import pandas as pd
import util as util

import warnings
import matplotlib.pyplot as plt

warnings.simplefilter(action='ignore', category=FutureWarning)

HEADER = ["time", "sat", "satValid", "hdopVal", "hdopValid", "lat", "lon", "locValid",
          "age",
          "ageValid", "alt", "altValid", "rssi", "snr", "counter", "isPacket"]

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))
output_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result', "Leuven"))

measurement = "leuven"

path_to_measurement = os.path.join(path_to_measurements, measurement)
all_files = glob.glob(os.path.join(path_to_measurement, "*.csv"))

print("--------------------- PREPROCESSING {} ---------------------".format(measurement))
size = 0
df_from_each_file = []
for idx, file in enumerate(all_files):
    size += os.path.getsize(file)

    df = pd.read_csv(file, sep=',', header=None, names=None)
    df.columns = HEADER
    df['file'] = os.path.basename(file)
    df_from_each_file.append(df)

df = pd.concat(df_from_each_file, ignore_index=True, sort=False)
print(" Reading {0} files {1:.2f} kB".format(len(all_files), size / 1024))
total_rows = df.shape[0]

df = util.filter(df)

with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)

    CENTER = config[measurement]["center"]

    util.addDistanceTo(df, CENTER)
    df = df[df['distance'] < 20 * 1000]
    df = df[df['distance'] > 1]
    util.addPathLossTo(df)

    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by=['time'], inplace=True)

    df_copy = df.copy()

    df_files = df_copy.groupby(['file'])

    df_packets = []

    distances_expected = []

    for name, df_file in df_files:
        print(name)
        tmp_counter = df_file.iloc[0].counter
        start_time_file = 0
        first_packet = False
        df_packets_per_file = []
        df_expected_packets_per_file = []

        for index, row in df_file.iterrows():
            if int(row.counter) != int(tmp_counter):
                if not first_packet:
                    first_packet = True
                    start_time_file = row["time"]

                row["isPacket"] = 1
                df_packets.append(row)
                df_packets_per_file.append(row)
                # print(row)
                tmp_counter = row.counter
        end_time_file = df_packets_per_file[-1]["time"]

        print(start_time_file)
        print(end_time_file)

        df_file.index = df_file["time"]
        df_file = df_file[~df_file.index.duplicated(keep='first')]
        # now that we have the beginning and end of the registration time
        # we can search for the distances of the terminal each 8 seconds
        # hence, when we expect a packet
        next_expected_packet_time = start_time_file
        prev_expected_packet_time = start_time_file
        while next_expected_packet_time < end_time_file:
            next_expected_packet_time = prev_expected_packet_time + datetime.timedelta(seconds=8)
            location_nearest = df_file.index.get_loc(next_expected_packet_time, method='nearest')
            df_expected_packets_per_file.append(df_file.iloc[location_nearest])
            prev_expected_packet_time = next_expected_packet_time

        print(len(df_expected_packets_per_file))
        distances_expected.extend([row["distance"] for row in df_expected_packets_per_file])

    distances_packets = [row["distance"] for row in df_packets]

    # plot histogram of distances where we have packet
    # and where we expeced a packet

    bin_edges = np.linspace(0, np.max(distances_expected), 10, endpoint=True)

    hist_val_expected, bin_edges = np.histogram(distances_expected, bins=bin_edges)
    hist_val_packets, _ = np.histogram(distances_packets, bins=bin_edges)

    per = 1 - hist_val_packets / hist_val_expected
    plt.scatter(x=bin_edges[1:], y=per)
    # plt.xscale("log")
    plt.show()

    print(per)
    print(bin_edges)

    for x, y in zip(bin_edges[1:], per):
        print(F"{x} {y}\\\\")

    df = pd.DataFrame(df_packets, columns=df_copy.columns)

    # residuals_ml = np.abs(df.pl_db - pl_ml)
    # residuals_ols = np.abs(df.pl_db - pl_ols)
    #
    # plt.subplots()
    # sns.scatterplot(x="distance", y=residuals_ml, s=100, color=".2", marker="+", data=df)
    # sns.scatterplot(x="distance", y=residuals_ols, s=100, color="red", marker="+", data=df)
    # df = pd.concat(df_packets, ignore_index=True, sort=False)

    # rmse_ml = np.sqrt(np.sum(residuals_ml ** 2) / residuals_ml.size)
    # print(rmse_ml)
    #
    # mae_ml = np.sum(residuals_ml) / residuals_ml.size
    # print(mae_ml)
    #
    # print(F"Std: {np.std(residuals_ml)}")
    #
    # r_squared_ml = metrics.r2_score(df.pl_db, pl_ml)
    # print(F"R-squared ML: {r_squared_ml}")
    #
    # r_ml = (np.sum((df.pl_db - np.mean(df.pl_db)) ** 2) - np.sum((pl_ml - df.pl_db) ** 2)) / (
    #     np.sum((df.pl_db - np.mean(df.pl_db)) ** 2))
    # print(F"R ML: {r_ml}")
    #
    # print(np.max(df.distance))
    #
    # ############################################################################
    #
    # rmse_ols = np.sqrt(np.sum(residuals_ols ** 2) / residuals_ols.size)
    # print(rmse_ols)
    #
    # mae_ols = np.sum(residuals_ols) / residuals_ols.size
    # print(mae_ols)
    #
    # print(F"Std: {np.std(residuals_ols)}")

    pickle.dump(df, open(os.path.join(output_path, "leuven-data.pkl"), "wb"))

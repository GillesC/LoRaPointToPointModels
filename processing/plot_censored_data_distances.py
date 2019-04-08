r"""
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

           File:
        Created: 2018-10-30
         Author: Gilles Callebaut
    Description:
"""
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))

input_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))
input_file_name = "processed_data_with_censored_data.pkl"
input_file_path = os.path.join(input_path, input_file_name)

warnings.simplefilter("ignore")

with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)
    measurements = config["measurements"]
    data = pd.read_pickle(input_file_path)

    for measurement in measurements:
        print(F"--------------------- PATH LOSS MODEL {measurement} ---------------------")

        df = data[measurement]["data"]
        censored_packets_mask = data[measurement]["censored_packets_mask"]
        uncensored_packets_mask = np.invert(censored_packets_mask)

        df_censored = df.loc[censored_packets_mask]
        df_uncensored = df.loc[uncensored_packets_mask]

        d_censored = df_censored["distance"].values
        d_total = df["distance"].values
        d_uncensored= df_uncensored["distance"].values

        num_bin = int((d_uncensored.max() - d_uncensored.min())/10)
        hist_total, bin_edges = np.histogram(df["distance"].values, bins=num_bin)
        hist_censored, *_= np.histogram(d_censored, bins=bin_edges)
        hist_uncensored, *_ = np.histogram(d_uncensored, bins=bin_edges)

        plt.plot(bin_edges[:-1], hist_censored)
        plt.plot(bin_edges[:-1], hist_uncensored)

        #plt.plot(bin_edges[:-1], (hist_censored / hist_total)*100)
        plt.show(block=True)

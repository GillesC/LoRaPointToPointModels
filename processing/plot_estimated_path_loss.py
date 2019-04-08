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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.constants

import censored_ml as model
from get_weights import get_weights

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))

input_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))

input_file_measurements_path = os.path.join(input_path, "processed_data_with_censored_data.pkl")
input_file_estimated_pl_path = os.path.join(input_path, "estimated_path_loss.pkl")

output_file = os.path.abspath(os.path.join(input_path, "estimated_path_loss.pkl"))

with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)
    measurements = config["measurements"]
    data = pd.read_pickle(input_file_measurements_path)

    path_loss_estimates = pd.read_pickle(input_file_estimated_pl_path)

    for measurement in measurements:
        print(F"--------------------- PATH LOSS MODEL {measurement} ---------------------")
        fig = plt.figure(figsize=(4, 3))
        plt.xscale('log')

        censored_packets_mask = data[measurement]["censored_packets_mask"]

        df = path_loss_estimates.loc[path_loss_estimates['Measurement'] == measurement]

        d_all = df["distance"].values
        pld_all = df["pl_db"].values

        plt.scatter(d_all, pld_all, marker='x', label="Measured Path Loss", s=1, c='0.75')

        for ix, row in df.iterrow():
            plt.plot(row['Distances'], row['PLd'], label=F"{row['Weight']}{row['Std']}{row['Model']}")

        plt.xlabel(r'Log distance (m)')
        plt.ylabel(r'Path Loss (dB)')

        plt.show()

        print(F"Done for {measurement}")

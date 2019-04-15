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
import scipy.constants

import regression_models as model
from get_weights import get_weights

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))

input_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))

input_file_measurements_path = os.path.join(input_path, "processed_data_with_censored_data.pkl")
input_file_estimated_pl_path = os.path.join(input_path, "estimated_path_loss.pkl")

white_list = np.array(["Single Slope"])
black_list = np.array(["distant_dependent"])
ht = 1.75  # m
hr = 1.75
wavelength = scipy.constants.speed_of_light / (868 * 10 ** 6)

d_break_theoretical = (4 * ht * hr) / wavelength
warnings.simplefilter(action='ignore', category=FutureWarning)


def all_white_listed(white_list, values):
    for x in white_list:
        if x not in values:
            return False
    return True


with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)
    measurements = config["measurements"]
    data = pd.read_pickle(input_file_measurements_path)

    path_loss_estimates = pd.read_pickle(input_file_estimated_pl_path)

    for measurement in measurements:
        print(F"--------------------- PATH LOSS MODEL {measurement} ---------------------")
        fig = plt.figure(figsize=(4, 3))
        plt.xscale('log')

        df = data[measurement]["data"]
        d_all = df["distance"].values
        uncensored_packets_mask = np.invert(data[measurement]["censored_packets_mask"])

        df_uncensored = df.loc[uncensored_packets_mask]
        d_uncensored = df_uncensored["distance"].values
        pld_uncensored = df_uncensored["pl_db"].values

        plt.scatter(d_uncensored, pld_uncensored, marker='x', label="Measured Path Loss", s=1, c='0.75')

        df = path_loss_estimates.loc[path_loss_estimates['Measurement'] == measurement]

        if "Dual Slope" in white_list and "Dual Slope" not in black_list:
            plt.axvline(d_break_theoretical, ls="--", c='0.75')

        for ix, row in df.iterrows():

            if all_white_listed(white_list, row.values) and not any(x in black_list for x in row.values):
                print(row['Params'])

                sigma_bound_up = row['PLm'] + row['TwoSigmaBound'] / 2
                sigma_bound_down = row['PLm'] - row['TwoSigmaBound'] / 2
                p = plt.plot(row['Distances'], row['PLm'],
                             label=F"{row['Weight']}{row['Std']}{row['Model']}{row['Censored']}{row['MLValue']}")
                plt.fill_between(x=row['Distances'], y1=sigma_bound_down, y2=sigma_bound_up, alpha=0.05)
                plt.plot(row['Distances'], sigma_bound_down, color=p[0].get_color(), ls="--", alpha=0.5)
                plt.plot(row['Distances'], sigma_bound_up, color=p[0].get_color(), ls="--", alpha=0.5)

        plt.axhline(148, ls="--", c='0.75')
        plt.xlabel(r'Log distance (m)')
        plt.ylabel(r'Path Loss (dB)')

        plt.yticks(list(plt.yticks()[0]) + [148])

        plt.legend()
        plt.show()

        print(F"Done for {measurement}")

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

import numpy as np
import pandas as pd
from get_weights import get_apparent_data_points
from LatexifyMatplotlib import LatexifyMatplotlib as lm
import matplotlib.pyplot as plt

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))

input_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))
input_file_name = "processed_data_with_censored_data.pkl"
input_file_path = os.path.join(input_path, input_file_name)

lm.latexify()


first = True

with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)
    measurements = config["measurements"]
    data = pd.read_pickle(input_file_path)

    for cnt, measurement in enumerate(measurements):
        fig = plt.figure(figsize=(4, 3))
        ax = plt.gca()
        print(F"--------------------- PATH LOSS MODEL {measurement} ---------------------")

        df = data[measurement]["data"]
        censored_packets_mask = data[measurement]["censored_packets_mask"]
        uncensored_packets_mask = np.invert(censored_packets_mask)

        df_uncensored = df.loc[uncensored_packets_mask]

        d_uncensored = df_uncensored["distance"].values
        pld_uncensored = df_uncensored["pl_db"].values

        d_all = df["distance"].values
        pld_all = df["pl_db"].values

        #num_bin = int((d_uncensored.max() - d_uncensored.min()) / 50)

        num_bin = 30
        print(F"Number of bins: {num_bin}")

        y, x = get_apparent_data_points(d_uncensored, weight_type=None, num_bins=num_bin)
        plt.plot(x, y-np.mean(y), label='No weighting')
        y, x = get_apparent_data_points(d_uncensored, weight_type='linear', num_bins=num_bin)

        plt.plot(x, y-np.mean(y), label='Linear')
        y, x = get_apparent_data_points(d_uncensored, weight_type='log10', num_bins=num_bin)
        plt.plot(x, y-np.mean(y), label='Logarithmic')
        y, x = get_apparent_data_points(d_uncensored, weight_type='square', num_bins=num_bin)
        plt.plot(x, y-np.mean(y), label='Square')

        ax.set_xlabel(r'Distance (m)')

        scale_legend = None
        if first:
            lm.legend(plt)
            ax.set_ylabel(r'Apparent Samples')
            first = False
            scale_legend = "0.75"

        lm.format_axes(ax)

        lm.save(F"apparent_data_points_{measurement}.tex", scale_legend=scale_legend, plt=plt)


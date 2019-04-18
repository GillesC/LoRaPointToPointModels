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
from LatexifyMatplotlib import LatexifyMatplotlib as lm
from scipy.stats import norm

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))

input_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))

input_file_measurements_path = os.path.join(input_path, "processed_data_with_censored_data.pkl")

fig, ax = plt.subplots(figsize=(4, 3))
lm.latexify()
plt.xscale('log')

max_pl = 0

markers = ["s", "^", "."]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)
    measurements = config["measurements"]
    data = pd.read_pickle(input_file_measurements_path)

    for measurement in measurements:
        print(F"--------------------- PATH LOSS MODEL {measurement} ---------------------")

        df = data[measurement]["data"]
        d_all = df["distance"].values
        uncensored_packets_mask = np.invert(data[measurement]["censored_packets_mask"])

        df_uncensored = df.loc[uncensored_packets_mask]
        d_uncensored = df_uncensored["distance"].values
        pl_uncensored = df_uncensored["pl_db"].values

        max_pl = max(max_pl, pl_uncensored.max())

        plt.scatter(d_uncensored, pl_uncensored, marker=markers.pop(), label=measurement, s=1.5, alpha=0.2, facecolors='none', edgecolors=colors.pop(0))

    plt.axhline(max_pl, ls="--")
    plt.xlabel(r'Distance (m)')
    plt.ylabel(r'')

    plt.legend()

lm.format_axes(ax)
lm.save(F"uncensored_rss_wrt_distance.tex", plt=plt)

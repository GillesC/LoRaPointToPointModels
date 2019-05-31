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
from scipy.stats import norm
from LatexifyMatplotlib import LatexifyMatplotlib as lm

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))

input_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))

input_file_measurements_path = os.path.join(input_path, "processed_data_with_censored_data.pkl")
input_file_estimated_pl_path = os.path.join(input_path, "estimated_path_loss.pkl")

white_list = np.array(["Censored", "Single Slope", "No"])
black_list = np.array([])

warnings.simplefilter(action='ignore', category=FutureWarning)


def all_white_listed(white_list, values):
    for x in white_list:
        if x not in values:
            return False
    return True



fig, ax = plt.subplots(figsize=(4, 3))
lm.latexify()
plt.xscale('log')

with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)
    measurements = config["measurements"]
    data = pd.read_pickle(input_file_measurements_path)

    path_loss_estimates = pd.read_pickle(input_file_estimated_pl_path)

    for measurement in measurements:
        print(F"--------------------- PATH LOSS MODEL {measurement} ---------------------")

        df = data[measurement]["data"]
        d_all = df["distance"].values
        censored_packets_mask = data[measurement]["censored_packets_mask"]
        uncensored_packets_mask = np.invert(censored_packets_mask)

        d_censored = d_all[censored_packets_mask]
        d_uncensored = d_all[uncensored_packets_mask]

        allocated_to_bins_uncensored, bins = pd.cut(d_uncensored, bins=100, right=False, retbins=True)
        allocated_to_bins_censored = pd.cut(d_uncensored, bins=bins, right=False)
        
        fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)

        # We can set the number of bins with the `bins` kwarg
        
        n_uncensored, bins,_ = axs[1].hist(d_uncensored, bins=100)
        n_censored, bins, patches = axs[0].hist(d_censored, bins=bins)

        fig, ax = plt.subplots(1, 1)

        plt.scatter(x=bins[:-1], y=(n_censored/n_uncensored))

        plt.show()


    plt.legend()

#lm.format_axes(ax)
#lm.save(F"prob_success.tex", plt=plt)

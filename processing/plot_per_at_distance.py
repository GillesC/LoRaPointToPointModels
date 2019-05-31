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
input_file_estimated_pl_path = os.path.join(input_path, "estimated_path_loss_2019513_39.pkl")

print(F"Getting data from {input_file_estimated_pl_path}")

white_list = np.array(["Censored", "Single Slope", "No"])
black_list = np.array([])

warnings.simplefilter(action='ignore', category=FutureWarning)


def all_white_listed(white_list, values):
    for x in white_list:
        if x not in values:
            return False
    return True

d0 = 1

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
        uncensored_packets_mask = np.invert(data[measurement]["censored_packets_mask"])

        df_uncensored = df.loc[uncensored_packets_mask]
        d_uncensored = df_uncensored["distance"].values
        pld_uncensored = df_uncensored["pl_db"].values

        df = path_loss_estimates.loc[path_loss_estimates['Measurement'] == measurement]

        for ix, row in df.iterrows():
            if all_white_listed(white_list, row.values) and not any(x in black_list for x in row.values):
                (Pld0, n, _sigma) = row['Params']

                d = np.array(range(int(d_all.min()), int(d_all.max())))

                plm = Pld0 + 10 * n * np.log10(d/d0)
                sigma = np.repeat(_sigma, len(d))

                llh_censored = 1 - norm.cdf((148 - plm) / sigma)


                def find_nearest(array, value):
                    array = np.asarray(array)
                    return (np.abs(array - value)).argmin()

                idx = find_nearest(llh_censored, 0.2)
                print(d[idx])

                p = plt.plot(d, llh_censored, label=F"{measurement}")

                d = np.array(range(int(d_all.max()), 4200))
                plm = Pld0 + 10 * n * np.log10(d/d0)
                sigma = np.repeat(_sigma, len(d))

                llh_censored = 1 - norm.cdf((148 - plm) / sigma)
                plt.plot(d, llh_censored, color=p[0].get_color(), ls="--", alpha=0.5)

    plt.xlabel(r'Distance (m)')
    plt.ylabel(r'Probability of Receiving a Packet below \(PL_{\textrm{th}}\)')

    plt.legend()

lm.format_axes(ax)
lm.save(F"per.tex", plt=plt)

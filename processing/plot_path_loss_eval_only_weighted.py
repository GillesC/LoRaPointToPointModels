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
import regression_models as model
from get_weights import get_weights, histcounts
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

with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)
    measurements = config["measurements"]
    data = pd.read_pickle(input_file_path)

    for measurement in measurements:
        print(F"--------------------- PATH LOSS MODEL {measurement} ---------------------")

        df = data[measurement]["data"]
        censored_packets_mask = data[measurement]["censored_packets_mask"]
        uncensored_packets_mask = np.invert(censored_packets_mask)

        d_all = df["distance"].values
        pld_all = df["pl_db"].values

        df_uncensored = df.loc[uncensored_packets_mask]

        d_uncensored = df_uncensored["distance"].values
        pld_uncensored = df_uncensored["pl_db"].values

        num_bins = int((d_uncensored.max() - d_uncensored.min()) / 20)

        print(F"Using {num_bins} bins with a width of {(d_uncensored.max() - d_uncensored.min()) / num_bins} m")

        w_lin = get_weights(d_uncensored, weight_type='linear', num_bins=num_bins)
        w_log = get_weights(d_uncensored, weight_type='log10', num_bins=num_bins)
        w_sq = get_weights(d_uncensored, weight_type='square', num_bins=num_bins)

        (pld0_ols, n_ols, sigma_ols) = model.ols(d0=1, d=d_uncensored, pld=pld_uncensored)

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(4, 3))

        for (i, (sigma, sigma_type)) in enumerate(zip([sigma_ols, [0, sigma_ols]], ["constant", "distant_dependent"])):

            for weight_type, weight_name in zip([None, w_lin, w_log, w_sq], ["No", "Linear", "Log10", "Square"]):

                print(F"Plotting mean(PLm-PLd) (20m) with {weight_name} and {sigma_type}")

                (pld0, n, *_) = model.ml(d0=1, d=d_uncensored, pld=pld_uncensored, c=148, pld0=pld0_ols, n=n_ols,
                                         sigma=sigma, weights=weight_type,
                                         censored=False)

                plm = pld0 + 10 * n * np.log10(d_uncensored)
                diff = plm - pld_uncensored

                hist_val, inds, bin_edges = histcounts(d_uncensored, num_bins=num_bins)

                val = np.zeros(len(hist_val))
                for (difference, ix) in zip(diff, inds):
                    if type(val[ix]) is not np.ndarray:
                        val[ix] = np.array([difference])
                    else:
                        val[ix].append(difference)

                y = [np.array(l).mean() for l in val]

                axes[i].set_title(F"ML w/ censored data for {measurement} and {sigma_type}")
                axes[i].plot(bin_edges, y, label=weight_name)
                axes[i].fill_between(bin_edges, y, alpha=0.15)

            plt.legend()
            lm.format_axes(axes[i])

            lm.save(F"mean_path_loss_{measurement}_{sigma_type}.tex", plt=plt, show=True)

        del df_uncensored

        fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(4, 3))
        w_lin = get_weights(d_all, weight_type='linear', num_bins=num_bins)
        w_log = get_weights(d_all, weight_type='log10', num_bins=num_bins)
        w_sq = get_weights(d_all, weight_type='square', num_bins=num_bins)

        for (i, sigma, sigma_type) in enumerate(zip([sigma_ols, [0, sigma_ols]], ["constant", "distant_dependent"])):

            for weight_type, weight_name in zip([None, w_lin, w_log, w_sq], ["No", "Linear", "Log10", "Square"]):

                print(F"Plotting mean(PLm-PLd) (20m) with {weight_name} and {sigma_type}")

                (pld0, n, *_) = model.ml(d0=1, d=d_all, pld=pld_all, c=148, pld0=pld0_ols, n=n_ols,
                                         sigma=sigma, weights=weight_type, censored_mask=censored_packets_mask,
                                         censored=True)

                plm = pld0 + 10 * n * np.log10(d_uncensored)
                diff = plm - pld_uncensored

                hist_val, inds, bin_edges = histcounts(d_uncensored, num_bins=num_bins)

                val = np.zeros(len(hist_val))
                for (difference, ix) in zip(diff, inds):
                    if type(val[ix]) is not np.ndarray:
                        val[ix] = np.array([difference])
                    else:
                        val[ix].append(difference)

                y = [np.array(l).mean() for l in val]

                axes[i].plot(bin_edges, y, label=weight_name)
                axes[i].fill_between(bin_edges, y, alpha=0.15)

            axes[i].set_title(F"ML with censored data for {measurement} and {sigma_type}")
            plt.legend()
            lm.format_axes(axes[i])

            lm.save(F"mean_path_loss_{measurement}_{sigma_type}.tex", plt=plt, show=True)

            fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, sharex=True, figsize=(4, 3))

            for (i, sigma, sigma_type) in enumerate(
                    zip([sigma_ols, [0, 0, sigma_ols]], ["constant", "distant_dependent"])):

                for weight_type, weight_name in zip([None, w_lin, w_log, w_sq], ["No", "Linear", "Log10", "Square"]):

                    print(F"Plotting mean(PLm-PLd) (20m) with {weight_name} and {sigma_type}")

                    (pld0, n, *_) = model.ml_dual_slope(d0=1, d=d_all, pld=pld_all, c=148, pld0=pld0_ols, n=n_ols,
                                                        sigma=sigma, weights=weight_type,
                                                        censored_mask=censored_packets_mask,
                                                        censored=True)

                    plm = pld0 + 10 * n * np.log10(d_uncensored)
                    diff = plm - pld_uncensored

                    hist_val, inds, bin_edges = histcounts(d_uncensored, num_bins=num_bins)

                    val = np.zeros(len(hist_val))
                    for (difference, ix) in zip(diff, inds):
                        if type(val[ix]) is not np.ndarray:
                            val[ix] = np.array([difference])
                        else:
                            val[ix].append(difference)

                    y = [np.array(l).mean() for l in val]

                    axes[i].plot(bin_edges, y, label=weight_name)
                    axes[i].fill_between(bin_edges, y, alpha=0.15)

                axes[i].set_title(F"ML with censored data for {measurement} and {sigma_type} dualslope")
                plt.legend()
                lm.format_axes(axes[i])

                lm.save(F"mean_path_loss_{measurement}_{sigma_type}_dual_slope.tex", plt=plt, show=True)

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
input_file_name = "processed_data_with_censored_data.pkl"
input_file_path = os.path.join(input_path, input_file_name)


output_file = os.path.abspath(os.path.join(input_path, "estimated_path_loss.pkl"))

result_df = pd.DataFrame(columns=['Measurement', 'Weight', 'Std', 'Model', 'Params', 'PLm', 'NumBins', "Distances"])

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

        plm = pld0_ols + 10 * n_ols * np.log10(d_uncensored)

        result_df.append({
            'Measurement': measurement,
            'Weight': 'None',
            'Std': 'constant',
            'Model': 'OLS',
            'Params': (pld0_ols, n_ols, sigma_ols),
            'PLm': plm,
            'NumBins': num_bins,
            'Distances': d_uncensored,
        },ignore_index=True)

        for (i, (sigma, sigma_name)) in enumerate(zip([sigma_ols, [0, sigma_ols]], ["constant", "distant_dependent"])):

            for weight_type, weight_name in zip([None, w_lin, w_log, w_sq], ["No", "Linear", "Log10", "Square"]):
                (pld0, n, *_) = model.ml(d0=1, d=d_uncensored, pld=pld_uncensored, c=148, pld0=pld0_ols, n=n_ols,
                                         sigma=sigma, weights=weight_type,
                                         censored=False)

                plm = pld0 + 10 * n * np.log10(d_uncensored)

                result_df.append({
                    'Measurement': measurement,
                    'Weight': weight_name,
                    'Std': sigma_name,
                    'Model': 'Single Slope',
                    'Params': (pld0, n, *_),
                    'PLm': plm,
                    'NumBins': num_bins,
                    'Distances': d_uncensored,
                },ignore_index=True)

            w_lin_all = get_weights(d_all, weight_type='linear', num_bins=num_bins)
            w_log_all = get_weights(d_all, weight_type='log10', num_bins=num_bins)
            w_sq_all = get_weights(d_all, weight_type='square', num_bins=num_bins)

            for weight_type, weight_name in zip([None, w_lin_all, w_log_all, w_sq_all],
                                                ["No", "Linear", "Log10", "Square"]):
                (pld0, n, *_) = model.ml(d0=1, d=d_all, pld=pld_all, c=148, pld0=pld0_ols, n=n_ols,
                                         sigma=sigma, weights=weight_type, censored_mask=censored_packets_mask,
                                         censored=True)
                plm = pld0 + 10 * n * np.log10(d_all)

                result_df.append({
                    'Measurement': measurement,
                    'Weight': weight_name,
                    'Std': sigma_name,
                    'Model': 'Single Slope',
                    'Params': (pld0, n, *_),
                    'PLm': plm,
                    'NumBins': num_bins,
                    'Distances': d_all,
                },ignore_index=True)

        ht = 1.75  # m
        hr = 1.75
        wavelength = scipy.constants.speed_of_light / (868 * 10 ** 6)

        d_break = (4 * ht * hr) / wavelength

        print(F"Plotting the dual-slope model")
        for (i, (sigma, sigma_name)) in enumerate(
                zip([sigma_ols, [0, 0, sigma_ols]], ["constant", "distant_dependent"])):

            for weight_type, weight_name in zip([None, w_lin_all, w_log_all, w_sq_all],
                                                ["No", "Linear", "Log10", "Square"]):

                d0 = 1
                if sigma_name == "constant":
                    x0 = [pld0_ols, n_ols, n_ols, d_break, sigma]
                else:
                    x0 = [pld0_ols, n_ols, n_ols, d_break, sigma[0], sigma[1], sigma[2]]

                (pld0, n1, n2, d_break, *_) = model.ml_dual_slope(d0=d0, d=d_all, pld=pld_all, c=148, x0=x0,
                                                                  weights=weight_type,
                                                                  censored_mask=censored_packets_mask,
                                                                  censored=True)
                mask_below_d_break = d_all < d_break
                mask_above_d_break = np.invert(mask_below_d_break)

                plm[mask_below_d_break] = 10 * n1 * np.log10(d_all[mask_below_d_break] / d0) + pld0
                plm[mask_above_d_break] = 10 * n1 * np.log10(d_break / d0) + 10 * n2 * np.log10(
                    d_all[mask_above_d_break] / d_break) + pld0

                result_df.append({
                    'Measurement': measurement,
                    'Weight': weight_name,
                    'Std': sigma_name,
                    'Model': 'Dual Slope',
                    'Params': (pld0, n1, n2, d_break, *_),
                    'PLm': plm,
                    'NumBins': num_bins,
                    'Distances': d_all,
                },ignore_index=True)

        print(F"Done for {measurement}")

result_df.to_pickle(output_file)

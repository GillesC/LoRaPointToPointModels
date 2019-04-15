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
import scipy.constants

import regression_models as model
from get_weights import get_weights

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))

input_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))
input_file_name = "processed_data_with_censored_data.pkl"
input_file_path = os.path.join(input_path, input_file_name)

output_file = os.path.abspath(os.path.join(input_path, "estimated_path_loss_a_pos.pkl"))

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

        d_plot = np.array(range(int(np.amin(d_all)), int(np.amax(d_all))))
        d_plot_uncensored = np.array(range(int(np.amin(d_uncensored)), int(np.amax(d_uncensored))))

        num_bins = int((d_uncensored.max() - d_uncensored.min()) / 20)

        print(F"Using {num_bins} bins with a width of {(d_uncensored.max() - d_uncensored.min()) / num_bins} m")

        w_lin = get_weights(d_uncensored, weight_type='linear', num_bins=num_bins)
        w_log = get_weights(d_uncensored, weight_type='log10', num_bins=num_bins)
        w_sq = get_weights(d_uncensored, weight_type='square', num_bins=num_bins)

        (pld0_ols, n_ols, sigma_ols) = model.ols(d0=1, d=d_uncensored, pld=pld_uncensored)

        sigma = [0, sigma_ols]
        sigma_name = "distant_dependent_fixed"
        w_lin_all = get_weights(d_all, weight_type='linear', num_bins=num_bins)
        w_log_all = get_weights(d_all, weight_type='log10', num_bins=num_bins)
        w_sq_all = get_weights(d_all, weight_type='square', num_bins=num_bins)

        for weight_type, weight_name in zip([None, w_lin_all, w_log_all, w_sq_all],
                                            ["No", "Linear", "Log10", "Square"]):
            res = model.ml_with_constraints(d0=1, d=d_all, pld=pld_all, c=148, pld0=pld0_ols, n=n_ols,
                                            sigma=sigma, weights=weight_type, censored_mask=censored_packets_mask,
                                            censored=True)

            print(res)
            (pld0, n, *_) = res.x
            plm = pld0 + 10 * n * np.log10(d_plot)
            plm_est = pld0 + 10 * n * np.log10(d_all)

            if len(_) == 1:
                sig = _[0]
                two_sigma_bound = np.repeat(2 * sig, len(d_plot))
                sigma_est = np.repeat(2 * sig, len(d_all))
            else:
                a = _[0]
                b = _[1]
                two_sigma_bound = 2 * (a * np.log10(d_plot) + b)
                sigma_est = (a * np.log10(d_all) + b)

            ml_value = model.ml_value(pld=pld_all, plm_est=plm_est, sigma_est=sigma_est,
                                      censored_mask=censored_packets_mask)

            result_df = result_df.append({
                'Measurement': measurement,
                'Weight': weight_name,
                'Std': sigma_name,
                'Model': 'Single Slope',
                'Params': (pld0, n, *_),
                'PLm': plm,
                'NumBins': num_bins,
                'Distances': d_plot,
                'TwoSigmaBound': two_sigma_bound,
                "Censored": "Censored",
                "MLValue": ml_value
            }, ignore_index=True)

            print(result_df.iloc[-1, :])

result_df.to_pickle(output_file)

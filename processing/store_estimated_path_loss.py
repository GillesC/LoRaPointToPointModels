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

        d_plot = np.array(range(int(np.amin(d_all)), int(np.amax(d_all))))
        d_plot_uncensored = np.array(range(int(np.amin(d_uncensored)), int(np.amax(d_uncensored))))

        num_bins = int((d_uncensored.max() - d_uncensored.min()) / 20)

        print(F"Using {num_bins} bins with a width of {(d_uncensored.max() - d_uncensored.min()) / num_bins} m")

        w_lin = get_weights(d_uncensored, weight_type='linear', num_bins=num_bins)
        w_log = get_weights(d_uncensored, weight_type='log10', num_bins=num_bins)
        w_sq = get_weights(d_uncensored, weight_type='square', num_bins=num_bins)

        (pld0_ols, n_ols, sigma_ols) = model.ols(d0=1, d=d_uncensored, pld=pld_uncensored)

        plm = pld0_ols + 10 * n_ols * np.log10(d_plot_uncensored)
        two_sigma_bound = np.zeros(len(d_plot_uncensored)) + 2 * sigma_ols

        result_df = result_df.append({
            'Measurement': measurement,
            'Weight': 'None',
            'Std': 'constant',
            'Model': 'OLS',
            'Params': (pld0_ols, n_ols, sigma_ols),
            'PLm': plm,
            'NumBins': num_bins,
            'Distances': d_plot_uncensored,
            'TwoSigmaBound': two_sigma_bound,
            "Censored": "Uncensored",
            "MLValue": None,
        }, ignore_index=True)
        print(result_df.iloc[-1, :])

        for (i, (sigma, sigma_name)) in enumerate(zip([sigma_ols, [0, sigma_ols]], ["constant", "distant_dependent"])):

            for weight_type, weight_name in zip([None, w_lin, w_log, w_sq], ["No", "Linear", "Log10", "Square"]):
                res = model.ml_with_constraints(d0=1, d=d_uncensored, pld=pld_uncensored, c=148, pld0=pld0_ols, n=n_ols,
                                          sigma=sigma, weights=weight_type,
                                          censored=False)

                print(res)
                (pld0, n, *_) = res.x


                plm = pld0 + 10 * n * np.log10(d_plot_uncensored)
                if len(_) == 1:
                    sig = _[0]
                    two_sigma_bound = np.repeat(2 * sig, len(d_plot_uncensored))
                    sigma_est = np.repeat(sig, len(d_uncensored))
                else:
                    a = _[0]
                    b = _[1]
                    two_sigma_bound = 2 * (a * np.log10(d_plot_uncensored) + b)
                    sigma_est = a * np.log10(d_uncensored) + b

                plm_est = pld0 + 10 * n * np.log10(d_uncensored)

                ml_value = model.ml_value(pld=pld_uncensored, sigma_est=sigma_est, plm_est=plm_est)

                result_df = result_df.append({
                    'Measurement': measurement,
                    'Weight': weight_name,
                    'Std': sigma_name,
                    'Model': 'Single Slope',
                    'Params': (pld0, n, *_),
                    'PLm': plm,
                    'NumBins': num_bins,
                    'Distances': d_plot_uncensored,
                    'TwoSigmaBound': two_sigma_bound,
                    "Censored": "Uncensored",
                    "MLValue": ml_value
                }, ignore_index=True)

                print(result_df.iloc[-1, :])

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

        ht = 1.75  # m
        hr = 1.75
        wavelength = scipy.constants.speed_of_light / (868 * 10 ** 6)

        d_break = (4 * ht * hr) / wavelength

        print(F"Done for {measurement}")

"""
        print(F"Processing the dual-slope model")
        for (i, (sigma, sigma_name)) in enumerate(
                zip([sigma_ols, [0, 0, sigma_ols]], ["constant", "distant_dependent"])):

            for weight_type, weight_name in zip([None, w_lin_all, w_log_all, w_sq_all],
                                                ["No", "Linear", "Log10", "Square"]):

                d0 = 1
                if sigma_name == "constant":
                    x0 = [pld0_ols, n_ols, n_ols, d_break, sigma]
                else:
                    x0 = [pld0_ols, n_ols, n_ols, d_break, sigma[0], sigma[1], sigma[2]]

                xopt, fopt, *_ = model.ml_dual_slope(d0=d0, d=d_all, pld=pld_all, c=148, x0=x0,
                                                     weights=weight_type,
                                                     censored_mask=censored_packets_mask,
                                                     censored=True)
                (pld0, n1, n2, d_break, *_) = xopt
                mask_below_d_break = d_plot < d_break
                mask_above_d_break = np.invert(mask_below_d_break)

                plm = np.zeros(len(d_plot))
                plm[mask_below_d_break] = 10 * n1 * np.log10(d_plot[mask_below_d_break] / d0) + pld0
                plm[mask_above_d_break] = 10 * n1 * np.log10(d_break / d0) + 10 * n2 * np.log10(
                    d_plot[mask_above_d_break] / d_break) + pld0

                mask_below_d_break_est = d_all < d_break
                mask_above_d_break_est = np.invert(mask_below_d_break_est)
                plm_est = np.zeros(len(d_all))
                plm_est[mask_below_d_break_est] = 10 * n1 * np.log10(d_all[mask_below_d_break_est] / d0) + pld0
                plm_est[mask_above_d_break_est] = 10 * n1 * np.log10(d_break / d0) + 10 * n2 * np.log10(
                    d_all[mask_above_d_break_est] / d_break) + pld0

                if len(_) == 1:
                    sig = _[0]
                    two_sigma_bound = np.zeros(len(d_plot)) + 2 * sig
                    sigma_est = np.repeat(sig, len(d_all))
                else:
                    a1 = _[0]
                    a2 = _[1]
                    b = _[2]

                    two_sigma_bound = np.zeros(len(d_plot))
                    two_sigma_bound[mask_below_d_break] = a1 * np.log10(d_plot[mask_below_d_break]) + b
                    two_sigma_bound[mask_above_d_break] = a1 * np.log10(d_plot[mask_above_d_break]) + a2 * np.log10(
                        d_plot[mask_above_d_break] / d_break) + b

                    sigma_est = np.zeros(len(d_all))
                    sigma_est[mask_below_d_break_est] = a1 * np.log10(d_all[mask_below_d_break_est]) + b
                    sigma_est[mask_above_d_break_est] = a1 * np.log10(d_all[mask_above_d_break_est]) + a2 * np.log10(
                        d_all[mask_above_d_break_est] / d_break) + b

                ml_value = model.ml_value(pld=pld_all, plm_est=plm_est, sigma_est=sigma_est,
                                          censored_mask=censored_packets_mask)

                result_df = result_df.append({
                    'Measurement': measurement,
                    'Weight': weight_name,
                    'Std': sigma_name,
                    'Model': 'Dual Slope',
                    'Params': (pld0, n1, n2, d_break, *_),
                    'PLm': plm,
                    'NumBins': num_bins,
                    'Distances': d_plot,
                    'TwoSigmaBound': two_sigma_bound,
                    "Censored": "Censored",
                    "MLValue": ml_value
                }, ignore_index=True)

                print(result_df.iloc[-1, :])
"""


result_df.to_pickle(output_file)

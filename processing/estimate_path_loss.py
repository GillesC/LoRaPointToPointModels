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
from get_weights import get_weights
import warnings
import scipy.constants

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))

input_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))
input_file_name = "processed_data_with_censored_data.pkl"
input_file_path = os.path.join(input_path, input_file_name)

warnings.simplefilter("ignore")


with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)
    measurements = config["measurements"]
    data = pd.read_pickle(input_file_path)

    for measurement in measurements:
        print(F"--------------------- PATH LOSS MODEL {measurement} ---------------------")

        df = data[measurement]["data"]
        censored_packets_mask = data[measurement]["censored_packets_mask"]
        uncensored_packets_mask = np.invert(censored_packets_mask)

        df_uncensored = df.loc[uncensored_packets_mask]

        d_uncensored = df_uncensored["distance"].values
        pld_uncensored = df_uncensored["pl_db"].values

        d_all = df["distance"].values
        pld_all = df["pl_db"].values

        num_bins = int((d_all.max() - d_all.min()) / 20)

        print(F"Using {num_bins} bins with a width of {(d_all.max() - d_all.min()) / num_bins} m")

        w_lin = get_weights(d_all, weight_type='linear', num_bins=num_bins)
        w_log = get_weights(d_all, weight_type='log10', num_bins=num_bins)
        w_sq = get_weights(d_all, weight_type='square', num_bins=num_bins)

        (pld0_ols, n_ols, sigma_ols) = model.ols(d0=1, d=d_uncensored, pld=pld_uncensored)



        """
        LaTex table with results
        """

        w_lin_all = get_weights(d_all, weight_type='linear', num_bins=num_bins)
        w_log_all = get_weights(d_all, weight_type='log10', num_bins=num_bins)
        w_sq_all = get_weights(d_all, weight_type='square', num_bins=num_bins)

        w_lin_uncensored = get_weights(d_uncensored, weight_type='linear', num_bins=num_bins)
        w_log_uncensored = get_weights(d_uncensored, weight_type='log10', num_bins=num_bins)
        w_sq_uncensored = get_weights(d_uncensored, weight_type='square', num_bins=num_bins)

        print(r"""
        \begin{table}[]
	        \centering
	        \caption{}%
	        \label{tab:}
	        \begin{tabularx}{0.8\linewidth}{lllll}
		        \toprule
		        \textbf{n} & \textbf{PL(\(d_0\))} & \textbf{\(\sigma\)} & \textbf{Weighting} & \textbf{Censored Data} \\ \midrule
        """)

        # fixed sigma no weights no censored data
        (pld0, n, sigma) = model.ml(d0=1, d=d_uncensored, pld=pld_uncensored, c=148,
                                    pld0=pld0_ols, n=n_ols, sigma=sigma_ols, censored=False)
        print(F"                {n:.2f} & {pld0:.2f} & {sigma:.2f} & None & No \\\\")

        # fixed sigma lin weights no censored data
        (pld0, n, sigma) = model.ml(d0=1, d=d_uncensored, pld=pld_uncensored, c=148, pld0=pld0_ols, n=n_ols,
                                    sigma=sigma_ols, weights=w_lin_uncensored,
                                    censored=False)
        print(F"                {n:.2f} & {pld0:.2f} & {sigma:.2f} & Linear & No \\\\")

        # fixed sigma log weights no censored data
        (pld0, n, sigma) = model.ml(d0=1, d=d_uncensored, pld=pld_uncensored, c=148, pld0=pld0_ols, n=n_ols,
                                    sigma=sigma_ols, weights=w_log_uncensored,
                                    censored=False)
        print(F"                {n:.2f} & {pld0:.2f} & {sigma:.2f} & Logarithmic & No \\\\")

        # fixed sigma square weights no censored data
        (pld0, n, sigma) = model.ml(d0=1, d=d_uncensored, pld=pld_uncensored, c=148, pld0=pld0_ols, n=n_ols,
                                    sigma=sigma_ols, weights=w_sq_uncensored,
                                    censored=False)
        print(F"                {n:.2f} & {pld0:.2f} & {sigma:.2f} & Square & No \\\\")

        # =============================================================================

        # fixed sigma square weights no censored data
        (pld0, n, sigma) = model.ml(d0=1, d=d_all, pld=pld_all, censored_mask=censored_packets_mask, c=148,
                                    pld0=pld0_ols, n=n_ols, sigma=sigma_ols, censored=True)
        print(F"                {n:.2f} & {pld0:.2f} & {sigma:.2f} & None & Yes \\\\")

        # fixed sigma lin weights no censored data
        (pld0, n, sigma) = model.ml(d0=1, d=d_all, pld=pld_all, censored_mask=censored_packets_mask, c=148,
                                    pld0=pld0_ols, n=n_ols,
                                    sigma=sigma_ols, weights=w_lin_all,
                                    censored=True)
        print(F"                {n:.2f} & {pld0:.2f} & {sigma:.2f} & Linear & Yes \\\\")

        # fixed sigma log weights no censored data
        (pld0, n, sigma) = model.ml(d0=1, d=d_all, pld=pld_all, censored_mask=censored_packets_mask, c=148,
                                    pld0=pld0_ols, n=n_ols,
                                    sigma=sigma_ols, weights=w_log_all,
                                    censored=True)
        print(F"                {n:.2f} & {pld0:.2f} & {sigma:.2f} & Logarithmic & Yes \\\\")

        # fixed sigma square weights no censored data
        (pld0, n, sigma) = model.ml(d0=1, d=d_all, pld=pld_all, censored_mask=censored_packets_mask, c=148,
                                    pld0=pld0_ols, n=n_ols,
                                    sigma=sigma_ols, weights=w_sq_all,
                                    censored=True)
        print(F"                {n:.2f} & {pld0:.2f} & {sigma:.2f} & Square & Yes \\\\")

        # =============================================================================

        # var sigma square weights  censored data
        (pld0, n, a, b) = model.ml(d0=1, d=d_all, pld=pld_all, censored_mask=censored_packets_mask, c=148,
                                   pld0=pld0_ols, n=n_ols, sigma=[0, sigma_ols], censored=True)
        print(F"                {n:.2f} & {pld0:.2f} & {a:.2f} log10(d/d0) + {b:.2f} & None & Yes \\\\")

        (pld0, n, a, b) = model.ml(d0=1, d=d_all, pld=pld_all, censored_mask=censored_packets_mask, c=148,
                                   pld0=pld0_ols, n=n_ols,
                                   sigma=[0, sigma_ols], weights=w_lin_all,
                                   censored=True)
        print(F"                {n:.2f} & {pld0:.2f} & {a:.2f} log10(d/d0) + {b:.2f} & Linear & Yes \\\\")

        (pld0, n, a, b) = model.ml(d0=1, d=d_all, pld=pld_all, censored_mask=censored_packets_mask, c=148,
                                   pld0=pld0_ols, n=n_ols,
                                   sigma=[0, sigma_ols], weights=w_log_all,
                                   censored=True)
        print(F"                {n:.2f} & {pld0:.2f} & {a:.2f} log10(d/d0) + {b:.2f} & Logarithmic & Yes \\\\")

        (pld0, n, a, b) = model.ml(d0=1, d=d_all, pld=pld_all, censored_mask=censored_packets_mask, c=148,
                                   pld0=pld0_ols, n=n_ols,
                                   sigma=[0, sigma_ols], weights=w_sq_all,
                                   censored=True)
        print(F"                {n:.2f} & {pld0:.2f} & {a:.2f} log10(d/d0) + {b:.2f} & Square & Yes \\\\")

        print(
            r"""                \bottomrule
	                    \end{tabularx}
            \end{table}
            """
        )

        # =====================================================================================================
        ht = 1.75  # m
        hr = 1.75
        wavelength = scipy.constants.speed_of_light / (868 * 10 ** 6)

        d_break = (4 * ht * hr) / wavelength

        print(F"Theoretical dbreak {d_break}")

        x0 = [pld0_ols, n_ols, n_ols, d_break, sigma_ols]
        (pld0, n1, n2, d_break_est, sigma_est) = model.ml_dual_slope(d0=1, d=d_all, pld=pld_all,
                                                                     censored_mask=censored_packets_mask, x0=x0,
                                                                     censored=True)
        print((pld0, n1, n2, d_break_est, sigma_est))

        x0 = [pld0_ols, n_ols, n_ols, d_break, sigma_ols]
        (pld0, n1, n2, sigma_est) = model.ml_dual_slope(d0=1, d=d_all, pld=pld_all,
                                                        censored_mask=censored_packets_mask, x0=x0,
                                                        censored=True, fixed_d_break=True)
        print((pld0, n1, n2, sigma_est))

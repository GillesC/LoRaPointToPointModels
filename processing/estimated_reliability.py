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

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))

input_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))

input_file_measurements_path = os.path.join(input_path, "processed_data_with_censored_data.pkl")
input_file_estimated_pl_path = os.path.join(input_path, "estimated_path_loss.pkl")

with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)
    measurements = config["measurements"]
    data = pd.read_pickle(input_file_measurements_path)

    path_loss_estimates = pd.read_pickle(input_file_estimated_pl_path)

    print(r"""
            \begin{table}[]
    	        \centering
    	        \caption{}%
    	        \label{tab:}
    	        \begin{tabularx}{0.8\linewidth}{llll}
    		        \toprule
    		        \textbf{Environment} & \textbf{Weight} & \textbf{\(\sigma\)-bound} & \textbf{\(2\sigma\)-bound} \\ \midrule
            """)

    for measurement in measurements:
        # print(F"--------------------- MEASUREMENT {measurement} ---------------------")
        df = path_loss_estimates.loc[path_loss_estimates['Measurement'] == measurement]

        first = True
        print(r"                        \multirow{4}{*}{",measurement, r"}")
        for ix, row in df.iterrows():
            if row['Model'] == "Single Slope" and row["Std"] == "constant" and row["Censored"] == "Censored":
                pld0, n, sigma = row['Params']

                d_sigma = 10 ** ((148 - (pld0 + sigma)) / (10 * n))

                d_sigma2 = 10 ** ((148 - (pld0 + 2 * sigma)) / (10 * n))

                print("                            & ", row['Weight'], " & ", F"{d_sigma:.2f}", " & ", F"{d_sigma2:.2f}", r"\\")

    print(
        r"""                \bottomrule
                    \end{tabularx}
        \end{table}
        """
    )

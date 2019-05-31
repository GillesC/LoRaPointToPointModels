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
input_file_estimated_pl_path = os.path.join(input_path, "estimated_path_loss_20190513_37.pkl")

print(F"Getting data from {input_file_estimated_pl_path}")

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
    	        \begin{tabularx}{0.8\linewidth}{lllll}
    		        \toprule
    		        \multirow{2}[3]{*}{Environment} & \multicolumn{2}{c}{$\hat{PL}(d_0)$} & \multicolumn{2}{c}{$\hat{n}$} & \multicolumn{2}{c}{$\hat{\sigma}$}                           \\
		\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}
		                                & ML                                  & OLS                           & ML                                 & OLS   & ML     & OLS    \\
		\midrule""")

    for measurement in measurements:
        params = dict()
        # print(F"--------------------- MEASUREMENT {measurement} ---------------------")
        df = path_loss_estimates.loc[path_loss_estimates['Measurement'] == measurement]

        first = True
        for ix, row in df.iterrows():
            if row['Model'] != "Dual Slope" and row["Std"] == "constant" and row["Weight"] in ["No","None"]:
                params[row['Model']] = row['Params']
        print(F"{measurement}","&",F"{params['Single Slope'][0]:.2f}",
              "&", F"{params['OLS'][0]:.2f}",
              "&", F"{params['Single Slope'][1]:.2f}",
              "&", F"{params['OLS'][1]:.2f}",
              "&", F"{params['Single Slope'][2]:.2f}",
              "&", F"{params['OLS'][2]:.2f}",
        r"\\")

    print(
        r"""                \bottomrule
                    \end{tabularx}
        \end{table}
        """
    )
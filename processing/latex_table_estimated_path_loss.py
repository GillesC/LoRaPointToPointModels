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
input_file_estimated_pl_path = os.path.join(input_path, "estimated_path_loss_2019514_11.pkl")

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
    		        \textbf{Environment} & \textbf{weight} & \textbf{\(\hat{PL(d_0)}\)} & \textbf{\(\hat{n}\)} & \textbf{\(\hat{\sigma}\)} \\ \midrule
            """)

    for measurement in measurements:
        # print(F"--------------------- MEASUREMENT {measurement} ---------------------")
        df = path_loss_estimates.loc[path_loss_estimates['Measurement'] == measurement]

        first = True
        print(r"                        \multirow{4}{*}{",measurement, r"}")
        for ix, row in df.iterrows():
            if row['Model'] == "Single Slope" and row["Std"] == "constant" and row["Censored"] == "Censored":
                pld0, n, sigma = row['Params']
                print("                            & ",row["Weight"],"& ", F"{pld0:.2f}", " & ", F"{n:.2f}", " & ", F"{sigma:.2f}", r"\\")

    print(
        r"""                \bottomrule
                    \end{tabularx}
        \end{table}
        """
    )

    print(r"""
                \begin{table}[]
        	        \centering
        	        \caption{Dual Slope}%
        	        \label{tab:}
        	        \begin{tabularx}{0.8\linewidth}{lllllll}
        		        \toprule
        		        \textbf{Environment} & \textbf{weight} & \textbf{\(\hat{PL(d_0)}\)} & \textbf{\(\hat{n_1}\)} & \textbf{\(\hat{n_2}\)} & \textbf{\(\hat{\sigma}\)}  & \textbf{\(\hat{\d_c}\)}\\ \midrule
                """)

    for measurement in measurements:
        # print(F"--------------------- MEASUREMENT {measurement} ---------------------")
        df = path_loss_estimates.loc[path_loss_estimates['Measurement'] == measurement]

        first = True
        print(r"                        \multirow{4}{*}{", measurement, r"}")
        for ix, row in df.iterrows():
            if row['Model'] == "Dual Slope" and row["Std"] == "constant" and row["Censored"] == "Censored":
                pld0, n1, n2, d_c, sigma = row['Params']
                print("                            & ", row["Weight"], "& ", F"{pld0:.2f}", " & ", F"{n1:.2f}", " & ", F"{n2:.2f}", " & "
                      F"{sigma:.2f}", " & "
                      F"{d_c:.2f}" , r"\\")

    print(
        r"""                \bottomrule
                    \end{tabularx}
        \end{table}
        """
    )

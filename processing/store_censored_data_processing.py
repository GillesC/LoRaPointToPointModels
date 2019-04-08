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
import pickle

import matlab.engine
import numpy as np
import pandas as pd
import util as util

eng = matlab.engine.start_matlab()

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))
input_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))
input_file_name = "preprocessed_data_with_censored_data.pkl"

output_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))
output_file = "processed_data_with_censored_data.pkl"
output_file_path = os.path.join(output_path, output_file)

PL_THRESHOLD = 148

result = dict()

with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)
    measurements = config["measurements"]
    for measurement in measurements:
        print(F"--------------------- PATH LOSS MODEL {measurement} ---------------------")
        input_file_path = os.path.join(input_path, measurement, input_file_name)

        df = pd.read_pickle(input_file_path)
        df = df[df.isPacket > 0]  # censored and uncensored packets
        print(F"Max. Path Loss = {df['pl_db'].max()}")

        censored_packets_mask = np.logical_or(df["isPacket"] == 2, df['pl_db'] > PL_THRESHOLD).values
        print(F" Found {len(censored_packets_mask)} censored packets.")
        uncensored_packets_mask = np.invert(censored_packets_mask)

        df.loc[censored_packets_mask, "isPacket"] = 0
        df.loc[censored_packets_mask, "pl_db"] = PL_THRESHOLD

        num_censored_packets = censored_packets_mask.sum()
        print(F"{num_censored_packets} detected {num_censored_packets * 100 / util.numberOfRows(df):.2f}%")

        df = df[df["distance"] > 1]

        result[measurement] = {
            "data": df,
            "censored_packets_mask": censored_packets_mask
        }

    print(F"Storing processed censored data to {output_file_path}")
    pickle.dump(result, open(output_file_path, "wb"))

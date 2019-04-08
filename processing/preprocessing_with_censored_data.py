"""
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

           File: preprocessing.py
        Created: 2018-10-30
         Author: Gilles Callebaut
    Description: Preprocessing raw data coming from Arduino Receiver
                 - concatenates different raw csv files
                 - filters data (e.g., values withouth GPS)
                 - removes unneeded data
                 - sorts by time
                 - distance to transmitters (CENTER)
                 - PL
                Stores a pickle file in ../result
"""
import datetime
import glob
import json
import math
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import util as util

pd.options.mode.chained_assignment = None  # default='warn'

HEADER_LONG = ["time", "sat", "satValid", "hdopVal", "hdopValid", "vdopVal", "pdopVal", "lat", "lon", "locValid", "age",
               "ageValid", "alt", "altValid", "course", "courseValid", "speed", "speedValid", "rssi", "snr",
               "freqError", "sf", "isPacket"]
HEADER_SHORT = ["time", "sat", "satValid", "hdopVal", "hdopValid", "vdopVal", "pdopVal", "lat", "lon", "locValid",
                "age",
                "ageValid", "alt", "altValid", "rssi", "snr", "sf", "isPacket"]

update_interval = {
    12: 8,
    9: 2,
    7: 1
}

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))
output_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))

with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)
    measurements = config["measurements"]
    for measurement in measurements:
        path_to_measurement = os.path.join(path_to_measurements, measurement, 'all')
        all_files = glob.glob(os.path.join(path_to_measurement, "*.csv"))

        print("--------------------- PREPROCESSING {} ---------------------".format(measurement))
        size = 0
        df_from_each_file = []
        for idx, file in enumerate(all_files):
            size += os.path.getsize(file)

            df = pd.read_csv(file, sep=',', header=None, names=None)
            df.columns = HEADER_LONG if (df.shape[1] == len(HEADER_LONG)) else HEADER_SHORT
            df['file'] = os.path.basename(file)
            df_from_each_file.append(df)

        df = pd.concat(df_from_each_file, ignore_index=True, sort=False)
        print(" Reading {0} files {1:.2f} kB".format(len(all_files), size / 1024))
        total_rows = df.shape[0]

        df = util.filter(df)

        conf_file = os.path.join(path_to_measurement, 'conf.json')

        CENTER = config[measurement]["center"]

        util.addDistanceTo(df, CENTER)
        df = df[df['distance'] < 100 * 1000]
        util.addPathLossTo(df)

        df.set_index('time', inplace=True)
        df.sort_values('time')

        df_copy = df.copy()

        df_files = df_copy.groupby(['file'])
        for name, df_file in df_files:

            interval_list = []
            # create empty data frame to contain all values for one interval
            interval_df = pd.DataFrame(columns=df.columns)

            tmp_sf = df_file.iloc[0].sf

            # add rows together which have a the same SF (without others in between)
            # rows include GPS and LoRa packets
            for index, row in df_file.iterrows():
                if int(row.sf) != int(tmp_sf):
                    interval_list.append(interval_df)
                    interval_df = pd.DataFrame(columns=df.columns)

                interval_df = interval_df.append(row)
                tmp_sf = row.sf

            interval_list.append(interval_df)

            censored_packets = []
            # print(F"Checking {len(interval_list)} intervals")

            for interval in interval_list:

                interval.index = pd.to_datetime(interval.index)
                interval = interval.sort_index()
                unique_interval = interval[~interval.index.duplicated(keep='first')]
                # for each interval see if we find censored data
                packets = interval.query('isPacket == 1')
                # print(F"\t checking {len(packets)} in interval")
                if len(packets.index) < 1:
                    time_interval = update_interval[12]
                    last_time = unique_interval.last_valid_index()
                    first_time = unique_interval.first_valid_index()
                    time_diff = (last_time - first_time).total_seconds()
                    if time_diff >= 2 * time_interval:
                        # start + update_interval -> isPacket = 2 for each censored point
                        num_censored_points = math.floor(time_diff / time_interval) - 1
                        for i in range(num_censored_points):
                            timestamp = first_time + datetime.timedelta(seconds=(i + 1) * time_interval)
                            location_nearest = unique_interval.index.get_loc(timestamp, method='nearest')
                            censored_row = unique_interval.iloc[location_nearest]
                            censored_row.isPacket = 2
                            censored_packets.append(censored_row)

                elif len(packets.index) == 1:
                    # use time between packet received and last GPS to determine number of censored packets
                    time_interval = update_interval[12]
                    last_time = unique_interval.last_valid_index()
                    packet_time = packets.first_valid_index()
                    time_diff = (last_time - packet_time).total_seconds()
                    if time_diff >= 2 * time_interval:
                        # start + update_interval -> isPacket = 2 for each censored point
                        num_censored_points = math.floor(time_diff / time_interval) - 1
                        for i in range(num_censored_points):
                            timestamp = packet_time + datetime.timedelta(seconds=(i + 1) * time_interval)
                            location_nearest = unique_interval.index.get_loc(timestamp, method='nearest')
                            censored_row = unique_interval.iloc[location_nearest]
                            censored_row.isPacket = 2
                            censored_packets.append(censored_row)
                else:
                    first = True
                    time_interval = update_interval[int(packets.iloc[0].sf)]
                    for packet_time, packet in packets.iterrows():
                        if not first:
                            time_diff = (packet_time - prev_packet_time).total_seconds()
                            if time_diff >= 2 * time_interval:
                                # start + update_interval -> isPacket = 2 for each censored point
                                num_censored_points = math.floor(time_diff / time_interval) - 1
                                for i in range(num_censored_points):
                                    timestamp = prev_packet_time + datetime.timedelta(seconds=(i + 1) * time_interval)
                                    # use nearest GPS location at that time
                                    # but first remove duplicates
                                    location_nearest = unique_interval.index.get_loc(timestamp, method='nearest')
                                    censored_row = unique_interval.iloc[location_nearest]
                                    censored_row.isPacket = 2
                                    censored_packets.append(censored_row)
                        else:
                            first = False
                        prev_packet_time = packet_time

            if len(censored_packets) > 0:
                censored_packets = pd.DataFrame(censored_packets)
                df = df.append(censored_packets)

        output_file = "preprocessed_data_with_censored_data.pkl"
        output_file_path = os.path.join(output_path, measurement, output_file)
        df.to_pickle(output_file_path)

        print(F" Saving to {output_file_path}")
        print("--------------------- DONE preprocessing.py ---------------------")

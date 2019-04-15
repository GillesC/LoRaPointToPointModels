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

           File: util.py
        Created: 2018-10-26
         Author: Gilles Callebaut
    Description:
"""

import branca.colormap as cm
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def update(df, data) -> pd.DataFrame:
    cond = data['Index'] != df['Index']
    if False in cond:
        df.where(cond=cond, other=pd.DataFrame(data), inplace=True)
    else:
        df = df.append(data, ignore_index=True)
    return df


def get_geojson_grid(df, n, plot_snr):
    """

    :param df:
    :param n:
    :param plot_snr:
    :return:
    """
    if plot_snr:
        values_to_plot_id = 'snr'
        caption = 'snr'
    else:
        values_to_plot_id = 'rss'
        caption = 'rss'
    df['lat_discreteat'] = normalize(
        data=df['lat'], num_bins=n)

    df['lon_discrete'] = normalize(
        data=df['lon'], num_bins=n)

    colors = np.zeros((n, n))
    num_values = np.zeros((n, n))
    transparent_matrix = np.zeros((n, n))

    for idx, row in df.iterrows():
        row_idx = int(row['lat_discreteat'])
        col_idx = int(row['lon_discrete'])
        colors[row_idx][col_idx] += row[values_to_plot_id]
        num_values[row_idx][col_idx] += 1
        transparent_matrix[row_idx][col_idx] = 1

    # values == 0 -> = 1 in order to not divide by
    num_values[num_values < 1] = 1
    colors = colors / num_values
    colors = np.nan_to_num(colors)

    max_lat = df['lat'].max()
    max_lon = df['lon'].max()

    min_lat = df['lat'].min()
    min_lon = df['lon'].min()

    upper_right = [max_lat, max_lon]
    lower_left = [min_lat, min_lon]
    center = [lower_left[0] + (upper_right[0] - lower_left[0]) /
              2, lower_left[1] + (upper_right[1] - lower_left[1]) / 2]

    all_boxes = []

    lat_steps = np.linspace(lower_left[0], upper_right[0], n + 1)
    lon_steps = np.linspace(lower_left[1], upper_right[1], n + 1)

    cmap = plt.cm.magma
    norm = mpl.colors.Normalize(vmin=0, vmax=1)

    lat_stride = lat_steps[1] - lat_steps[0]
    lon_stride = lon_steps[1] - lon_steps[0]

    for row_idx, lat in enumerate(lat_steps[:-1]):
        for col_idx, lon in enumerate(lon_steps[:-1]):
            # Define dimensions of box in grid
            upper_left = [lon, lat + lat_stride]
            upper_right = [lon + lon_stride, lat + lat_stride]
            lower_right = [lon + lon_stride, lat]
            lower_left = [lon, lat]
            # Define json coordinates for polygon
            coordinates = [
                upper_left,
                upper_right,
                lower_right,
                lower_left,
                upper_left
            ]
            color = cmap(norm(colors[row_idx][col_idx]))
            color = mpl.colors.to_hex(color)
            geo_json = {"type": "FeatureCollection",
                        "properties": {
                            "lower_left": lower_left,
                            "upper_right": upper_right
                        },
                        "features": []}
            grid_feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coordinates],
                },
                "properties": {
                    "show": transparent_matrix[row_idx][col_idx],
                    "color": color,
                    "val": norm(colors[row_idx][col_idx])
                }
            }
            geo_json["features"].append(grid_feature)
            all_boxes.append(geo_json)

    min_val = 100
    max_val = -100
    for val in np.nditer(colors):
        min_val = val if val < min_val else min_val
        max_val = val if val > max_val and val < 0 else max_val

    colormap = cm.linear.viridis.scale(
        min_val, max_val)
    colormap.caption = caption.upper()
    return all_boxes, colormap


def numberOfRows(df):
    return len(df.index)


def onlyPackets(df):
    return df[df.isPacket == 1]


def addPathLossTo(df: pd.DataFrame, tp=20, gain=0):
    """
    Calculate the path loss in dB.
    Parameters
    ----------
    df : pd.DataFrame containinng the tp and rssi values
    gain: float antenna gain in dB
    Default 0
    """

    # correction term according to https://www.semtech.com/uploads/documents/DS_SX1276-7-8-9_W_APP_V5.pdf p87
    df["pl_db"] = tp + gain - (df.rss)


def addDistanceTo(df: pd.DataFrame, origin):
    lat = origin[0]
    lon = origin[1]

    """
    Calculate the Haversine distance.

    Parameters
    ----------
    origin : list of float
        [lat, long]
    df : pd.DataFrame
    (excluding altitude)
    """

    radius = 6371 * 1000  # m

    dlat = np.radians(df.lat - lat)
    dlon = np.radians(df.lon - lon)

    a = (np.sin(dlat / 2) * np.sin(dlat / 2) +
         np.cos(np.radians(lat)) * np.cos(np.radians(df.lat)) *
         np.sin(dlon / 2) * np.sin(dlon / 2))
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = (radius * c)

    df['distance'] = d


def filter(data):
    total_rows = data.shape[0]
    current_rows_data = data[data.isPacket > 0].shape[0]
    print(F" Packet points {current_rows_data}/{total_rows} {(current_rows_data / total_rows) * 100:.1f}% rows")

    with_gps_data = data[(data.sat > 0) & data.isPacket > 0].shape[0]
    print(" Packet points with GPS {0}/{1} {2:.1f}% rows".format(
        with_gps_data, current_rows_data, (with_gps_data / current_rows_data) * 100))

    data = data[data.sat > 0]
    data = data[data.ageValid > 0]
    data = data[data.hdopVal < 75]
    data = data[data.vdopVal < 75]
    data = data[data.pdopVal < 75]
    data = data[data.locValid > 0]
    data = data[data.ageValid > 0]

    # print(
    #    F" Packet points with GPS without RSS filtering "
    #    F"{data[data.isPacket > 0].shape[0]}/{current_rows_data} {(data[data.isPacket > 0].shape[0] / current_rows_data) * 100:.1f}% rows")

    data.loc[:, 'time'] = pd.to_datetime(data['time'], format='%m/%d/%Y %H:%M:%S ', utc=True, errors='coerce')
    # removes NaT if datatime conversion was unsuccessful
    data.dropna(subset=['time'], inplace=True)

    data = data[(data.sf == 12) | (data.sf == 9) | (data.sf == 7)]
    # print(" Packet points with GPS with SF filtering {0}/{1} {2:.1f}% rows".format(
    #    data[data.isPacket > 0].shape[0], current_rows_data,
    #    (data[data.isPacket > 0].shape[0] / current_rows_data) * 100))

    # Correction factor as described in https://cdn-shop.adafruit.com/product-files/3179/sx1276_77_78_79.pdf

    # default: Packet Strength (dBm) = -157 + PacketRssi
    # if SNR < 0 : Packet Strength (dBm) = -157 + PacketRssi + PacketSnr * 0.25
    # if RSSI > -100dBm: RSSI = -157 + 16/15 * PacketRssi

    data.loc[:, "rssi"] = data.rssi.astype(float)
    data.loc[data["rssi"] > 20, "rssi"] = -1 * data[data["rssi"] > 20]["rssi"]

    packet_rssi = data["rssi"] + 157
    rssi_correction = packet_rssi

    snr_correction = data.snr.copy()
    snr_correction[snr_correction > 0] = 0
    rssi_correction = packet_rssi + snr_correction * 0.25

    data.loc[:,
    'correction_factor'] = 1  # data.add(pd.DataFrame(np.ones(data.shape[0]), index=data.index, columns=['correction_factor']), fill_value=0)
    data.loc[data["rssi"] > -100, 'correction_factor'] = 16 / 15
    rssi_correction = packet_rssi * data['correction_factor']

    data.loc[:, "rss"] = - 157 + rssi_correction

    data = data[data["rss"] < 20]
    # remove unneeded columns
    cols = ["sat", "freqError", "alt", "satValid", "hdopVal", "hdopValid", "vdopVal", "pdopVal", "locValid", "age",
            "ageValid", "altValid", "course", "courseValid", "speed", "speedValid", "rssi", "correction_factor"]
    cols = [c for c in cols if c in data.columns]
    data.drop(columns=cols, axis=1, inplace=True)

    current_rows_data_after_filtering = data[data.isPacket > 0].shape[0]
    print(" Packet points after filtering {0}/{1} {2:.1f}% rows".format(
        current_rows_data_after_filtering, current_rows_data,
        (current_rows_data_after_filtering / current_rows_data) * 100))

    return data


def normalize(data, min_val=None, max_val=None, num_bins=None):
    """Return a value between [0,1] scaled with respect to min and max values of the array and made discrete

    Parameters
    ----------

    data: pandas DataFrame or Series 

    min_val: minimum value 
    """

    if not min_val:
        min_val = data.min()
    if not max_val:
        max_val = data.max()

    if num_bins:

        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return (((data - min_val) / abs(max_val - min_val)) * (num_bins - 1)).astype(int)
        else:
            return ValueError("data needs to be a panda dataframe or series")
    else:
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            return (data - min_val) / abs(max_val - min_val)
        else:
            return ValueError("data needs to be a panda dataframe or series")

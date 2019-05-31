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
from LatexifyMatplotlib import LatexifyMatplotlib as lm

currentDir = os.path.dirname(os.path.abspath(__file__))
path_to_measurements = os.path.abspath(os.path.join(
    currentDir, '..', 'data'))

input_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result'))
input_file_name = "processed_data_with_censored_data.pkl"
input_file_path = os.path.join(input_path, input_file_name)

lm.latexify()
fig, ax = plt.subplots(figsize=(4, 3))
with open(os.path.join(path_to_measurements, "measurements.json")) as f:
    config = json.load(f)
    measurements = config["measurements"]
    data = pd.read_pickle(input_file_path)

    for measurement in measurements:
        print(F"--------------------- Environment {measurement} ---------------------")

        #if measurement == "seaside":
        if True:
            df = data[measurement]["data"]
            df = df.loc[(df["sf"] == 12), :]
            censored_packets_mask =(df["isPacket"] == 0)

            df_censored = df.loc[censored_packets_mask]
            distances = df_censored.loc[df_censored["distance"] > 1]["distance"]

            # p = sns.kdeplot(distances, cumulative=True, label=measurement, ax=ax)
            # x, y = p.get_lines()[0].get_data()

            # y, x, *_= plt.hist(distances, bins=100, density=True, cumulative=True,histtype='step')

            x = np.sort(distances)
            N = len(distances)
            y = np.array(range(N)) / float(N)

            q1 = np.abs(y - 0.75).argmin()
            q2 = np.abs(y - 0.5).argmin()
            q3 = np.abs(y - 0.25).argmin()

            p = plt.plot(x, y)
            c = p[0].get_color()

            plt.fill_between(x=x, y1=y, color=c, alpha=0.1)
            plt.fill_between(x=x[:q3], y1=y[:q3], color=c, alpha=0.2)
            plt.fill_between(x=x[:q2], y1=y[:q2], color=c, alpha=0.2)
            plt.fill_between(x=x[:q1], y1=y[:q1], color=c, alpha=0.2)

            plt.scatter(x[q1], y[q1], edgecolors=c, facecolors=c)
            plt.scatter(x[q2], y[q2], edgecolors=c, facecolors=c)
            plt.scatter(x[q3], y[q3], edgecolors=c, facecolors=c)

            # plt.hlines(y=y[q1], xmin=0, xmax=x[q1], linestyles="dashed", colors="0.7")
            plt.vlines(x=x[q1], ymin=0, ymax=y[q1], colors=c, alpha=0.1)
            text_label = F"{x[q1]:0.0f}"
            plt.annotate(xy=(x[q1],y[q1]), xytext=(x[q1], y[q1]+0.08), s=text_label, ha='center', arrowprops= None)


            # plt.hlines(y=y[q2], xmin=0, xmax=x[q2], linestyles="dashed", colors="0.7")
            plt.vlines(x=x[q2], ymin=0, ymax=y[q2], colors=c, alpha=0.2)
            text_label = F"{x[q2]:0.0f}"
            plt.annotate(xy=(x[q2], y[q2]), xytext=(x[q2]-20, y[q2] + 0.1), s=text_label, ha='center', arrowprops= None)


            # plt.hlines(y=y[q3], xmin=0, xmax=x[q3], linestyles="dashed", colors="0.7")
            plt.vlines(x=x[q3], ymin=0, ymax=y[q3], colors=c, alpha=0.2)
            text_label = F"{x[q3]:0.0f}"
            plt.annotate(xy=(x[q3], y[q3]), xytext=(x[q3], 0.05), s=text_label, ha='center', arrowprops= None)


            #plt.xticks(list(plt.xticks()[0]) + x[q3] + x[q2] + x[q1])

            
plt.xlabel(r'Distance (m)')
plt.ylabel(r'CDF uncensored samples')

plt.show()

#lm.format_axes(ax)
#lm.save(F"cdf_censored_distances_all.tex", plt=plt)
#
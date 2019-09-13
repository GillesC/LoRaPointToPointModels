
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from LatexifyMatplotlib import LatexifyMatplotlib as lm

currentDir = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.abspath(os.path.join(
    currentDir, '..', 'result', "Leuven"))
df = pickle.load(open(os.path.join(input_path, "leuven-data.pkl"), "rb"))
print(df.head())

fig, ax = plt.subplots()
lm.latexify()
plt.xscale('log')
ax.xaxis.set_major_formatter(ScalarFormatter())
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

x = range(1, int(max(df.distance)))

pl_ml = 74.85 + 2.75 * 10 * np.log10(df.distance)
pl_ols = 60.06 + 3.2 * 10 * np.log10(df.distance)

fc = 868
hb = 1.5
hm = 1.5
Ch = 0.8 + (1.1 * np.log10(fc) - 0.7) * hm - 1.56 * np.log10(fc)
hata = 13.82 * np.log10(hb) - Ch + (44.9 - 6.55 * np.log10(hb)) * np.log10(df.distance / 1000)
pl_hata = 69.55 + 26.16 * np.log(fc) + hata

pl_cost_hata = 46.3 + 33.9 * np.log10(fc) + hata

pl_oulu_car = 128.95 + 10 * 2.32 * np.log10(df.distance / 1000)
pl_oulu_boat = 126.43 + 10 * 1.76 * np.log10(df.distance / 1000)

ax.plot(df.distance, pl_ml, label=r"Our Model (ML)")
ax.plot(df.distance, pl_ols, label=r"Our Model (OLS)")
#
ax.plot(df.distance, pl_oulu_car, label=r"Oulu Model (Car)~\cite{7377400}")
ax.plot(df.distance, pl_oulu_boat, label=r"Oulu Model (Boat)~\cite{7377400}")
#
ax.plot(df.distance, pl_hata, label=r"Okumura Hata Model (Urban small or medium-sized city)")
ax.plot(df.distance, pl_cost_hata, label=r"Cost Hata Model (Suburban/Rural environment)")
ax.scatter(df.distance, df.pl_db, marker='x', label="Measured Path Loss", s=1, c='0.75')

ax.set_xlabel(r'Log distance (m)')
ax.set_ylabel(r'Path Loss (dB)')

lm.format_axes(ax)
lm.legend(plt)
lm.save(F"path-loss-eval-leuven.tex", plt=plt, show=True)

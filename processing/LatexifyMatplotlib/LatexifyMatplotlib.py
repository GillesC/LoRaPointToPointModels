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

           File: LatexifyMatplotlib.py
        Created: 2019-01-10
         Author: Gilles Callebaut
    Description:
    Usage:

"""
import os
from math import sqrt
import matplotlib as mpl
import tikzplotlib

SPINE_COLOR = 'gray'

def latexify(fig_width=None, fig_height=None, columns=1):
    """Set up matplotlib's RC params for LaTeX plotting.
    Call this before plotting a figure.

    Parameters
    ----------
    fig_width : float, optional, inches
    fig_height : float,  optional, inches
    columns : {1, 2}
    """

    # code adapted from http://www.scipy.org/Cookbook/Matplotlib/LaTeX_Examples

    # Width and max height in inches for IEEE journals taken from
    # computer.org/cms/Computer.org/Journal%20templates/transactions_art_guide.pdf

    assert (columns in [1, 2])

    if fig_width is None:
        fig_width = 2.7 if columns == 1 else 6.9  # width in inches

    if fig_height is None:
        golden_mean = (sqrt(5) - 1.0) / 2.0  # Aesthetic ratio
        fig_height = fig_width * golden_mean  # height in inches

    MAX_HEIGHT_INCHES = 8.0
    if fig_height > MAX_HEIGHT_INCHES:
        print("WARNING: fig_height too large:" + fig_height +
              "so will reduce to" + MAX_HEIGHT_INCHES + "inches.")
        fig_height = MAX_HEIGHT_INCHES

    params = {
        'backend': 'ps',
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'legend.fontsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'text.usetex': True,
        "pgf.rcfonts": False,
        "pgf.texsystem": "lualatex",
        'figure.figsize': [fig_width, fig_height],
        'font.family': 'serif',
        "pgf.preamble": [
            r"\usepackage[utf8x]{inputenc}",  # use utf8 fonts
            r"\usepackage[T1]{fontenc}",  # plots will be generated
            r"\usepackage[detect-all]{siunitx}",  # to use si units,
            r"\DeclareSIUnit{\belmilliwatt}{Bm}",
            r"\DeclareSIUnit{\dBm}{\deci\belmilliwatt}",
            r"\usepackage{booktabs}",
            r"\renewcommand{\arraystretch}{1.2}"
        ]
    }

    mpl.rcParams.update(params)


def format_axes(ax):
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color(SPINE_COLOR)
        ax.spines[spine].set_linewidth(0.5)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_tick_params(direction='out', color=SPINE_COLOR)

    return ax


def legend(plt):
    plt.legend(framealpha=0.0)


def save(filename, scale_legend=None, show=False, plt=None):
    #assert show == (plt is not None), "Add plt argument in order to show the figure"

    ext = filename.rsplit('.', 1)[-1]

    assert ext == "tex", ValueError("Only tex extensions are allowed")

    current_dir = os.path.dirname(os.path.abspath(__file__))

    output_path = os.path.abspath(os.path.join(
        current_dir, '..', '..', 'result'))

    out = os.path.join(output_path, filename)

    print(F"Saving to: {out} with extension {ext}")

    extra_axis_param = [
        r"width = \linewidth",
        r"axis lines* = {left}"
    ]

    if scale_legend:
        extra_axis_param.extend([r"legend style={nodes={scale="+scale_legend+", transform shape}}"])

    fig = plt.gcf()

    if show:
        plt.show()

    print(r"Replace 'table' with 'table[row sep=\\]' in the tex file. I have opened an issue in matplotlib2tikz; let's hope that this is resolved in a future release")
    print("If you have still problems, you will probably need to add some newlines manually in the file (because the inline is too long)")

    tikzplotlib.save(out, figure=fig, textsize=8, extra_axis_parameters=extra_axis_param, float_format="{:.5f}", table_row_sep=r"\\")



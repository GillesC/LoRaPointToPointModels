import numpy as np


def histcounts(x, num_bins=None):
    if num_bins is None:
        bins = 'auto'
    else:
        bins = num_bins

    hist, bin_edges = np.histogram(x, bins=bins)
    inds = np.digitize(x, bin_edges[:-1])

    return hist, inds - 1, bin_edges[:-1]


def get_weights(d, weight_type, num_bins=100):
    supported_weights = ['linear', 'log10', 'square']
    assert weight_type in supported_weights, ValueError(
        F"Weight {weight_type} is not implemented, use {supported_weights}")

    total_points = len(d)

    if weight_type is 'linear':
        x = d

    elif weight_type is 'log10':
        x = np.log10(d)

    elif weight_type is 'square':
        x = d ** 2

    hist, inds, *_ = histcounts(x, num_bins=num_bins)

    n_b = hist[inds]
    w = (1.0 / n_b) * (total_points / num_bins)

    # limit the < 1% of the bins being over sampled
    avg_number_sampel_per_bin = total_points/num_bins


    x_percent_limit = []
    for ix, h in enumerate(hist):
        # less than 5 % of the average
        if h < avg_number_sampel_per_bin * 0.05:
            x_percent_limit.append(ix)

    w = [1 if ix in x_percent_limit else w[ix] for ix in inds]
    return np.array(w)


def get_apparent_data_points(d, weight_type, num_bins=100):
    if weight_type is None:
        w = np.ones(d.shape)
    else:
        w = get_weights(d, weight_type=weight_type, num_bins=num_bins)

    hist_val, bin_edges = np.histogram(d, bins=num_bins, weights=w)
    return hist_val, bin_edges[:-1]

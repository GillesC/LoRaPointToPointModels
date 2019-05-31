import numpy as np
import scipy.optimize
import statsmodels.api as sm
from scipy.stats import norm
import warnings

warnings.filterwarnings("error")


def ml(d0, d, pld, c, pld0, n, sigma, censored_mask=None, weights=None, censored=True):
    assert (censored_mask is not None) == censored, ValueError("Define the censored_mask if censored data is included")

    if weights is None:
        w = np.ones(d.shape)
    else:
        w = weights

    if censored:
        uncensored_mask = np.invert(censored_mask)
        pld_uncensored = pld[uncensored_mask]
    else:
        pld_uncensored = pld

    def llf(x):
        _n = x[0]
        _sigma = x[1:]  # in case of _sigma ~d -> a and b

        # if _n < 1:
        #    return float('inf')

        if len(_sigma) == 2:
            _a = _sigma[0]
            _b = _sigma[1]
            _sigma = _a * np.log10(d / d0) + _b
        else:
            _sigma = np.repeat(_sigma, d.shape)

        plm = 10 * _n * np.log10(d / d0) + pld0

        _sigma[_sigma < 0.001] = 0.001

        if censored:
            plm_uncensored = plm[uncensored_mask].copy()
            w_uncensored = w[uncensored_mask].copy()
            sigma_uncensored = _sigma[uncensored_mask].copy()
            sigma_censored = _sigma[censored_mask].copy()
        else:
            plm_uncensored = plm
            w_uncensored = w
            sigma_uncensored = _sigma.copy()

        norm_pdf = norm.pdf((pld_uncensored - plm_uncensored) / sigma_uncensored)
        norm_pdf[norm_pdf < 0.0000001] = 0.0000001
        llh = np.multiply(w_uncensored, -np.log(sigma_uncensored) + np.log(norm_pdf))

        if censored:
            llh_censored = np.multiply(w[censored_mask],
                                       (np.log(1 - norm.cdf((c[censored_mask] - plm[censored_mask]) / sigma_censored))))
            llh = np.append(arr=llh, values=llh_censored)

        return - np.sum(llh)

    x0 = np.array([n])
    x0 = np.append(x0, sigma)
    return scipy.optimize.fmin(llf, x0, maxiter=20000, maxfun=20000, disp=False, full_output=True)


def ols(d0, d, pld):
    x = 10 * np.log10(d / d0)
    x = sm.add_constant(x)
    y = pld

    results = sm.OLS(y, x).fit()

    pld0 = results.params[0]
    n = results.params[1]

    epl = n * 10 * np.log10(d / d0) + pld0
    sigma = np.std(epl - pld)
    return pld0, n, sigma

"""
def ml_dual_slope(d0, d, pld, x0, pld0, c=148, censored_mask=None, weights=None, censored=True, fixed_d_break=False):
    assert (censored_mask is not None) == censored, ValueError("Define the censored_mask if censored data is included")

    if fixed_d_break:
        # [pld0_ols, n_ols, n_ols, d_break, sigma_ols]
        x0 = [x0[0], x0[1], x0[2], x0[4]]
        d_break = x0[3]

    if weights is None:
        w = np.ones(d.shape)
    else:
        w = weights

    if censored:
        uncensored_mask = np.invert(censored_mask)
        pld_uncensored = pld[uncensored_mask]
    else:
        pld_uncensored = pld

    def llf(x):
        _pld0 = x[0]
        _n_1 = x[1]
        _n_2 = x[2]

        # if _n_1 < 1 or _n_2 < 1:
        #    return float('inf')

        if fixed_d_break:
            _sigma = x[3:]
            _d_break = d_break
        else:
            _d_break = x[3]
            _sigma = x[4:]  # in case of _sigma ~d -> a and b

        mask_below_d_break = d < _d_break
        mask_above_d_break = np.invert(mask_below_d_break)

        if len(_sigma) == 3:
            # raise ValueError("Non constant sigma not yet supported")
            _a1 = _sigma[0]
            _b = _sigma[1]
            _a2 = _sigma[2]

            _sigma = np.zeros(d.shape)
            _sigma[mask_below_d_break] = _a1 * np.log10(d[mask_below_d_break] / d0) + _b
            _sigma[mask_above_d_break] = _a1 * np.log10(_d_break / d0) + _a2 * np.log10(
                d[mask_above_d_break] / _d_break) + _b
        else:
            _sigma = np.ones(d.shape) * _sigma

        plm = np.zeros(d.shape)

        plm[mask_below_d_break] = 10 * _n_1 * np.log10(d[mask_below_d_break] / d0) + _pld0
        plm[mask_above_d_break] = 10 * _n_1 * np.log10(_d_break / d0) + 10 * _n_2 * np.log10(
            d[mask_above_d_break] / _d_break) + _pld0

        if len(_sigma[_sigma < 0]) > 1:
            return float('inf')

        _sigma[_sigma < 0.001] = 0.001

        if censored:
            plm_uncensored = plm[uncensored_mask]
            w_uncensored = w[uncensored_mask]
            sigma_uncensored = _sigma[uncensored_mask]
        else:
            plm_uncensored = plm
            w_uncensored = w
            sigma_uncensored = _sigma

        llh = np.multiply(w_uncensored,
                          (-np.log(sigma_uncensored) + np.log(
                              norm.pdf((pld_uncensored - plm_uncensored) / sigma_uncensored))))

        if censored:
            llh_censored = np.multiply(w[censored_mask],
                                       (np.log(1 - norm.cdf((c[censored_mask] - plm[censored_mask]) / _sigma[censored_mask]))))
            llh = np.append(arr=llh, values=llh_censored)

        return - np.sum(llh)

    return scipy.optimize.fmin(llf, x0, maxiter=20000, maxfun=20000, disp=False, full_output=True)
"""

def ml_value(pld, plm_est, sigma_est, censored_mask=None, c=148):
    if censored_mask is not None:
        uncensored_mask = np.invert(censored_mask)
        llh = -np.log(sigma_est[uncensored_mask]) + np.log(
            norm.pdf((pld[uncensored_mask] - plm_est[uncensored_mask]) / sigma_est[uncensored_mask]))
        llh_censored = np.log(1 - norm.cdf((c[censored_mask] - plm_est[censored_mask]) / sigma_est[censored_mask]))
        llh = np.append(arr=llh, values=llh_censored)
    else:
        norm_pdf = norm.pdf((pld - plm_est) / sigma_est)
        assert not (norm_pdf.min() < 0), ValueError(F"PDF can not be less than zero: min. value is {norm_pdf.min()}")
        llh = -np.log(sigma_est) + np.log(norm_pdf)

    return np.sum(np.exp(llh))


def ml_with_constraints(d0, d, pld, c, pld0, n, sigma, censored_mask=None, weights=None, censored=True):
    assert (censored_mask is not None) == censored, ValueError("Define the censored_mask if censored data is included")

    def con(x):
        _sigma = x[1:]
        _n = x[0]
        _a = 0
        _b = 0

        if len(_sigma) == 2:
            _a = _sigma[0]
            _b = _sigma[1]
            _sigma = _a * np.log10(d / d0) + _b
        else:
            _sigma = np.repeat(_sigma, len(d))

        return min(_a, _sigma.min(), _b, _n)

    cons = [{
        'type': 'ineq',
        'fun': con
    }]

    if weights is None:
        w = np.ones(d.shape)
    else:
        w = weights

    if censored:
        uncensored_mask = np.invert(censored_mask)
        pld_uncensored = pld[uncensored_mask]
    else:
        pld_uncensored = pld


    def llf(x):
        _n = x[0]
        _sigma = x[1:]  # in case of _sigma ~d -> a and b

        # if _n < 1:
        #    return float('inf')

        if len(_sigma) == 2:
            _a = _sigma[0]
            _b = _sigma[1]
            _sigma = _a * np.log10(d / d0) + _b
        else:
            _sigma = np.repeat(_sigma, d.shape)

        plm = 10 * _n * np.log10(d / d0) + pld0

        if censored:
            plm_uncensored = plm[uncensored_mask].copy()
            w_uncensored = w[uncensored_mask].copy()
            sigma_uncensored = _sigma[uncensored_mask].copy()
            sigma_censored = _sigma[censored_mask].copy()
        else:
            plm_uncensored = plm
            w_uncensored = w
            sigma_uncensored = _sigma.copy()

        norm_pdf = norm.pdf((pld_uncensored - plm_uncensored) / sigma_uncensored)
        norm_pdf[norm_pdf < 0.0000001] = 0.0000001
        llh = np.multiply(w_uncensored, -np.log(sigma_uncensored) + np.log(norm_pdf))

        if censored:
            llh_censored = np.multiply(w[censored_mask],
                                       (np.log(1 - norm.cdf((c[censored_mask] - plm[censored_mask]) / sigma_censored))))
            llh = np.append(arr=llh, values=llh_censored)

        return - np.sum(llh)

    x0 = np.array([n])
    x0 = np.append(x0, sigma)
    return scipy.optimize.minimize(llf, x0, constraints=cons, method='COBYLA', options={
        'maxiter': 200000
    })

"""
def ml_dual_slope_with_constraints(d0, d, pld, x0, c=148, censored_mask=None, weights=None, censored=True):
    assert len(x0) == 5, ValueError("Only constant variance is now supported")

    def con(x):
        _n_1 = x[1]
        _n_2 = x[2]
        _sigma = x[4]
        return min(_n_1, _sigma.min(), _n_2)

    cons = [{
        'type': 'ineq',
        'fun': con
    }]

    if weights is None:
        w = np.ones(d.shape)
    else:
        w = weights

    if censored:
        uncensored_mask = np.invert(censored_mask)
        pld_uncensored = pld[uncensored_mask]
    else:
        pld_uncensored = pld

    def llf(x):

        _pld0 = x[0]
        _n_1 = x[1]
        _n_2 = x[2]
        _d_break = x[3]
        _sigma = x[4:]  # in case of _sigma ~d -> a and b


        mask_below_d_break = d < _d_break
        mask_above_d_break = np.invert(mask_below_d_break)

        if len(_sigma) == 3:
            # raise ValueError("Non constant sigma not yet supported")
            _a1 = _sigma[0]
            _b = _sigma[1]
            _a2 = _sigma[2]

            _sigma = np.zeros(d.shape)
            _sigma[mask_below_d_break] = _a1 * np.log10(d[mask_below_d_break] / d0) + _b
            _sigma[mask_above_d_break] = _a1 * np.log10(_d_break / d0) + _a2 * np.log10(
                d[mask_above_d_break] / _d_break) + _b
        else:
            _sigma = np.ones(d.shape) * _sigma

        plm = np.zeros(d.shape)

        plm[mask_below_d_break] = 10 * _n_1 * np.log10(d[mask_below_d_break] / d0) + _pld0
        plm[mask_above_d_break] = 10 * _n_1 * np.log10(_d_break / d0) + 10 * _n_2 * np.log10(
            d[mask_above_d_break] / _d_break) + _pld0

        if censored:
            plm_uncensored = plm[uncensored_mask]
            w_uncensored = w[uncensored_mask]
            sigma_uncensored = _sigma[uncensored_mask]
        else:
            plm_uncensored = plm
            w_uncensored = w
            sigma_uncensored = _sigma

        llh = np.multiply(w_uncensored,
                          (-np.log(sigma_uncensored) + np.log(
                              norm.pdf((pld_uncensored - plm_uncensored) / sigma_uncensored))))

        if censored:
            llh_censored = np.multiply(w[censored_mask],
                                       (np.log(1 - norm.cdf((c[censored_mask] - plm[censored_mask]) / _sigma[censored_mask]))))
            llh = np.append(arr=llh, values=llh_censored)

        return - np.sum(llh)

    return scipy.optimize.minimize(llf, x0, constraints=cons, method='COBYLA', options={
        'maxiter': 20000000
    })
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression


def kernel_wts(dist, center, bw, kernel="triangular"):
    """Kernel weights function (triangular kernel)"""
    if kernel == "triangular":
        w = 1 - np.abs(dist) / bw
        w = np.clip(w, 0, None)  # Ensure non-negative weights
    return w


def DCdensity(runvar, cutpoint=0, bin_size=None, bw=None, verbose=False, plot=True,
              ext_out=False, htest=False):
    runvar = runvar[~np.isnan(runvar)]

    # Grab some summary vars
    rn = len(runvar)
    rsd = np.std(runvar)
    rmin = np.min(runvar)
    rmax = np.max(runvar)

    if cutpoint <= rmin or cutpoint >= rmax:
        raise ValueError("Cutpoint must lie within range of runvar")

    if bin_size is None:
        bin_size = 2 * rsd * rn ** (-1 / 2)
        if verbose:
            print(f"Using calculated bin_size size: {bin_size:.3f}")

    l = np.floor((rmin - cutpoint) / bin_size) * bin_size + bin_size / 2 + cutpoint
    r = np.floor((rmax - cutpoint) / bin_size) * bin_size + bin_size / 2 + cutpoint
    lc = cutpoint - (bin_size / 2)
    rc = cutpoint + (bin_size / 2)
    j = int(np.floor((rmax - rmin) / bin_size) + 2)

    bin_sizenum = np.round(((np.floor(
        (runvar - cutpoint) / bin_size) * bin_size + bin_size / 2 + cutpoint) - l) / bin_size) + 1
    cellval = np.zeros(j)

    for i in range(rn):
        cnum = int(bin_sizenum[i])
        cellval[cnum] += 1
    cellval = (cellval / rn) / bin_size

    cellmp = np.floor(
        (l + (np.arange(1, j + 1) - 1) * bin_size) - cutpoint) / bin_size * bin_size + bin_size / 2 + cutpoint

    # If no bandwidth is given, calculate it
    if bw is None:
        leftofc = int(np.round(((np.floor(
            (lc - cutpoint) / bin_size) * bin_size + bin_size / 2 + cutpoint) - l) / bin_size) + 1)
        rightofc = int(np.round(((np.floor(
            (rc - cutpoint) / bin_size) * bin_size + bin_size / 2 + cutpoint) - l) / bin_size) + 1)

        if rightofc - leftofc != 1:
            raise ValueError("Error occurred in bandwidth calculation")

        cellmpleft = cellmp[:leftofc]
        cellmpright = cellmp[rightofc:]

        # Estimate 4th order polynomial to the left
        P_lm = LinearRegression().fit(cellmp[cellmp < cutpoint].reshape(-1, 1),
                                      cellval[cellmp < cutpoint])
        mse4 = np.mean((P_lm.predict(
            cellmp[cellmp < cutpoint].reshape(-1, 1)) - cellval[
                            cellmp < cutpoint]) ** 2)

        fppleft = 2 * P_lm.coef_[0] + 6 * P_lm.coef_[1] * cellmpleft + 12 * \
                  P_lm.coef_[2] * cellmpleft ** 2
        hleft = 3.348 * (mse4 * (cutpoint - l) / np.sum(fppleft ** 2)) ** (
                    1 / 5)

        # Estimate 4th order polynomial to the right
        P_lm = LinearRegression().fit(
            cellmp[cellmp >= cutpoint].reshape(-1, 1),
            cellval[cellmp >= cutpoint])
        mse4 = np.mean((P_lm.predict(
            cellmp[cellmp >= cutpoint].reshape(-1, 1)) - cellval[
                            cellmp >= cutpoint]) ** 2)

        fppright = 2 * P_lm.coef_[0] + 6 * P_lm.coef_[1] * cellmpright + 12 * \
                   P_lm.coef_[2] * cellmpright ** 2
        hright = 3.348 * (mse4 * (r - cutpoint) / np.sum(fppright ** 2)) ** (
                    1 / 5)

        bw = 0.5 * (hleft + hright)
        if verbose:
            print(f"Using calculated bandwidth: {bw:.3f}")

    if np.sum((runvar > cutpoint - bw) & (runvar < cutpoint)) == 0 or np.sum(
            (runvar < cutpoint + bw) & (runvar >= cutpoint)) == 0:
        raise ValueError("Insufficient data within the bandwidth.")

    if plot:
        # Estimate density to either side of the cutpoint using a triangular kernel
        d_l = pd.DataFrame({'cellmp': cellmp[cellmp < cutpoint],
                            'cellval': cellval[cellmp < cutpoint],
                            'dist': np.nan, 'est': np.nan, 'lwr': np.nan,
                            'upr': np.nan})
        pmin = cutpoint - 2 * rsd
        pmax = cutpoint + 2 * rsd

        for i in range(len(d_l)):
            d_l['dist'] = d_l['cellmp'] - d_l.loc[i, 'cellmp']
            w = kernel_wts(d_l['dist'], 0, bw, kernel="triangular")
            model = LinearRegression().fit(d_l[['dist']], d_l['cellval'],
                                           sample_weight=w)
            pred = model.predict([[0]])
            d_l.loc[i, 'est'] = pred[0]
            d_l.loc[i, 'lwr'], d_l.loc[i, 'upr'] = pred[0] - 1.96, pred[
                0] + 1.96

        d_r = pd.DataFrame({'cellmp': cellmp[cellmp >= cutpoint],
                            'cellval': cellval[cellmp >= cutpoint],
                            'dist': np.nan, 'est': np.nan, 'lwr': np.nan,
                            'upr': np.nan})
        for i in range(len(d_r)):
            d_r['dist'] = d_r['cellmp'] - d_r.loc[i, 'cellmp']
            w = kernel_wts(d_r['dist'], 0, bw, kernel="triangular")
            model = LinearRegression().fit(d_r[['dist']], d_r['cellval'],
                                           sample_weight=w)
            pred = model.predict([[0]])
            d_r.loc[i, 'est'] = pred[0]
            d_r.loc[i, 'lwr'], d_r.loc[i, 'upr'] = pred[0] - 1.96, pred[
                0] + 1.96

        # Plot to the left and right
        plt.plot(d_l['cellmp'], d_l['est'], label="Estimate Left", lw=2,
                 color='black')
        plt.fill_between(d_l['cellmp'], d_l['lwr'], d_l['upr'], color='gray',
                         alpha=0.5)
        plt.plot(d_r['cellmp'], d_r['est'], label="Estimate Right", lw=2,
                 color='black')
        plt.fill_between(d_r['cellmp'], d_r['lwr'], d_r['upr'], color='gray',
                         alpha=0.5)
        plt.scatter(cellmp, cellval, color='black')
        plt.xlim([pmin, pmax])
        plt.show()

    # Calculate and display discontinuity estimate
    dist = cmp - cutpoint
    w = 1 - np.abs(dist) / bw
    w = np.where(w > 0, w * (cmp < cutpoint), 0)
    w = (w / np.sum(w)) * j
    fhatl = \
    LinearRegression().fit(dist.reshape(-1, 1), cval, sample_weight=w).predict(
        [[0]])[0]

    w = 1 - np.abs(dist) / bw
    w = np.where(w > 0, w * (cmp >= cutpoint), 0)
    w = (w / np.sum(w)) * j
    fhatr = \
    LinearRegression().fit(dist.reshape(-1, 1), cval, sample_weight=w).predict(
        [[0]])[0]

    thetahat = np.log(fhatr) - np.log(fhatl)
    sethetahat = np.sqrt(
        (1 / (rn * bw)) * (24 / 5) * ((1 / fhatr) + (1 / fhatl)))
    z = thetahat / sethetahat
    p = 2 * stats.norm.sf(np.abs(z))

    if verbose:
        print(f"Log difference in heights is {thetahat:.3f} with SE {sethet

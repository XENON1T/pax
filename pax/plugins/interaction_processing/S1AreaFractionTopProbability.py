import numpy as np

from pax import plugin, utils
from pax.InterpolatingMap import InterpolatingMap

from scipy.special import betainc, gammaln


def bdtrc(k, n, p):
        if (k < 0):
            return (1.0)

        if (k == n):
            return (0.0)
        dn = n - k
        if (k == 0):
            if (p < .01):
                dk = -np.expm1(dn * np.log1p(-p))
            else:
                dk = 1.0 - np.exp(dn * np.log(1.0-p))
        else:
            dk = k + 1
            dk = betainc(dk, dn, p)
        return dk


def bdtr(k, n, p):
        if (k < 0):
            return np.nan

        if (k == n):
            return (1.0)

        dn = n - k
        if (k == 0):
            dk = np.exp(dn*np.log(1.0 - p))
        else:
            dk = k + 1
            dk = betainc(dn, dk, 1.0 - p)
        return dk


def binom_pmf(k, n, p):
    scale_log = gammaln(n+1) - gammaln(n-k+1) - gammaln(k+1)
    ret_log = scale_log + k*np.log(p) + (n-k)*np.log(1-p)
    return np.exp(ret_log)


def binom_cdf(k, n, p):
        return bdtr(k, n, p)


def binom_sf(k, n, p):
    return bdtrc(k, n, p)


def binom_test(k, n, p):
    '''
    The main purpose of this algorithm is to find the value j on the
    other side of the mean that has the same probability as k, and
    integrate the tails outward from k and j. In the case where either
    k or j are zero, only the non-zero tail is integrated.
    '''
    if n < k:
        raise ValueError("n must be >= k")
    if (p > 1.0) or (p < 0.0):
        raise ValueError("p must be in range [0, 1]")
    if k < 0:
        raise ValueError("k must be >= 0")

    d = binom_pmf(k, n, p)
    rerr = 1 + 1e-7
    d = d*rerr
    n_iter = int(max(np.round(np.log10(n))+1, 2))
    if k < n*p:
        j_min, j_max = n*p, n

        def check(d, y0, y1):
            return ((y0 >= d) and (d > y1))

    else:
        if binom_pmf(0, n, p) > d:
            j_min = j_max = 0
            n_iter = 0
        else:
            j_min, j_max = 0, n*p

            def check(d, y0, y1):
                return ((y0 <= d) and (d < y1))

    for _ in range(n_iter):  # successive approximation loop
        j_range = np.linspace(j_min, j_max, 10, endpoint=True)
        y = binom_pmf(j_range, n, p)
        for i in range(len(j_range)-1):
            if check(d, y[i], y[i+1]):
                j_min, j_max = j_range[i], j_range[i+1]
                break
    j = max(min((j_min + j_max)/2, n), 0)

    if k*j == 0:  # one is zero, means we do a one-sided test
        pval = binom_sf(max(k, j), n, p)
    else:
        pval = binom_cdf(min(k, j), n, p) + binom_sf(max(k, j), n, p)
    return min(1.0, pval)


class S1AreaFractionTopProbability(plugin.TransformPlugin):
    """Computes p-value for S1 area fraction top for each interaction
    """

    def startup(self):
        aftmap_filename = utils.data_file_name('s1_aft_xyz_XENON1T_20170808.json')
        self.aft_map = InterpolatingMap(aftmap_filename)
        self.low_pe_threshold = 10  # below this in PE, transition to hits

    def transform_event(self, event):
        for ia in event.interactions:
            s1 = event.peaks[ia.s1]

            if s1.area < self.low_pe_threshold:
                s1_frac = s1.area/self.low_pe_threshold
                hits_top = s1.n_hits*s1.hits_fraction_top
                s1_top = s1.area*s1.area_fraction_top
                size_top = hits_top*(1.-s1_frac) + s1_top*s1_frac
                size_tot = s1.n_hits*(1.-s1_frac) + s1.area*s1_frac
            else:
                size_top = s1.area*s1.area_fraction_top
                size_tot = s1.area

            aft = self.aft_map.get_value(ia.x, ia.y, ia.z)
            ia.s1_area_fraction_top_probability = binom_test(size_top, size_tot, aft)

        return event

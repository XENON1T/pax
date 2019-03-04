import numba
import numpy as np

from pax import units, exceptions
from pax.datastructure import Hit


@numba.jit(numba.int64[:](numba.from_dtype(Hit.get_dtype())[:]),
           nopython=True)
def gaps_between_hits(hits):
    """Return array of gaps between hits: a hit's 'gap' is the # of samples before that hit free of other hits.
    The gap of the first hit is 0 by definition.
    Hits should already be sorted by index of maximum; we'll check this and throw an error if not.
    """
    n_hits = len(hits)
    gaps = np.zeros(n_hits, dtype=np.int64)
    if n_hits == 0:
        return gaps
    # Keep a running right boundary
    boundary = hits[0].index_of_maximum
    for i, hit in enumerate(hits[1:]):
        gaps[i + 1] = max(0, hit.index_of_maximum - boundary - 1)
        if hit.index_of_maximum < boundary:
            raise ValueError("Hits should be sorted by index_of_maximum")
        boundary = max(hit.index_of_maximum, boundary)
    return gaps


def count_hits_per_channel(peak, config, weights=None):
    return np.bincount(peak.hits['channel'].astype(np.int16), minlength=config['n_channels'], weights=weights)


def saturation_correction(peak, channels_in_pattern, expected_pattern, confused_channels, log):
    """Return multiplicative area correction obtained by replacing area in confused_channels by
    expected area based on expected_pattern in channels_in_pattern.
    expected_pattern does not have to be normalized: we'll do that for you.
    We'll also ensure any confused_channels not in channels_in_pattern are ignored.
    """
    try:
        confused_channels = np.intersect1d(confused_channels, channels_in_pattern).astype(np.int)
    except exceptions.CoordinateOutOfRangeException:
        log.warning("Expected area fractions for peak %d-%d are zero -- "
                    "cannot compute saturation & zombie correction!" % (peak.left, peak.right))
        return 1
    # PatternFitter should have normalized the pattern
    assert abs(np.sum(expected_pattern) - 1) < 0.01

    area_seen_in_pattern = peak.area_per_channel[channels_in_pattern].sum()
    area_in_good_channels = area_seen_in_pattern - peak.area_per_channel[confused_channels].sum()
    fraction_of_pattern_in_good_channels = 1 - expected_pattern[confused_channels].sum()

    # Area in channels not in channels_in_pattern is left alone
    new_area = peak.area - area_seen_in_pattern

    # Estimate the area in channels_in_pattern by excluding the confused channels
    new_area += area_in_good_channels / fraction_of_pattern_in_good_channels

    return new_area / peak.area


def adc_to_pe(config, channel, use_reference_gain=False, use_reference_gain_if_zero=False):
    """Gives the conversion factor from ADC counts above baseline to pe/bin
    Use as: w_in_pe_bin = adc_to_pe(config, channel) * w_in_adc_above_baseline
      - config should be a configuration dictionary (self.config in a pax plugin)
      - If use_reference_gain is True, will always use config.get('pmt_reference_gain', 2e6) rather than the pmt gain
      - If use_reference_gain_if_zero=True will do the above only if channel gain is 0.
    If neither of these are true, and gain is 0, will return 0.
    """
    c = config
    adc_to_e = c['sample_duration'] * c['digitizer_voltage_range'] / (
        2 ** (c['digitizer_bits']) *
        c['pmt_circuit_load_resistor'] *
        c['external_amplification'] *
        units.electron_charge)
    try:
        pmt_gain = c['gains'][channel]
    except IndexError:
        print("Attempt to request gain for channel %d, only %d channels known. "
              "Returning reference gain instead." % (channel, len(c['gains'])))
        return c.get('pmt_reference_gain', 2e6)
    if use_reference_gain_if_zero and pmt_gain == 0 or use_reference_gain:
        pmt_gain = c.get('pmt_reference_gain', 2e6)
    if pmt_gain == 0:
        return 0
    return adc_to_e / pmt_gain


def get_detector_by_channel(config):
    """Return a channel -> detector lookup dictionary from a configuration"""
    detector_by_channel = {}
    for name, chs in config['channels_in_detector'].items():
        for ch in chs:
            detector_by_channel[ch] = name
    return detector_by_channel


@numba.jit(numba.void(numba.float64[:], numba.int64[:, :], numba.int64, numba.int64),
           nopython=True)
def extend_intervals(w, intervals, left_extension, right_extension):
    """Extends intervals on w by left_extension to left and right_extension to right, never exceeding w's bounds
    :param w: Waveform intervals live on. Only used for edges (kind of pointless to pass...)
    :param intervals: numpy N*2 array of ints of interval bounds
    :param left_extension: Extend intervals left by this number of samples,
                           or as far as possible until the end of another interval / the end of w.
    :param right_extension: Same, extend to right.
    :return: None, modifes intervals in place
    When two intervals' extension claims compete, right extension has priority.
    Boundary indices are inclusive, i.e. without any extension settings, the right boundary is the last index
    which was still above low_threshold
    """
    n_intervals = len(intervals)
    last_index_in_w = len(w) - 1

    # Right extension
    if right_extension != 0:
        for i in range(n_intervals):
            if i == n_intervals - 1:
                max_possible_r = last_index_in_w
            else:
                max_possible_r = intervals[i + 1][0] - 1
            intervals[i][1] = min(max_possible_r, intervals[i][1] + right_extension)

    # Left extension
    if left_extension != 0:
        for i in range(n_intervals):
            if i == 0:
                min_possible_l = 0
            else:
                min_possible_l = intervals[i - 1][1] + 1
            intervals[i][0] = max(min_possible_l, intervals[i][0] - left_extension)


def smooth_lowess(y, x, frac=1.0/2.0):
    """
    Smoothing using lowess
    return y expectation at input x positions
    """
    n = len(x)
    neighborhood = np.ceil(frac*n)

    # w_ij = (1-abs((x_j-x_i)/(neighborhood))**3)**3 if x_j IN the neighborhood of x_i
    # w_ij = 0 if x_j in NOT IN the neighborhood of x_i
    # Of corse w_ij is symmetric
    w = np.clip(np.abs((x[:, None]-x[None, :])/neighborhood), 0.0, 1.0)
    w = (1-w**3)**3

    delta = np.ones(n)
    # We use matrices to speed up calculation but memory costly
    xmatrix = np.tile(x, n).reshape(n, n)  # x_*j
    ymatrix = np.tile(y, n).reshape(n, n)  # y_*j

    wmatrix = w * np.tile(delta, n).reshape(n, n)   # w_ij * delta_*j

    # Here capital case X Y has nothing to do with lower case x y,

    # Derived from linear regression in order to solve 'm' and 'k'
    # Ya_i=m_i+k_i*Xa_i and Yb_i=m_i+k_i*Xb_i
    Ya = np.sum(wmatrix * ymatrix, axis=1) / np.sum(wmatrix, axis=1)
    Xa = np.sum(wmatrix * xmatrix, axis=1) / np.sum(wmatrix, axis=1)
    Yb = np.sum(wmatrix * xmatrix * ymatrix, axis=1) / np.sum(wmatrix * xmatrix, axis=1)
    Xb = np.sum(wmatrix * xmatrix * xmatrix, axis=1) / np.sum(wmatrix * xmatrix, axis=1)

    yestimate = Ya + (Yb - Ya) / (Xb - Xa) * (x - Xa)
    residuals = y - yestimate
    s = np.median(np.abs(residuals))
    delta = np.clip(residuals / (6.0 * s), -1, 1)
    delta = (1 - delta ** 2) ** 2

    return yestimate


def find_intervals_above_threshold(w, threshold, result_buffer):
    """Fills result_buffer with l, r bounds of intervals in w > threshold.
    :param w: Waveform to do hitfinding in
    :param threshold: Threshold for including an interval
    :param result_buffer: numpy N*2 array of ints, will be filled by function.
                          if more than N intervals are found, none past the first N will be processed.
    :returns : number of intervals processed
    Boundary indices are inclusive, i.e. the right boundary is the last index which was > threshold
    """
    result_buffer_size = len(result_buffer)
    last_index_in_w = len(w) - 1

    in_interval = False
    current_interval = 0
    current_interval_start = -1

    for i, x in enumerate(w):

        if not in_interval and x > threshold:
            # Start of an interval
            in_interval = True
            current_interval_start = i

        if in_interval and (x <= threshold or i == last_index_in_w):
            # End of the current interval
            in_interval = False

            # The interval ended just before this index
            # Unless we ended ONLY because this is the last index, then the interval ends right here
            itv_end = i - 1 if x <= threshold else i

            # Split interval if the interval >350 samples
            # Use lowess to smooth raw waveform and split at relative minima.
            if itv_end - current_interval_start > 350:
                broader_interval_start = current_interval_start

                _w = w[broader_interval_start:itv_end].copy()
                conv = np.ones(100)/100
                _w = np.convolve(_w, conv, 'same')
                dw = _w[1:] - _w[:-1]
                for j in np.where((np.hstack((dw, -1)) > 0) & (np.hstack((1, dw)) <= 0))[0]:
                    result_buffer[current_interval, 0] = current_interval_start
                    result_buffer[current_interval, 1] = j+broader_interval_start

                    current_interval_start = j+broader_interval_start+1
                    current_interval += 1

                    if current_interval == result_buffer_size:
                        return current_interval

            # Add bounds to result buffer
            result_buffer[current_interval, 0] = current_interval_start
            result_buffer[current_interval, 1] = itv_end
            current_interval += 1

            if current_interval == result_buffer_size:
                return current_interval

    n_intervals = current_interval      # No +1, as current_interval was incremented also when the last interval closed
    return n_intervals


@numba.jit(numba.int32(numba.float64[:], numba.float64, numba.int64[:, :]),
           nopython=True)
def find_intervals_above_threshold_no_splitting(w, threshold, result_buffer):
    """Fills result_buffer with l, r bounds of intervals in w > threshold.
    Unlike find_intervals_above_threshold(), does not smooth and split hits,
    which allows speed increase in ZLE simulation.
    :param w: Waveform to do hitfinding in
    :param threshold: Threshold for including an interval
    :param result_buffer: numpy N*2 array of ints, will be filled by function.
                          if more than N intervals are found, none past the first N will be processed.
    :returns : number of intervals processed
    Boundary indices are inclusive, i.e. the right boundary is the last index which was > threshold
    """
    result_buffer_size = len(result_buffer)
    last_index_in_w = len(w) - 1

    in_interval = False
    current_interval = 0
    current_interval_start = -1

    for i, x in enumerate(w):

        if not in_interval and x > threshold:
            # Start of an interval
            in_interval = True
            current_interval_start = i

        if in_interval and (x <= threshold or i == last_index_in_w):
            # End of the current interval
            in_interval = False

            # The interval ended just before this index
            # Unless we ended ONLY because this is the last index, then the interval ends right here
            itv_end = i - 1 if x <= threshold else i

            # Add bounds to result buffer
            result_buffer[current_interval, 0] = current_interval_start
            result_buffer[current_interval, 1] = itv_end
            current_interval += 1

            if current_interval == result_buffer_size:
                break

    n_intervals = current_interval      # No +1, as current_interval was incremented also when the last interval closed
    return n_intervals

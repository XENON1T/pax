"""
Utilities for peakfinders etc.
Heavily used in SimpleDSP

"""

import math
import numpy as np
from itertools import chain


def intervals_above_threshold(signal, threshold):
    """Return boundary indices of all intervals in signal (strictly) above threshold"""
    cross_above, cross_below = sign_changes(signal - threshold)
    # Assuming each interval's left <= right, we can simply split sorted(lefts+rights) in pairs
    # Todo: come on, there must be a numpy method for this!
    return list(zip(*[iter(sorted(cross_above + cross_below))] * 2))

# From Xerawdp
derivative_kernel = [-0.003059, -0.035187, -0.118739, -0.143928, 0.000000, 0.143928, 0.118739, 0.035187, 0.003059]
assert len(derivative_kernel) % 2 == 1

def peaks_and_valleys(signal, test_function):
    """Find peaks and valleys based on derivative sign changes
    :param signal: signal to search in
    :param test_function: Function which accepts three args:
            - signal, signal begin tested
            - peak, index of peak
            - valley, index of valley
        must return True if peak/valley pair is acceptable, else False
    :return: two sorted lists: peaks, valleys
    """

    if len(signal) < len(derivative_kernel):
        # Signal is too small, can't calculate derivatives
        return [],[]
    slope = np.convolve(signal, derivative_kernel, mode='same')
    # Chop the invalid parts off - easier than mode='valid' and adding offset to results
    offset = (len(derivative_kernel)-1)/2
    slope[0:offset] = np.zeros(offset)
    slope[len(slope)-offset:] = np.zeros(offset)
    peaks, valleys = sign_changes(slope, report_first_index='never')
    peaks = np.array(sorted(peaks))
    valleys = np.array(sorted(valleys))
    assert len(peaks) == len(valleys)
    # Remove coinciding peak&valleys
    good_indices = np.where(peaks != valleys)[0]
    peaks = np.array(peaks[good_indices])
    valleys = np.array(valleys[good_indices])
    if not all(valleys > peaks):   # Valleys are AFTER the peaks
        print(valleys - peaks)
        raise RuntimeError("Peak & valley list weird!")

    if len(peaks) < 2:
        return peaks, valleys

    # Remove peaks and valleys which are too close to each other, or have too low a p/v ratio
    # This can't be a for-loop, as we are modifying the lists, and step back to recheck peaks.
    now_at_peak = 0
    while 1:

        # Find the next peak, if there is one
        if now_at_peak > len(peaks)-1:
            break
        peak = peaks[now_at_peak]
        if math.isnan(peak):
            now_at_peak += 1
            continue

        # Check the valleys around this peak
        if peak < min(valleys):
            fail_left = False
        else:
            valley_left = np.max(valleys[np.where(valleys < peak)[0]])
            fail_left = not test_function(signal, peak, valley_left)
        valley_right = np.min(valleys[np.where(valleys > peak)[0]])
        fail_right = not test_function(signal, peak, valley_right)
        if not (fail_left or fail_right):
            # We're good, move along
            now_at_peak += 1
            continue

        # Some check failed: we must remove a peak/valley pair.
        # Which valley should we remove?
        if fail_left and fail_right:
            #Both valleys are bad! Remove the most shallow valley.
            valley_to_remove = valley_left if signal[valley_left] > signal[valley_right] else valley_right
        elif fail_left:
            valley_to_remove = valley_left
        elif fail_right:
            valley_to_remove = valley_right

        # Remove the shallowest peak near the valley marked for removal
        left_peak  = max(peaks[np.where(peaks < valley_to_remove)[0]])
        if valley_to_remove > max(peaks):
            # There is no right peak, so remove the left peak
            peaks = peaks[np.where(peaks != left_peak)[0]]
        else:
            right_peak = min(peaks[np.where(peaks > valley_to_remove)[0]])
            if signal[left_peak] < signal[right_peak]:
                peaks = peaks[np.where(peaks != left_peak)[0]]
            else:
                peaks = peaks[np.where(peaks != right_peak)[0]]

        # Jump back a few peaks to be sure we repeat all checks,
        # even if we just removed a peak before the current peak
        now_at_peak = max(0, now_at_peak-1)
        valleys = valleys[np.where(valleys != valley_to_remove)[0]]

    peaks, valleys = [p for p in peaks if not math.isnan(p)], [v for v in valleys if not math.isnan(v)]
    # Return all remaining peaks & valleys
    return np.array(peaks), np.array(valleys)


def sign_changes(signal, report_first_index='positive'):
    """Return indices at which signal changes sign.
    Returns two sorted numpy arrays:
        - indices at which signal becomes positive (changes from  <=0 to >0)
        - indices at which signal becomes non-positive (changes from >0 to <=0)
    Arguments:
        - signal
        - report_first_index:    if 'positive', index 0 is reported only if it is positive (default)
                                 if 'non-positive', index 0 is reported if it is non-positive
                                 if 'never', index 0 is NEVER reported.
    """
    above0 = np.clip(np.sign(signal), 0, float('inf'))
    if report_first_index == 'positive':
        above0[-1] = 0
    elif report_first_index == 'non-positive':
        above0[-1] = 1
    else:      # report_first_index ==  'never':
        above0[-1] = -1234
    above0_next = np.roll(above0, 1)
    becomes_positive     = np.sort(np.where(above0 - above0_next == 1)[0])
    becomes_non_positive = np.sort(np.where(above0 - above0_next == -1)[0] - 1)
    return list(becomes_positive), list(becomes_non_positive)


def merge_overlapping_peaks(peaks):
    """ Merge overlapping peaks - highest peak consumes lower peak """
    for p in peaks:
        if p.type == 'consumed': continue
        for q in peaks:
            if p == q: continue
            if q.type == 'consumed': continue
            if q.left <= p.index_of_maximum <= q.right:
                if q.height > p.height:
                    consumed, consumer = p, q
                else:
                    consumed, consumer = q,p
                consumed.type = 'consumed'
    return [p for p in peaks if p.type != 'consumed']


def peak_bounds(signal, max_idx, fraction_of_max, zero_level=0):
    """
    Return (left, right) bounds of the fraction_of_max width of the peak in samples.
    TODO: add interpolation option

    :param signal: waveform to look in (numpy array)
    :param peak: Peak object
    :param fraction_of_max: Width at this fraction of maximum
    :param zero_level: Always end a peak before it is < this. Default: 0
    """
    if len(signal) == 0:
        raise RuntimeError("Empty signal, can't find peak bounds!")
    height = signal[max_idx]
    threshold = min(zero_level, height * fraction_of_max)
    threshold_test = np.vectorize(lambda x: x < threshold)
    if height < threshold:
        # Peak is always below threshold -> return smallest legal peak.
        return (max_idx, max_idx)
    # First find # of indices we need to move from max, so we can test if it is None
    # if max_idx == 0:
    #     left = 0
    # else:
    left = find_first_fast(signal[max_idx::-1], threshold_test)     # Note reversion acts before indexing!
    # if max_idx == len(signal)-1:
    #     right = len(signal)-1
    # else:
    right = find_first_fast(signal[max_idx:], threshold_test)
    if left is None: left = 0
    if right is None: right = len(signal)-1
    # Convert to indices in waveform
    right += max_idx
    left =  max_idx - left
    return (left, right)


# Stolen from https://github.com/numpy/numpy/issues/2269
# Numpy 2.0 may get a builtin to do this
# TODO: predicate = np.vectorize(predicate)?? Or earlier?
def find_first_fast(a, predicate, chunk_size=128):
    i0 = 0
    chunk_inds = chain(range(chunk_size, a.size, chunk_size),[None])
    for i1 in chunk_inds:
        chunk = a[i0:i1]
        for inds in zip(*predicate(chunk).nonzero()):
            return inds[0] + i0
        i0 = i1
    #HACK: None found... return the last index
    return len(a)-1
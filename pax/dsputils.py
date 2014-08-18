"""
Utilities for peakfinders etc.
Heavily used in SimpleDSP

"""


import numpy as np
from itertools import chain
from pax import datastructure


def remove_overlapping_peaks(peaks):
    """ Remove overlapping peaks - highest peak consumes lower peak """
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

def find_peaks_in_intervals(signal, candidate_intervals, peak_type):
    peaks = []
    for itv_left, itv_right in candidate_intervals:
        max_idx = itv_left + np.argmax(signal[itv_left:itv_right + 1])
        peaks.append(datastructure.Peak({
                'index_of_maximum': max_idx,
                'height':           signal[max_idx],
                'type':             peak_type
        }))
    return peaks


def intervals_above_threshold(signal, threshold):
    """Return boundary indices of all intervals in signal (strictly) above threshold"""
    above0 = np.clip(np.sign(signal - threshold), 0, float('inf'))
    above0[-1] = 0      # Last sample is always an end. Also prevents edge cases due to rolling it over.
    above0_next = np.roll(above0, 1)
    cross_above = np.sort(np.where(above0 - above0_next == 1)[0])
    cross_below = np.sort(np.where(above0 - above0_next == -1)[0] - 1)
    # Assuming each interval's left <= right, we can simply split sorted(lefts+rights) in pairs
    # Todo: come on, there must be a numpy method for this!
    return list(zip(*[iter(sorted(list(cross_above) + list(cross_below)))] * 2))


def peak_bounds(signal, peak, fraction_of_max, zero_level=0):
    """
    Return (left, right) bounds of the fraction_of_max width of the peak in samples.
    TODO: add interpolation option

    :param signal: waveform to look in (numpy array)
    :param peak: Peak object
    :param fraction_of_max: Width at this fraction of maximum
    :param zero_level: Always end a peak before it is < this. Default: 0
    """
    threshold = min(zero_level, peak.height * fraction_of_max)
    threshold_test = np.vectorize(lambda x: x < threshold)
    max_idx = peak.index_of_maximum
    if peak.height < threshold:
        # Peak is always below threshold -> return smallest legal peak.
        return (max_idx, max_idx)
    # First find # of indices we need to move from max, so we can test if it is None
    right = find_first_fast(signal[max_idx:], threshold_test)
    left = find_first_fast(signal[max_idx::-1], threshold_test)     # Note reversion acts before indexing!
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
    """
    Find the indices of array elements that match the predicate.

    Parameters
    ----------
    a : array_like
        Input data, must be 1D.

    predicate : function
        A function which operates on sections of the given array, returning
        element-wise True or False for each data value.

    chunk_size : integer
        The length of the chunks to use when searching for matching indices.
        For high probability predicates, a smaller number will make this
        function quicker, similarly choose a larger number for low
        probabilities.

    Returns
    -------
    index_generator : generator
        A generator of (indices, data value) tuples which make the predicate
        True.

    See Also
    --------
    where, nonzero

    Notes
    -----
    This function is best used for finding the first, or first few, data values
    which match the predicate.

    Examples
    --------
    >>> a = np.sin(np.linspace(0, np.pi, 200))
    >>> result = find(a, lambda arr: arr > 0.9)
    >>> next(result)
    ((71, ), 0.900479032457)
    >>> np.where(a > 0.9)[0][0]
    71


    """
    if a.ndim != 1:
        raise ValueError('The array must be 1D, not {}.'.format(a.ndim))

    i0 = 0
    chunk_inds = chain(range(chunk_size, a.size, chunk_size),
                 [None])

    for i1 in chunk_inds:
        chunk = a[i0:i1]
        for inds in zip(*predicate(chunk).nonzero()):
            return inds[0] + i0
        i0 = i1
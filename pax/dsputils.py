"""
Utilities for peakfinders etc.
Heavily used in SimpleDSP

"""

import math
import re
import json
import gzip
from itertools import chain

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

import logging
log = logging.getLogger('dsputils')

from pax import datastructure, units


##
# Peak finding routines
##

def intervals_above_threshold(signal, threshold):
    """Return boundary indices of all intervals in signal (strictly) above threshold"""
    cross_above, cross_below = sign_changes(signal - threshold)
    # Assuming each interval's left <= right, we can simply split sorted(lefts+rights) in pairs
    # Todo: come on, there must be a numpy method for this!
    return list(zip(*[iter(sorted(cross_above + cross_below))] * 2))

# From Xerawdp
derivative_kernel = [-0.003059, -0.035187, -0.118739, -0.143928, 0.000000, 0.143928, 0.118739, 0.035187, 0.003059]
assert len(derivative_kernel) % 2 == 1


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
    becomes_positive = np.sort(np.where(above0 - above0_next == 1)[0])
    becomes_non_positive = np.sort(np.where(above0 - above0_next == -1)[0] - 1)
    return list(becomes_positive), list(becomes_non_positive)


def find_peak_in_signal(signal, unfiltered, integration_bound_fraction, offset=0):
    """Finds 'the' peak in the candidate interval
    :param signal: Signal to use for peak finding & extent computation
    :param unfiltered: Unfiltered waveform (used for max, height, area computation)
    :param integration_bound_fraction: Fraction of max where you choose the peak to end.
    :param offset: index in the waveform of the first index of the signal passed. Default 0.
    :return: a pax datastructure Peak
    """
    # Find the peak's maximum and extent using 'signal'
    max_idx = np.argmax(signal)
    left, right = peak_bounds(signal, max_idx, integration_bound_fraction)
    # Compute properties of this peak using 'unfiltered'
    area = np.sum(unfiltered[left:right + 1])
    unfiltered_max = left + np.argmax(unfiltered[left:right + 1])
    return datastructure.Peak({
        'index_of_maximum': offset + unfiltered_max,
        'height':           unfiltered[unfiltered_max],
        'left':             offset + left,
        'right':            offset + right,
        'area':             area,
        # TODO: FWHM etc. On unfiltered wv? both?
    })


def peak_bounds(signal, max_idx, fraction_of_max, zero_level=0):
    """
    Return (left, right) indices closest to max_idx where signal drops below signal[max_idx]*fraction_of_max.

    :param signal: waveform to look in (numpy array)
    :param peak: Peak object
    :param fraction_of_max: Width at this fraction of maximum
    :param zero_level: Always end a peak before it is < this. Default: 0
    """
    if len(signal) == 0:
        raise RuntimeError("Empty signal, can't find peak bounds!")
    if max_idx > len(signal)-1:
        raise RuntimeError("Can't compute bounds: max at %s, peak wave is %s long" % (max_idx, len(signal)))
    if max_idx < 0:
        raise RuntimeError("Peak maximum index is negative (%s)... what are you smoking?" % max_idx )

    height = signal[max_idx]
    threshold = max(zero_level, height * fraction_of_max)

    if height < threshold:
        # Peak is always below threshold -> return smallest legal peak.
        return (max_idx, max_idx)

    # Note reversion acts before indexing!
    bla = np.where(signal[max_idx::-1] < threshold)
    left = find_first_fast(signal[max_idx::-1], threshold)
    if left is None:
        left = 0
    else:
        left = max_idx - left
    right = find_first_fast(signal[max_idx:], threshold)
    if right is None:
        right = len(signal) - 1
    else:
        right += max_idx

    return (left, right)


def width_at_fraction(peak_wave, fraction_of_max, max_idx, interpolate=False):
    """Returns width of a peak IN SAMPLES at fraction of maximum"""
    left, right = peak_bounds(peak_wave, max_idx=max_idx, fraction_of_max=fraction_of_max)
    # Try to do sub-sample width determination

    threshold = peak_wave[max_idx]*fraction_of_max

    if interpolate:     # Need at least 3 points to interpolate
        if left+1 in peak_wave and peak_wave[left] < threshold < peak_wave[left+1]:
            correction = (peak_wave[left] - threshold) / (peak_wave[left] - peak_wave[left+1])
            assert 0 <= correction <= 1
            left += correction
        else:
            # Weird peak, can't interpolate
            # Should not happen once peakfinder works well
            pass

        if right-1 in peak_wave and peak_wave[right] < threshold < peak_wave[right-1]:
            correction = (threshold - peak_wave[right]) / (peak_wave[right-1] - peak_wave[right])
            assert 0 <= correction <= 1
            right -= correction
        else:
            # Weird peak, can't interpolate
            # Should not happen once peakfinder works well
            pass

    # If you want to test this code, uncomment this:
    # plt.plot(peak_wave, 'bo-')
    # plt.plot([threshold for i in range(len(peak_wave))], '--', label='Threshold')
    # plt.plot([left, left], [0, threshold], '--', label='Left width bound')
    # plt.plot([right, right], [0, threshold], '--', label='Right width bound')
    # plt.scatter([max_idx], [peak_wave[max_idx]], marker='*', s=100, label='Maximum', color='purple')
    # plt.title('Peak bounds at %s of max' % fraction_of_max)
    # plt.legend()
    # plt.show()

    return right - left + 1




def find_first_fast(a, threshold, chunk_size=128):
    """Returns the first index in a below threshold.
    If a never goes below threshold, returns the last index in a."""
    # Numpy 2.0 may get a builtin to do this.
    # I don't know of anything beter than the below for now:
    indices = np.where(a < threshold)[0]
    if len(indices) > 0:
        return indices[0]
    else:
        #None found, return last index
        return len(a)-1
    # This was recommended by https://github.com/numpy/numpy/issues/2269
    # It actually performs significantly worse in our case...
    # Maybe I'm messing something up?
    # threshold_test = np.vectorize(lambda x: x < threshold)
    # i0 = 0
    # chunk_inds = chain(range(chunk_size, a.size, chunk_size), [None])
    # for i1 in chunk_inds:
    #     chunk = a[i0:i1]
    #     for inds in zip(*threshold_test(chunk).nonzero()):
    #         return inds[0] + i0
    #     i0 = i1
    # # HACK: None found... return the last index
    # return len(a) - 1



##
# Peak processing routines
##

def free_regions(event):
    """Find the free regions in the event's waveform - regions where peaks haven't yet been found"""
    lefts = sorted([0] + [p.left for p in event.peaks])
    rights = sorted([p.right for p in event.peaks] + [event.length() - 1])
    # Assuming each peak's right > left, we can simply split
    # sorted(lefts+rights) in pairs:
    return list(zip(*[iter(sorted(lefts + rights))] * 2))


# TODO: maybe move this to the peak splitter? It isn't used by anything
# else... yet


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
        return [], []
    slope = np.convolve(signal, derivative_kernel, mode='same')
    # Chop the invalid parts off - easier than mode='valid' and adding offset
    # to results
    offset = (len(derivative_kernel) - 1) / 2
    slope[0:offset] = np.zeros(offset)
    slope[len(slope) - offset:] = np.zeros(offset)
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
    # This can't be a for-loop, as we are modifying the lists, and step back
    # to recheck peaks.
    now_at_peak = 0
    while 1:

        # Find the next peak, if there is one
        if now_at_peak > len(peaks) - 1:
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
            # Both valleys are bad! Remove the most shallow valley.
            valley_to_remove = valley_left if signal[
                valley_left] > signal[valley_right] else valley_right
        elif fail_left:
            valley_to_remove = valley_left
        elif fail_right:
            valley_to_remove = valley_right

        # Remove the shallowest peak near the valley marked for removal
        left_peak = max(peaks[np.where(peaks < valley_to_remove)[0]])
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
        now_at_peak = max(0, now_at_peak - 1)
        valleys = valleys[np.where(valleys != valley_to_remove)[0]]

    peaks, valleys = [p for p in peaks if not math.isnan(
        p)], [v for v in valleys if not math.isnan(v)]
    # Return all remaining peaks & valleys
    return np.array(peaks), np.array(valleys)


##
# Correction map class
##
class InterpolatingMap(object):
    """
    Builds a scalar function of space using interpolation from sampling points on a regular grid.

    All interpolation is done linearly.
    Cartesian coordinates are supported and tested, cylindrical coordinates (z, r, phi) may also work...

    The map must be specified as a json containing a dictionary with keys
        'coordinate_system' :   [['x', x_min, x_max, n_x], ['y',...
        'your_map_name' :       [[valuex1y1, valuex1y2, ..], [valuex2y1, valuex2y2, ..], ...
        'another_map_name' :    idem
        'name':                 'Nice file with maps',
        'description':          'Say what the maps are and who you are',
        'timestamp':            unix timestamp
    with the straightforward generalization to 1d and 3d.

    See also examples/generate_mock_correction_map.py
    """
    data_field_names = ['timestamp', 'description', 'coordinate_system', 'name']

    def __init__(self, filename):
        self.log = logging.getLogger('InterpolatingMap')
        self.log.debug('Loading JSON map %s' % filename)

        if filename[-3:] == '.gz':
            bla = gzip.open(filename).read()
            self.data = json.loads(bla.decode())
        else:
            self.data = json.load(open(filename))
        self.coordinate_system = cs = self.data['coordinate_system']
        self.dimensions = len(cs)
        self.interpolators = {}
        self.map_names = sorted([k for k in self.data.keys() if k not in self.data_field_names])
        self.log.debug('Map name: %s' % self.data['name'])
        self.log.debug('Map description:\n    ' + re.sub(r'\n', r'\n    ', self.data['description']))
        self.log.debug("Map names found: %s" % self.map_names)

        for map_name in self.map_names:

            # 1 D interpolation
            if self.dimensions == 1:
                itp_fun = interpolate.interp1d(x = np.linspace(*(cs[0][1])),
                                                         y = self.data[map_name])

            # 2D interpolation
            elif self.dimensions == 2:
                itp_fun = interpolate.interp2d(x = np.linspace(*(cs[0][1])),
                                                         y = np.linspace(*(cs[1][1])),
                                                         z = self.data[map_name])

            # 3D interpolation
            elif self.dimensions == 3:
                # LinearNDInterpolator wants points as [(x1,y1,z1), (x2, y2, z2), ...]
                all_x, all_y, all_z = np.meshgrid(np.linspace(*(cs[0][1])),
                                                  np.linspace(*(cs[1][1])),
                                                  np.linspace(*(cs[2][1])))
                points = np.array([np.ravel(all_x), np.ravel(all_y), np.ravel(all_z)]).T
                values = np.ravel(self.data[map_name])
                itp_fun = interpolate.LinearNDInterpolator(points, values)

            else:
                raise RuntimeError("Can't use a %s-dimensional correction map!" % self.dimensions)

            self.interpolators[map_name] = itp_fun


    def get_value_at(self, position, map_name='map'):
        """Returns the value of the map map_name at a ReconstructedPosition
         position - pax.datastructure.ReconstructedPosition instance
        """
        return self.get_value(*[getattr(position, q[0]) for q in self.coordinate_system], map_name=map_name)
        
    def get_value(self, *coordinates, map_name='map'):
        """Returns the value of the map at the position given by coordinates"""
        result = self.interpolators[map_name](*coordinates)
        try:
            return float(result[0])
        except TypeError:
            return float(result)    # We don't want a 0d numpy array, which the 1d and 2d interpolators seem to give

    def plot(self, map_name='map', to_file=None):
        """Make a quick plot of the map map_name, for diagnostic purposes only"""
        cs = self.coordinate_system

        if self.dimensions == 2:
            x = np.linspace(*cs[0][1])
            y = np.linspace(*cs[1][1])
            plt.pcolor(x, y, np.array(self.data[map_name]))
            plt.xlabel("%s (cm)" % cs[0][0])
            plt.ylabel("%s (cm)" % cs[1][0])
            plt.axis([x.min(), x.max(), y.min(), y.max()])
            plt.colorbar()
            # Plot the TPC radius for reference
            # TODO: this hardcodes a XENON100 geometry value!
            # But I don't have the config here...
            # if cs[0][0] == 'x' and cs[1][0] == 'y':

        else:
            raise NotImplementedError("Still have to implement plotting for %s-dimensional maps" % self.dimensions)

        plt.title(map_name)
        if to_file is not None:
            plt.savefig(to_file)
        else:
            plt.show()
        plt.close()

# def rcosfilter(filter_length, rolloff, cutoff_freq, sampling_freq=1):
#     """
#     Returns a nd(float)-array describing a raised cosine (RC) filter (FIR) impulse response. Arguments:
#         - filter_length:    filter event_duration in samples
#         - rolloff:          roll-off factor
#         - cutoff_freq:      cutoff frequency = 1/(2*symbol period)
#         - sampling_freq:    sampling rate (in same units as cutoff_freq)
#     """
#     symbol_period = 1 / (2 * cutoff_freq)
#     h_rc = np.zeros(filter_length, dtype=float)
#
#     for x in np.arange(filter_length):
#         t = (x - filter_length / 2) / float(sampling_freq)
#         phase = np.pi * t / symbol_period
#         if t == 0.0:
#             h_rc[x] = 1.0
#         elif rolloff != 0 and abs(t) == symbol_period / (2 * rolloff):
#             h_rc[x] = (np.pi / 4) * (np.sin(phase) / phase)
#         else:
#             h_rc[x] = (np.sin(phase) / phase) * (
#                 np.cos(phase * rolloff) / (
#                     1 - (((2 * rolloff * t) / symbol_period) * ((2 * rolloff * t) / symbol_period))
#                 )
#             )
#
#     return h_rc / h_rc.sum()

# def merge_overlapping_peaks(peaks):
#     """ Merge overlapping peaks - highest peak consumes lower peak """
#     for p in peaks:
#         if p.type == 'consumed':
#             continue
#         for q in peaks:
#             if p == q:
#                 continue
#             if q.type == 'consumed':
#                 continue
#             if q.left <= p.index_of_maximum <= q.right:
#                 log.debug('Peak at %s overlaps wit peak at %s' % (p.index_of_maximum, q.index_of_maximum))
#                 if q.height > p.height:
#                     consumed, consumer = p, q
#                 else:
#                     consumed, consumer = q, p
#                 consumed.type = 'consumed'
#     return [p for p in peaks if p.type != 'consumed']
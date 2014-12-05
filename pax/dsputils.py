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
# Peak finding helper routines
##

def intervals_where(x):
    """Given numpy array of bools, return list of (left, right) inclusive bounds of all intervals of True
    """
    # In principle, boundaries are just points of change...
    becomes_true, becomes_false = where_changes(x, report_first_index_if=True)

    # ... except that the right boundaries are 1 index BEFORE the array becomes False ...
    assert 0 not in becomes_false     # Would indicate a bad bug in where_changes
    becomes_false -= 1

    # ... and if the last index is True, we must ensure it is in manually (it may or may not be a point of change).
    # can't say x[-1] is True, it is a numpy bool...
    if x[-1] and len(x)-1 not in becomes_true:
        becomes_true = np.concatenate((becomes_true, np.array([len(x)-1])))

    # Assuming each interval's left <= right, we can split sorted(cross_above+cross_below) into pairs to get our result
    return list(zip(*[iter(sorted(list(becomes_true) + list(becomes_false )))] * 2))

    # I've been looking for a proper numpy solution. It should be:
    # return np.vstack((cross_above, cross_below)).T
    # But it appears to make the processor run slower! (a tiny bit)
    # Maybe because we're dealing with many small arrays rather than big ones?


def where_changes(x, report_first_index_if=None):
    """Return indices where boolean array changes value.
    :param x: ndarray of bools
    :param report_first_index_if: When to report the first index in x.
        If True,  0 is reported (in first returned array)  if it is true.
        If False, 0 is reported (in second returned array) if it is false.
        If None, 0 is never reported. (Default)
    :returns: 2-tuple of integer ndarrays (becomes_true, becomes_false):
        becomes_true:  indices where x is True,  and was False one index before
        becomes_false: indices where x is False, and was True  one index before

    report_first_index_if can be
    """

    # To compare with the previous sample in a quick way, we use np.roll
    previous_x = np.roll(x, 1)
    points_of_difference = (x != previous_x)

    # The last sample, however, has nothing to do with the first sample
    # It can never be a point of difference, so we remove it:
    points_of_difference[0] = False

    # Now we can find where the array becomes True or False
    becomes_true =  np.sort(np.where(points_of_difference & x)[0])
    becomes_false = np.sort(np.where(points_of_difference & (-x))[0])

    # In case the user set report_first_index_if, we have to manually add 0 if it is True or False
    # Can't say x[0] is True, it is a numpy bool...
    if report_first_index_if is True and x[0]:
        becomes_true = np.concatenate((np.array([0]), becomes_true))
    if report_first_index_if is False and not x[0]:
        becomes_false = np.concatenate((np.array([0]), becomes_false))

    return becomes_true, becomes_false


def peak_bounds(signal, fraction_of_max=None, max_idx=None, zero_level=0, inclusive=True):
    """
    Return (left, right) indices closest to max_idx where signal drops below signal[max_idx]*fraction_of_max.

    :param signal: waveform to look in (numpy array)
    :param fraction_of_max: Width at this fraction of maximum. If None, peaks will extend until zero_level.
    :param max_idx: Index in signal of the maximum of the peak. Default None, will be determined by np.argmax(signal)
    :param zero_level: Always end a peak when it drops below this level. Default: 0
    :param inclusive: Include endpoints (first points where signal drops below), default True.
    TODO: proper support for inclusive (don't just subtract 1)
    """
    if max_idx is None:
        max_idx = np.argmax(signal)
    if len(signal) == 0:
        raise RuntimeError("Empty signal, can't find peak bounds!")
    if max_idx > len(signal)-1:
        raise RuntimeError("Can't compute bounds: max at %s, peak wave is %s long" % (max_idx, len(signal)))
    if max_idx < 0:
        raise RuntimeError("Peak maximum index is negative (%s)... what are you smoking?" % max_idx )

    height = signal[max_idx]

    if fraction_of_max is None:
        threshold = zero_level
    else:
        threshold = max(zero_level, height * fraction_of_max)

    if height < threshold:
        # Peak is always below threshold -> return smallest legal peak.
        return (max_idx, max_idx)

    # Note reversion acts before indexing in numpy!
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

    if not inclusive:
        if signal[left] < threshold and left != len(signal) - 1:
            left += 1
        if signal[right] < threshold and right != 0:
            right -= 1
    return (left, right)


#TODO: interpolate argument shadows interpolate imported from scipy
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



def free_regions(event, ignore_peak_types=()):
    """Find the free regions in the event's waveform - regions where peaks haven't yet been found
        ignore_peak_types: list of names of peak.type's which will be ignored for the computation
    :returns list of 2-tuples (left index, right index) of regions where no peaks have been found
    """
    lefts = sorted([0] + [p.left for p in event.peaks if p.type not in ignore_peak_types])
    rights = sorted([p.right for p in event.peaks if p.type not in ignore_peak_types] + [event.length() - 1])
    # Assuming each peak's right > left, we can simply split
    # sorted(lefts+rights) in pairs:
    return list(zip(*[iter(sorted(lefts + rights))] * 2))




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
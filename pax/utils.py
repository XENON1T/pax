"""
Utilities for peakfinders etc.

"""

import re
import json
import gzip
from itertools import zip_longest

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

import logging
log = logging.getLogger('dsputils')

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
    boundaries = list(becomes_true) + list(becomes_false - 1)
    boundaries.sort()

    # ... and if the last index is True, we must add it once more manually:
    #   - if the second to last index is True, it is not a point of change, so must be added
    #   - if the second to last index is False, it is a lone interval, so must be in boundaries twice
    # [True] would cause trouble here, but we've removed it already
    if x[-1]:
        boundaries.append(len(x)-1)

    # If some bug causes a non-even number of boundaries, better catch it here
    if len(boundaries) % 2 != 0:
        print(sorted(boundaries), x, x[-1], len(x)-1)
        raise RuntimeError("Number of boundaries is not even: can't return intervals!")

    # Assuming each interval's left <= right, we split sorted(boundaries) into pairs to get our result
    return chunk_in_ntuples(sorted(boundaries), n=2)

    # I've been looking for a proper numpy solution. It should be:
    # return np.vstack((cross_above, cross_below)).T
    # But it appears to make the processor run slower! (a tiny bit)
    # Maybe because we're dealing with many small arrays rather than big ones?



def chunk_in_ntuples(iterable, n, fillvalue=None):
    """ Chunks an iterable into a list of tuples
    :param iterable: input iterable
    :param n: length of n tuple
    :param fillvalue: if iterable is not divisible by chunk_size, pad last tuple with this value
    :return: list of n-tuples
    Stolen from http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    Modified for python3, and made it return lists
    """
    if not n>0 and int(n) == n:
        raise ValueError("Chunk size should be a positive integer, not %s" % n)
    return list(zip_longest(*[iter(iterable)]*n, fillvalue=fillvalue))
    # Numpy solution -- without filling though
    # return np.reshape(iterable, (-1,n))


def where_changes(x, report_first_index_if=None):
    """Return indices where boolean array changes value.
    :param x: ndarray or list of bools
    :param report_first_index_if: When to report the first index in x.
        If True,  0 is reported (in first returned array)  if it is true.
        If False, 0 is reported (in second returned array) if it is false.
        If None, 0 is never reported. (Default)
    :returns: 2-tuple of integer ndarrays (becomes_true, becomes_false):
        becomes_true:  indices where x is True,  and was False one index before
        becomes_false: indices where x is False, and was True  one index before

    report_first_index_if can be
    """
    x = np.array(x)

    # To compare with the previous sample in a quick way, we use np.roll
    previous_x = np.roll(x, 1)
    points_of_difference = (x != previous_x)

    # The last sample, however, has nothing to do with the first sample
    # It can never be a point of difference, so we remove it:
    points_of_difference[0] = False

    # Now we can find where the array becomes True or False
    becomes_true =  np.sort(np.where(points_of_difference & x)[0])
    becomes_false = np.sort(np.where(points_of_difference & (True ^ x))[0])

    # In case the user set report_first_index_if, we have to manually add 0 if it is True or False
    # Can't say x[0] is True, it is a numpy bool...
    if report_first_index_if is True and x[0]:
        becomes_true = np.concatenate((np.array([0]), becomes_true))
    if report_first_index_if is False and not x[0]:
        becomes_false = np.concatenate((np.array([0]), becomes_false))

    return becomes_true, becomes_false


##
# Peak processing helper routines
##

def free_regions(event, detector='tpc'):
    """Find the free regions in the event's waveform - regions where peaks haven't yet been found
        detector: give free regions wrt this detector
    :returns list of 2-tuples (left index, right index) of regions where no peaks have been found
    """
    lefts = [0] + [p.left for p in event.peaks if p.detector is detector]
    rights = [p.right for p in event.peaks if p.detector is detector] + [event.length() - 1]
    # Assuming each peak's right > left, we can simply split sorted(lefts+rights) in pairs:
    return chunk_in_ntuples(sorted(lefts + rights), n=2)




def cluster_by_diff(x, diff_threshold, return_indices=False):
    """Returns list of lists of indices of clusters in x,
    making cluster boundaries whenever values are >= threshold apart.
    """
    x = sorted(x)
    if len(x) == 0:
        return []
    clusters = []
    current_cluster = []
    previous_t = x[0]
    for i, t in enumerate(x):
        if t - previous_t > diff_threshold:
            clusters.append(current_cluster)
            current_cluster = []
        current_cluster.append(i if return_indices else t)
        previous_t = t
    clusters.append(current_cluster)
    return clusters
    # Numpy solution below appears to make processor run slower!
    # x.sort()
    # if not isinstance(x, np.ndarray):
    #     x = np.array(x)
    # split_indices = np.where(np.diff(x) >= diff_threshold)[0] + 1
    # if return_indices:
    #     return np.split(np.arange(len(x)), split_indices)
    # else:
    #     return np.split(x, split_indices)



def mad(data, axis=None):
    """ Return median absolute deviation of numpy array"""
    return np.mean(np.absolute(data - np.median(data, axis)), axis)


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
    # I don't know of anything better than the below for now:
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
# Interpolating map class
##

class InterpolatingMap(object):
    """
    Builds a scalar function of space using interpolation from sampling points on a regular grid.

    All interpolation is done linearly.
    Cartesian coordinates are supported, cylindrical coordinates (z, r, phi) may also work...

    The map must be specified as a json containing a dictionary with keys
        'coordinate_system' :   [['x', x_min, x_max, n_x], ['y',...
        'your_map_name' :       [[valuex1y1, valuex1y2, ..], [valuex2y1, valuex2y2, ..], ...
        'another_map_name' :    idem
        'name':                 'Nice file with maps',
        'description':          'Say what the maps are, who you are, your favorite food, etc',
        'timestamp':            unix epoch seconds timestamp
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


# useful for testing
from pax.datastructure import Event
from pax import units
def empty_event():
    return Event(config={'n_pmts': 1, 'digitizer_t_resolution': 10 * units.ns}, start_time=0, length=1)

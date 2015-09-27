"""Helper routines needed in pax

Please only put stuff here that you *really* can't find any other place for!
e.g. a list clustering routine that isn't in some standard, library but several plugins depend on it
"""

import re
import inspect
import json
import gzip
import logging
import time
import os
import glob

import numpy as np
from scipy import interpolate

from pax import units

log = logging.getLogger('pax_utils')


##
# Utilities for finding files inside pax.
##

# Store the directory of pax (i.e. this file's directory) as PAX_DIR
PAX_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def data_file_name(filename):
    """Returns filename if a file exists there, else returns PAX_DIR/data/filename"""
    if os.path.exists(filename):
        return filename
    new_filename = os.path.join(PAX_DIR, 'data', filename)
    if os.path.exists(new_filename):
        return new_filename
    else:
        raise ValueError('File name or path %s not found!' % filename)


def get_named_configuration_options():
    """ Return the names of all working named configurations
    """
    config_files = []
    for filename in glob.glob(os.path.join(PAX_DIR, 'config', '*.ini')):
        filename = os.path.basename(filename)
        m = re.match(r'(\w+)\.ini', filename)
        if m is None:
            print("Weird file in config dir: %s" % filename)
        filename = m.group(1)
        # Config files starting with '_' won't appear in the usage list (they won't work by themselves)
        if filename[0] == '_':
            continue
        config_files.append(filename)
    return config_files


##
# Interpolating map class
##

class InterpolateAndExtrapolate(object):
    """Linearly interpolate, but use nearest-neighbour when out of range
    Initialize and call just like scipy.interpolate.LinearNDInterpolator
    """

    def __init__(self, points, values):
        self.interpolator = interpolate.LinearNDInterpolator(points, values)
        self.extrapolator = interpolate.NearestNDInterpolator(points, values)

    def __call__(self, *args):
        result = self.interpolator(*args)
        if np.isnan(result):
            result = self.extrapolator(*args)
        return result


class InterpolatingMap(object):

    """Construct s a scalar function using linear interpolation, weighted by euclidean distance.

    The map must be specified as a json translating to a dictionary like this:
        'coordinate_system' :   [[x1, y2], [x2, y2], [x3, y3], [x4, y4], ...],
        'map' :                 [[valuex1y1, valuex1y2, ..], [valuex2y1, valuex2y2, ..], ...
        'another_map' :         idem
        'name':                 'Nice file with maps',
        'description':          'Say what the maps are, who you are, your favorite food, etc',
        'timestamp':            unix epoch seconds timestamp
        'use_points':           number of points to use for linear interpolation (with euclidean distance weights)
    with the straightforward generalization to 1d and 3d. The default map name is 'map', I'd recommend you use that.

    For a 0d placeholder map, use
        'points': [],
        'map': 42,
        etc

    The json can be gzip compressed: if so, it must have a .gz extension.

    See also examples/generate_mock_correction_map.py
    """
    data_field_names = ['timestamp', 'description', 'coordinate_system', 'name', 'irregular']

    def __init__(self, filename, **kwargs):
        self.log = logging.getLogger('InterpolatingMap')
        self.log.debug('Loading JSON map %s' % filename)

        if filename.endswith('.gz'):
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
            map_data = np.array(self.data[map_name])
            if self.dimensions == 0:
                # 0 D -- placeholder maps which take no arguments and always return a single value
                itp_fun = lambda: map_data
            else:
                itp_fun = InterpolateAndExtrapolate(points=np.array(cs), values=np.array(map_data))

            self.interpolators[map_name] = itp_fun

    def get_value_at(self, position, map_name='map'):
        """Returns the value of the map map_name at a ReconstructedPosition
         position - pax.datastructure.ReconstructedPosition instance
        """
        position_names = ['x', 'y', 'z']
        return self.get_value(*[getattr(position, q) for q in position_names[:self.dimensions]], map_name=map_name)

    def get_value(self, *coordinates, **kwargs):
        """Returns the value of the map at the position given by coordinates
        Keyword arguments:
          - map_name: Name of the map to use. By default: 'map'.
        """
        map_name = kwargs.get('map_name', 'map')
        result = self.interpolators[map_name](*coordinates)
        try:
            return float(result[0])
        except (TypeError, IndexError):
            return float(result)    # We don't want a 0d numpy array, which the 1d and 2d interpolators seem to give


##
# General helper functions
##

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
    pmt_gain = c['gains'][channel]
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


def gaps_between_hits(hits):
    """Return array of gaps between hits: a hit's 'gap' is the # of samples before that hit free of other hits.
    The gap of the first hit is 0 by definition.
    Hits should already be sorted by left boundary; we'll check this and throw an error if not.
    """
    gaps = np.zeros(len(hits), dtype=np.int32)
    if len(hits) == 0:
        return gaps
    # Keep a running right boundary
    boundary = hits[0].right
    last_left = hits[0].left
    for i, hit in enumerate(hits[1:]):
        gaps[i + 1] = max(0, hit.left - boundary - 1)
        boundary = max(hit.right, boundary)
        if hit.left < last_left:
            raise ValueError("Hits should be sorted by left boundary!")
        last_left = hit.left
    return gaps


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


# Caching decorator
# Stolen from http://avinashv.net/2008/04/python-decorators-syntactic-sugar/
class Memoize:

    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
            return self.memoized[args]


class Timer:
    """Simple stopwatch timer
    punch() returns ms since timer creation or last punch
    """
    last_t = 0

    def __init__(self):
        self.punch()

    def punch(self):
        now = time.time()
        result = (now - self.last_t) * 1000
        self.last_t = now
        return result

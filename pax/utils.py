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
from scipy.ndimage.interpolation import zoom as image_zoom
import matplotlib.pyplot as plt

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

class InterpolatingMap(object):

    """
    Builds a scalar function of space using interpolation from sampling points on a regular grid.

    All interpolation is done linearly.
    Cartesian coordinates are supported, cylindrical coordinates (z, r, phi) may also work...

    The map must be specified as a json translating to a dictionary with keys
        'coordinate_system' :   [['x', x_min, x_max, n_x], ['y',...
        'your_map_name' :       [[valuex1y1, valuex1y2, ..], [valuex2y1, valuex2y2, ..], ...
        'another_map_name' :    idem
        'name':                 'Nice file with maps',
        'description':          'Say what the maps are, who you are, your favorite food, etc',
        'timestamp':            unix epoch seconds timestamp
    with the straightforward generalization to 1d and 3d.

    For a 0d placeholder map, the map value must be a single number, and the coordinate system must be [].

    The json can be gzip compressed: if so, it must have a .gz extension.

    See also examples/generate_mock_correction_map.py
    """
    data_field_names = ['timestamp', 'description', 'coordinate_system', 'name']

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
                itp_fun = self.init_map(map_data, **kwargs)

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
        except (TypeError, IndexError):
            return float(result)    # We don't want a 0d numpy array, which the 1d and 2d interpolators seem to give

    def init_map(self, map_data, **kwargs):
        # 1 D interpolation
        cs = self.coordinate_system
        if self.dimensions == 1:
            # TODO: intper1d is very inefficient for regular grids!!
            return interpolate.interp1d(x=np.linspace(*(cs[0][1])),
                                        y=map_data)

        # 2D interpolation
        elif self.dimensions == 2:
            return interpolate.RectBivariateSpline(x=np.linspace(*(cs[0][1])),
                                                   y=np.linspace(*(cs[1][1])),
                                                   z=np.array(map_data).T,
                                                   s=0)

        # 3D interpolation
        elif self.dimensions == 3:
            # TODO: LinearNDInterpolator is very inefficient for regular grids!!
            # LinearNDInterpolator wants points as [(x1,y1,z1), (x2, y2, z2), ...]
            all_x, all_y, all_z = np.meshgrid(np.linspace(*(cs[0][1])),
                                              np.linspace(*(cs[1][1])),
                                              np.linspace(*(cs[2][1])))
            points = np.array([np.ravel(all_x), np.ravel(all_y), np.ravel(all_z)]).T
            values = np.ravel(map_data)
            return interpolate.LinearNDInterpolator(points, values)

        else:
            raise RuntimeError("Can't use a %s-dimensional correction map!" % self.dimensions)

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


class Vector2DGridMap(InterpolatingMap):

    def init_map(self, map_data, zoom_factor=1, **kwargs):
        x_min, x_max, n_x = self.coordinate_system[0][1]
        y_min, y_max, n_y = self.coordinate_system[0][1]

        if zoom_factor != 1:
            if len(map_data.shape) == 2:
                map_data = self.upsample_map(map_data, zoom_factor)
                n_x, n_y = map_data.shape
            else:
                n_maps = map_data.shape[2]
                new_map_data = np.zeros((zoom_factor * n_x, zoom_factor * n_y, n_maps))
                for map_i in range(n_maps):
                    new_map_data[:, :, map_i] = self.upsample_map(map_data[:, :, map_i], zoom_factor)
                map_data = new_map_data
                n_x, n_y, _ = map_data.shape

        x_spacing = (x_max-x_min)/(n_x-1)
        y_spacing = (y_max-y_min)/(n_y-1)

        def get_data(x, y):
            if np.isnan(x) or np.isnan(y):
                return float('nan') * np.ones_like(map_data[0, 0])
            # Remember we're returning an array now... if you don't copy, you'l get a view...
            # .. then the map will get modified while pax is running... then you will cry...
            return get_data.map_data[int((x - x_min)/x_spacing + 0.5),
                                     int((y - y_min)/y_spacing + 0.5)].copy()
        get_data.map_data = map_data

        return get_data

    def get_value(self, *coordinates, map_name='map'):
        """Returns the value of the map at the position given by coordinates"""
        return self.interpolators[map_name](*coordinates)

    @staticmethod
    def upsample_map(map_data, zoom_factor):
        # Upsample the map using bilinear (order=1) spline interpolation
        return image_zoom(map_data, zoom_factor, order=1)


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
    gaps = np.zeros(len(hits), dtype=np.int16)
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


def weighted_mean_variance(values, weights):
    """
    Return the weighted mean, and the weighted sum square deviation from the weighted mean.
    values, weights -- Numpy ndarrays with the same shape.
    Stolen from http://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    weighted_mean = np.average(values, weights=weights)
    weighted_variance = np.average((values-weighted_mean)**2, weights=weights)  # Fast and numerically precise
    return weighted_mean, weighted_variance


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

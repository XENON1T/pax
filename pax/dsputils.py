import logging
import gzip
import json
import re

import numba
import numpy as np

from pax import units
from pax.datastructure import Hit
from scipy.spatial import KDTree


##
# Interpolating map class
##

class InterpolateAndExtrapolate(object):
    """Linearly interpolate- or extrapolation between nearest N points
    Needed to roll our own because scipy's linear Nd interpolator refuses to extrapolate
    """

    def __init__(self, points, values, neighbours_to_use=None):
        """By default, interpolates between the 2 * dimensions of space nearest neighbours,
        weighting factors = 1 / distance to neighbour
        """
        self.kdtree = KDTree(points)
        self.values = values
        if neighbours_to_use is None:
            neighbours_to_use = points.shape[1] * 2
        self.neighbours_to_use = neighbours_to_use

    def __call__(self, *args):
        # Call with one point at a time only!!!
        distances, indices = self.kdtree.query(args, self.neighbours_to_use)
        return np.average(self.values[indices], weights=1/np.clip(distances, 1e-6, float('inf')))


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


@numba.jit(numba.int64[:](numba.from_dtype(Hit.get_dtype())[:]),
           nopython=True, cache=True)
def gaps_between_hits(hits):
    """Return array of gaps between hits: a hit's 'gap' is the # of samples before that hit free of other hits.
    The gap of the first hit is 0 by definition.
    Hits should already be sorted by left boundary; we'll check this and throw an error if not.
    """
    n_hits = len(hits)
    gaps = np.zeros(n_hits, dtype=np.int64)
    if n_hits == 0:
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


def count_hits_per_channel(peak, config, weights=None):
    # hist, bins = np.histogram(peak.hits['channel'], weights=weights,
    #                           range=(0, config['n_channels']),
    #                           bins=config['n_channels'])
    return np.bincount(peak.hits['channel'], minlength=config['n_channels'], weights=weights)


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

import logging
import gzip
import json
import re

import numpy as np
from scipy.spatial import cKDTree


##
# Interpolating map class
##

class InterpolateAndExtrapolate(object):
    """Linearly interpolate- or extrapolate between nearest N points
    Needed to roll our own because scipy's linear Nd interpolator refuses to extrapolate
    """

    def __init__(self, points, values, neighbours_to_use=None):
        """By default, interpolates between the 2 * dimensions of space nearest neighbours,
        weighting factors = 1 / distance to neighbour
        """
        self.kdtree = cKDTree(points)
        self.values = values
        if neighbours_to_use is None:
            neighbours_to_use = points.shape[1] * 2
        self.neighbours_to_use = neighbours_to_use

    def __call__(self, args):
        # Call with one point at a time only!!!
        # Now can call with multiple points in form of (self, [(x, y, z), (x, y, z)...])
        if np.any(np.isnan(args)):
            return np.nan
        distances, indices = self.kdtree.query(args, self.neighbours_to_use)
        # We are not ruturning sum of weight when using np.average
        return np.average(self.values[indices], weights=1/np.clip(distances, 1e-6, float('inf')))

    #v__call__ = np.vectorize(__call__, signature = '(),(i)->()')


class InterpolatingMap(object):

    """Construct s a scalar function using linear interpolation, weighted by euclidean distance.

    The map must be specified as a json translating to a dictionary like this:
        'coordinate_system' :   [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...],
        'map' :                 [value1, value2, value3, value4, ...]
        'another_map' :         idem
        'name':                 'Nice file with maps',
        'description':          'Say what the maps are, who you are, your favorite food, etc',
        'timestamp':            unix epoch seconds timestamp
    with the straightforward generalization to 1d and 3d. The default map name is 'map', I'd recommend you use that.

    For a 0d placeholder map, use
        'points': [],
        'map': 42,
        etc

    The json can be gzip compressed: if so, it must have a .gz extension.

    See also examples/generate_mock_correction_map.py
    """
    data_field_names = ['timestamp', 'description', 'coordinate_system', 'name', 'irregular']

    def __init__(self, filename):
        self.log = logging.getLogger('InterpolatingMap')
        self.log.debug('Loading JSON map %s' % filename)

        if filename.endswith('.gz'):
            bla = gzip.open(filename).read()
            self.data = json.loads(bla.decode())
        else:
            with open(filename) as data_file:
                self.data = json.load(data_file)
        self.coordinate_system = cs = self.data['coordinate_system']
        if not len(cs):
            self.dimensions = 0
        else:
            self.dimensions = len(cs[0])
        self.interpolators = {}
        self.map_names = sorted([k for k in self.data.keys() if k not in self.data_field_names])
        self.log.debug('Map name: %s' % self.data['name'])
        self.log.debug('Map description:\n    ' + re.sub(r'\n', r'\n    ', self.data['description']))
        self.log.debug("Map names found: %s" % self.map_names)

        for map_name in self.map_names:
            map_data = np.array(self.data[map_name])
            if self.dimensions == 0:
                # 0 D -- placeholder maps which take no arguments and always return a single value
                itp_fun = lambda *args: map_data  # flake8: noqa
            else:
                itp_fun = InterpolateAndExtrapolate(points=np.array(cs), values=np.array(map_data))

            self.interpolators[map_name] = itp_fun

    def get_value_at(self, position, map_name='map'):
        """Returns the value of the map map_name at a ReconstructedPosition
         position - pax.datastructure.ReconstructedPosition instance
        """
        position_names = ['x', 'y', 'z']
        return self.get_value(*[getattr(position, q) for q in position_names[:self.dimensions]],
                              map_name=map_name)

    # get_value accepts only the map_name keyword argument, but we have to let it accept
    # **kwargs, otherwise python 2 will freak out...
    def get_value(self, *coordinates, **kwargs):
        """Returns the value of the map at the position given by coordinates
        Keyword arguments:
          - map_name: Name of the map to use. By default: 'map'.
        """
        # Ensure nobody passed something other than map_name
        for k in kwargs.keys():
            if k != 'map_name':
                raise ValueError("InterpolatingMap.get_value only takes map_name keyword argument")

        map_name = kwargs.get('map_name', 'map')
        if True: #len(np.array(coordinates).shape) <= 1:
            result = self.interpolators[map_name](coordinates)
            try:
                return float(result[0])
            except(TypeError, IndexError):
                return float(result)

        # Following part is not in use due to couldn't implement np.vectorize for numpy<1.12
        else:
            coordinates = np.array(coordinates).reshape((self.dimensions,-1)).T
            result = self.interpolators[map_name].v__call__(self.interpolators[map_name], coordinates)

            if len(result) == 1:
                return float(result[0])
            else:
                return result

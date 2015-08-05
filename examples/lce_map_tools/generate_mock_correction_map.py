"""
Script to generate example detector maps in the cwd.
The maps can be loaded using InterpolatingMap in dsputils.
"""

import json

import numpy as np

import time
from pax import units


coordinate_system = [
    # Left boundary, Right boundary (inclusive), number of points
    # Boundaries always in cm units!
    ('x',  (-100 * units.cm, 100 * units.cm, 10)),
    ('y',  (-100 * units.cm, 100 * units.cm, 10)),
    ('z',  (-100 * units.cm, 300 * units.cm, 10)),
]


for dim in (1,2,3):
    this_coord_system = coordinate_system[:dim]
    shape = [q[1][2] for q in this_coord_system]
    funny_correction = np.random.normal(0, 1, shape).tolist()

    output_file = 'example_%sd_correction_map.json' % dim

    data = {
        'name':                 'Example %sd correction map' % dim,
        'description':          'A randomly initialized correction map, made by generate_mock_correction_map.py.\n'+\
                                "Contains one map called 'map' initialized with random normal(0,1)'s.",
        'timestamp':            time.time(),
        'coordinate_system':    this_coord_system,
        'map':                  funny_correction,
    }

    json.dump(data, open(output_file, 'w'))
    # For compressed output, use the following:
    # import gzip
    # gzip.open(output_file + '.gz', 'wb').write(json.dumps(data).encode())
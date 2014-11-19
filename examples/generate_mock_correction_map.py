import numpy as np
import json
from pax import units

coordinate_system = [
    # Left boundary, Right boundary (inclusive), number of points
    ('x',  (-100 * units.cm, 100 * units.cm, 10)),
    ('y',  (-100 * units.cm, 100 * units.cm, 10)),
    ('z',  (-100 * units.cm, 300 * units.cm, 10)),
]


for dim in (1,2,3):
    this_coord_system = coordinate_system[:dim]
    shape = [q[1][2] for q in this_coord_system]
    funny_correction  = np.random.normal(0, 1, shape).tolist()

    output_file = 'example_%sd_correction_map.json' % dim

    data = {
        'name'  :                 'Example %sd correction map' % dim,
        'coordinate_system'  :    this_coord_system,
        'map' :                   funny_correction,
    }
    json.dump(data, open(output_file, 'w'))
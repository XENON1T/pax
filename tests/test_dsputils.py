import unittest

import numpy as np

from pax import utils


class TestDSPUtils(unittest.TestCase):

    def test_cluster_by_diff(self):

        example = [-100, 2, 3, 40, 40.5, 41, 100, 101]
        self.assertEqual(
            list(map(list, utils.cluster_by_diff(example, 10))),
            [[-100, ], [2, 3], [40, 40.5, 41, ], [100, 101, ]]
        )

        # return indices
        self.assertEqual(
            list(map(list,
                     utils.cluster_by_diff(example, 10, return_indices=True))),
            [[0, ], [1, 2], [3, 4, 5, ], [6, 7, ]]
        )

        # numpy input
        example = np.array(example)
        self.assertEqual(
            list(map(list, utils.cluster_by_diff(example, 10))),
            [[-100, ], [2, 3], [40, 40.5, 41, ], [100, 101, ]]
        )

        # unsorted input
        # Don't do shuffling here, don't want a nondeterministic test!
        example = [2.0, -100.0, 40.5, 40.0, 3.0, 101.0, 41.0, 100.0]
        self.assertEqual(
            list(map(list, utils.cluster_by_diff(example, 10))),
            [[-100, ], [2, 3], [40, 40.5, 41, ], [100, 101, ]]
        )

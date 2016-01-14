import unittest
import numpy as np

from pax import core, plugin
from pax.datastructure import Event, Peak


class TestPosRecNeuralNet(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON1T', just_testing=True, config_dict={'pax': {
            'plugin_group_names': ['test'],
            'test':               'NeuralNet.PosRecNeuralNet'}})
        self.plugin = self.pax.get_plugin_by_name('PosRecNeuralNet')

    def tearDown(self):
        delattr(self, 'pax')
        delattr(self, 'plugin')

    @staticmethod
    def example_event():
        # Hitpattern taken from "S2_5e3Phts_1e5Evts.root"
        # MC truth (x,y) = (100.833, 177.593) [mm]
        # Recontructed by Yuehuans c++/ROOT code at (102.581, 177.855) [mm]
        # Reconstucted by the PAX implementation at (10.258076101568305 17.785535721706857) [cm]
        top_hits = [7, 8, 8, 5, 8, 10, 6, 9, 3, 7, 6, 4, 5, 2, 1, 0, 7, 1, 3, 1, 4, 2, 5, 1, 4, 3,
                    1, 3, 2, 4, 3, 0, 4, 4, 1, 6, 2, 4, 9, 12, 8, 10, 9, 6, 9, 1, 2, 1, 2, 1, 4, 10,
                    0, 0, 1, 2, 1, 0, 2, 3, 6, 1, 3, 2, 3, 5, 2, 6, 30, 18, 24, 10, 8, 3, 4, 2, 4, 2,
                    1, 4, 3, 4, 5, 5, 2, 1, 2, 2, 2, 4, 12, 48, 139, 89, 19, 9, 3, 4, 2, 3, 1, 1, 6,
                    0, 3, 1, 2, 4, 12, 97, 87, 15, 6, 3, 4, 4, 0, 2, 3, 6, 13, 21, 3, 4, 3, 1, 7]
        hits = np.append(top_hits, np.zeros(254 - 127))
        e = Event.empty_event()
        e.peaks.append(Peak({'left':  5,
                             'right': 9,
                             'type':  'S2',
                             'detector':  'tpc',
                             'area_per_channel': hits}))
        return e

    def test_get_nn_plugin(self):
        self.assertIsInstance(self.plugin, plugin.TransformPlugin)
        self.assertEqual(self.plugin.__class__.__name__, 'PosRecNeuralNet')

    def test_posrec(self):
        e = self.example_event()
        e = self.plugin.transform_event(e)
        self.assertIsInstance(e, Event)
        self.assertEqual(len(e.peaks), 1)
        self.assertEqual(len(e.S2s()), 1)
        self.assertEqual(len(e.peaks[0].reconstructed_positions), 1)
        rp = e.peaks[0].reconstructed_positions[0]
        self.assertEqual(rp.algorithm, self.plugin.name)
        self.assertEqual(rp.x, 10.258076101568305)
        self.assertEqual(rp.y, 17.785535721706857)


if __name__ == '__main__':
    unittest.main()

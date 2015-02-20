import unittest
import numpy as np

from pax import core, plugin
from pax.datastructure import Event, Peak


class TestPosRecNeuralNet(unittest.TestCase):

    def setUp(self):
        self.pax = core.Processor(config_names='XENON100', just_testing=True, config_dict={'pax': {
            'plugin_group_names': ['test'],
            'test':               'NeuralNet.PosRecNeuralNet'}})
        self.plugin = self.pax.get_plugin_by_name('PosRecNeuralNet')

    @staticmethod
    def example_event(channels_with_something):
        bla = np.zeros(243)
        bla[np.array(channels_with_something)] = 1
        e = Event.empty_event()
        e.peaks.append(Peak({'left':  5,
                             'right': 9,
                             'type':  'S2',
                             'detector':  'tpc',
                             'area_per_channel': bla}))
        return e

    def test_get_nn_plugin(self):
        self.assertIsInstance(self.plugin, plugin.TransformPlugin)
        self.assertEqual(self.plugin.__class__.__name__, 'PosRecNeuralNet')

    def test_posrec(self):
        e = self.example_event([40, 41, 42])
        e = self.plugin.transform_event(e)
        self.assertIsInstance(e, Event)
        self.assertEqual(len(e.peaks), 1)
        self.assertEqual(len(e.S2s()), 1)
        self.assertEqual(len(e.peaks[0].reconstructed_positions), 1)
        rp = e.peaks[0].reconstructed_positions[0]
        self.assertEqual(rp.algorithm, 'NeuralNet')
        self.assertEqual(rp.x, 11.076582570681966)
        self.assertEqual(rp.y, 6.831207460290031)


if __name__ == '__main__':
    unittest.main()

import unittest

from pax import core, datastructure


class TestBuildInteractions(unittest.TestCase):

    def setUp(self):  # noqa
        self.pax = core.Processor(config_names='XENON100',
                                  just_testing=True,
                                  config_dict={
                                      'pax': {
                                          'plugin_group_names': ['test'],
                                          'test':               'BuildInteractions.BuildInteractions'},
                                      'BuildInteractions.BuildInteractions': {
                                          'pair_n_s2s': 3,
                                          'pair_n_s1s': 3,
                                          's2_pairing_threshold': 101 * 7 + 1,
                                          'xy_posrec_preference': ['a', 'b']}})
        self.plugin = self.pax.get_plugin_by_name('BuildInteractions')

    def tearDown(self):
        delattr(self, 'pax')
        delattr(self, 'plugin')

    def test_interaction_building(self):
        e = datastructure.Event.empty_event()
        recposes = [datastructure.ReconstructedPosition(x=0, y=0, algorithm='c'),
                    datastructure.ReconstructedPosition(x=1, y=1, algorithm='b'),
                    datastructure.ReconstructedPosition(x=2, y=2, algorithm='a')]

        # 10 S1s, with 10 S2s just behind them
        e.peaks = [datastructure.Peak(type='s1',
                                      detector='tpc',
                                      area=100 * i,
                                      hit_time_mean=100 * i) for i in range(10)]
        e.peaks += [datastructure.Peak(type='s2',
                                       detector='tpc',
                                       area=101 * i,
                                       hit_time_mean=101 * i,
                                       reconstructed_positions=recposes) for i in range(10)]

        e = self.plugin.process_event(e)
        self.assertIsInstance(e.interactions, list)
        self.assertGreater(len(e.interactions), 0)   # So the test fails if the list is empty, rather than error
        self.assertIsInstance(e.interactions[0], datastructure.Interaction)

        # First interaction: (largest S2, largest s1)
        self.assertEqual(e.interactions[0].s1.area, 100 * 9)
        self.assertEqual(e.interactions[0].s2.area, 101 * 9)

        # Largest S1 can't be paired with any further S2s -- no more after it.
        # Second and third interactions are with second largest S1
        self.assertEqual(e.interactions[1].s1.area, 100 * 8)
        self.assertEqual(e.interactions[1].s2.area, 101 * 9)

        self.assertEqual(e.interactions[2].s1.area, 100 * 8)
        self.assertEqual(e.interactions[2].s2.area, 101 * 8)

        # Similarly, fourth and fifth interaction are with third largest S1
        # No sixth interaction: further S2s below pairing threshold, and 3 largest S1s have now been used
        self.assertEqual(e.interactions[3].s1.area, 100 * 7)
        self.assertEqual(e.interactions[3].s2.area, 101 * 9)

        self.assertEqual(e.interactions[4].s1.area, 100 * 7)
        self.assertEqual(e.interactions[4].s2.area, 101 * 8)

        self.assertEqual(len(e.interactions), 5)


if __name__ == '__main__':
    unittest.main()

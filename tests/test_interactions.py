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
                                          'pair_n_s2s': 100,
                                          'pair_n_s1s': 100,
                                          's2_pairing_threshold': 101 * (7 + 1) + 1,
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

        # 10 S1s, with 10 S2s just behind each of them.
        e.peaks = [datastructure.Peak(type='s1',
                                      detector='tpc',
                                      area=100 * (i + 1),
                                      index_of_maximum=1000 * i,
                                      hit_time_mean=100 * i) for i in range(10)]
        e.peaks += [datastructure.Peak(type='s2',
                                       detector='tpc',
                                       area=101 * (i + 1),
                                       index_of_maximum=1010 * i,
                                       hit_time_mean=101 * i,
                                       reconstructed_positions=recposes) for i in range(10)]

        e = self.plugin.process_event(e)
        self.assertIsInstance(e.interactions, list)
        self.assertGreater(len(e.interactions), 0)   # So the test fails if the list is empty, rather than error
        self.assertIsInstance(e.interactions[0], datastructure.Interaction)

        # Interaction 0-9 are with largest S2
        for i in range(9 + 1):
            print(i, e.peaks[e.interactions[i].s1].area, e.peaks[e.interactions[i].s2].area)
            self.assertEqual(e.peaks[e.interactions[i].s1].area, 100 * (1 + 9 - i))
            self.assertEqual(e.peaks[e.interactions[i].s2].area, 101 * (1 + 9))

        # Interaction 10-18 are with second largest S2. Not paired to main S1, since it's after it.
        for i in range(10, 18 + 1):
            print(i, e.peaks[e.interactions[i].s1].area, e.peaks[e.interactions[i].s2].area)
            self.assertEqual(e.peaks[e.interactions[i].s1].area, 100 * (1 + 8 - (i - 10)))
            self.assertEqual(e.peaks[e.interactions[i].s2].area, 101 * (1 + 8))

        # There is no 20th interaction (index 19), the next S2 is below the pairing threshold
        self.assertEqual(len(e.interactions), 19)


if __name__ == '__main__':
    unittest.main()

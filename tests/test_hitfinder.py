import unittest
import numpy as np

from pax import core, datastructure
from pax.plugins.signal_processing import HitFinder, PulseProperties


class TestHitFinder(unittest.TestCase):

    def peak_at(self, i, amplitude=10, width=1):
        w = np.zeros(200)
        w[i:i + width] = amplitude
        return w

    def test_hitfinder(self):
        # Integration test for the hitfinder
        self.pax = core.Processor(config_names='XENON100',
                                  just_testing=True,
                                  config_dict={
                                      'pax': {
                                          'plugin_group_names': ['test'],
                                          'encoder_plugin': None,
                                          'decoder_plugin': None,
                                          'test':               ['PulseProperties.PulseProperties',
                                                                 'HitFinder.FindHits']}})
        for test_w, hit_bounds, pulse_min, pulse_max in (
            # Keep in mind the hitfinder flips the pulse...
            [np.zeros(100), [], 0, 0],
            [np.ones(100), [], 0, 0],
            [-3 * np.ones(100), [], 0, 0],
            [self.peak_at(70, amplitude=-100, width=4), [[70, 73]], 0, 100],
            [self.peak_at(70, amplitude=-100, width=4) + self.peak_at(80, amplitude=10, width=4),
             [[70, 73]], -10, 100],
            [self.peak_at(70, amplitude=-100, width=4) + self.peak_at(80, amplitude=-100, width=4),
             [[70, 73], [80, 83]], 0, 100],
        ):
            e = datastructure.Event(n_channels=self.pax.config['DEFAULT']['n_channels'],
                                    start_time=0,
                                    sample_duration=self.pax.config['DEFAULT']['sample_duration'],
                                    stop_time=int(1e6),
                                    pulses=[dict(left=0,
                                                 raw_data=np.array(test_w).astype(np.int16),
                                                 channel=1)])
            e = self.pax.process_event(e)
            self.assertEqual(hit_bounds, [[hit['left'], hit['right']] for hit in e.all_hits])
            self.assertEqual(pulse_min, e.pulses[0].minimum)
            self.assertEqual(pulse_max, e.pulses[0].maximum)

        delattr(self, 'pax')

    def test_intervals_above_threshold(self):
        # Test of the "hitfinder part" of the hitfinder
        for test_waveform, a in (
            ([0, 1, 2, 0, 4, -1, 60, 700, -4], [[1, 2], [4, 4], [6, 7]]),
            ([1, 1, 2, 0, 4, -1, 60, 700, -4], [[0, 2], [4, 4], [6, 7]]),
            ([1, 0, 2, 3, 4, -1, 60, 700, -4], [[0, 0], [2, 4], [6, 7]]),
            ([1, 0, 2, 3, 4, -1, 60, 700, 800], [[0, 0], [2, 4], [6, 8]]),
            ([0, 0, 2, 3, 4, -1, 60, 700, 800], [[2, 4], [6, 8]]),
        ):
            result_buffer = -1 * np.ones((100, 2), dtype=np.int64)
            hits_found = HitFinder.find_intervals_above_threshold(np.array(test_waveform, dtype=np.float64),
                                                                  high_threshold=0,
                                                                  low_threshold=0,
                                                                  result_buffer=result_buffer,
                                                                  dynamic_low_threshold_coeff=0)
            found = result_buffer[:hits_found]
            self.assertEqual(found.tolist(), a)

    def test_pulse_properties(self):
        # Test of the pulse property computation: baseline, noise, min, max
        for w in (
            [45, 38, 69, 44, 73, 68, 57, 94, 71, 41, 30, 42, 70, 71, 85, 33, 32, 28, 84, 80],
            [47, 67, 51, 84, 81, 25, 67, 23, 62, 20,  5, 21, 97, 88, 74],
            [35,  7, 45, 36, 22, 85, 82, 29, 32, 19, 13, 64, 57, 42, 47],
            [42, 94, 70, 18, 93, 17,  3, 54, 30,  9, 98, 29, 17,  3, 59],
            [64, 43, 86,  3, 53, 11, 20, 62, 81, 23,  4, 96, 24, 45, 65],
            [61, 74, 49, 47, 17, 90, 56, 65,  3, 38, 24, 36, 43, 73, 15],
            [10, 71, 11, 81,  3, 84, 75, 66, 77, 40, 91,  1, 11, 56, 57],
            [42, 57, 14, 43, 43,  6, 85, 47, 55, 15, 79],
            [76, 42, 77, 30,  8, 74, 12, 15,  8, 25],
            [15, 98, 55,  8, 77, 26, 82, 67, 57],
            [42],
        ):
            w = np.array(w, dtype=np.float64)
            results = PulseProperties.compute_pulse_properties(w, baseline_samples=10)
            baseline, baseline_increase, noise, min_w, max_w = results
            bl_before = np.mean(w[:min(len(w), 10)])
            bl_after = np.mean(w[-min(len(w), 10):])
            self.assertEqual(baseline, bl_before)
            self.assertEqual(baseline_increase, bl_after - bl_before)
            w -= baseline
            self.assertEqual(min_w, np.min(w))
            self.assertEqual(max_w, np.max(w))
            below_bl = w[w < 0]
            self.assertAlmostEqual(noise, np.sqrt(np.sum(below_bl**2 / len(below_bl))))

    def test_hit_properties(self):
        # Test of the hit property computation: argmax, area, center
        w = np.array([47, 67, 51, 84, 81, 25, 67, 23, 62, 20,  5, 21, 97, 88, 74], dtype=np.float64)
        hits = np.zeros(100, dtype=datastructure.Hit.get_dtype())

        for raw_hits in (
            [[0, 0], [4, 4], [14, 14]],
            [[0, 1], [13, 14]],
            [[0, 14]]
        ):
            raw_hits = np.array(raw_hits, dtype=np.int64)
            # adc_to_pe, channel, noise_sigma_pe, dt, start, pulse_i, saturation_threshold
            HitFinder.build_hits(w, raw_hits, hits, 1, 1, 1, 1, 0, 0, 0)
            for i, (l, r) in enumerate(raw_hits):
                hitw = w[l:r + 1]
                self.assertAlmostEqual(hits['area'][i], np.sum(hitw))
                self.assertAlmostEqual(hits['index_of_maximum'][i], np.argmax(hitw) + l)
                self.assertAlmostEqual(hits['center'][i], l + np.average(np.arange(len(hitw)), weights=hitw))
                self.assertAlmostEqual(hits['sum_absolute_deviation'][i],
                                       np.average(np.abs(np.arange(len(hitw)) - (hits['center'][i] - l)), weights=hitw))


if __name__ == '__main__':
    unittest.main()

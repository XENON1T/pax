#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pax
----------------------------------

Tests for `pax` module.
"""
import unittest

import numpy as np

from pax.datastructure import Event, Peak, SumWaveform


class TestDatastructure(unittest.TestCase):

    def test_does_not_exist(self):
        e = Event.empty_event()
        with self.assertRaises(AttributeError):
            e.does_not_exist = 3.0

    def test_good(self):
        e = Event.empty_event()
        e.event_number = 3
        self.assertEqual(e.event_number, 3)

    def test_to_json(self):
        e = Event.empty_event()
        e.to_json()

    def test_to_dict(self):
        e = Event.empty_event()
        e.to_dict()

    def test_defaults(self):
        e = Event.empty_event()
        self.assertEqual(e.event_number, 0)
        self.assertEqual(e.peaks, [])

    def test_wrong_type(self):
        e = Event.empty_event()

        with self.assertRaises(TypeError):
            e.event_number = "abc"

        with self.assertRaises(TypeError):
            e.event_number = 3.0

    def test_peaks(self):
        e = Event.empty_event()
        e.peaks = [Peak(
            area=0,
            index_of_maximum=0,
            detector='tpc')]

        self.assertEqual(len(e.peaks), 1)
        self.assertIsInstance(e.peaks[0], Peak)
        self.assertEqual(e.peaks[0].area, 0)

    def test_peak_instantiation(self):
        p = Peak({'area': 3.0,
                  'index_of_maximum': 0,
                  'detector':         'tpc', })
        self.assertIsInstance(p, Peak)
        self.assertEqual(p.area, 3.0)

    def test_peaks_append(self):
        e = Event.empty_event()
        e.peaks.append(Peak({'area': 2.0,
                             'index_of_maximum': 0,
                             'type': 'S1',
                             'detector':         'tpc'}))

        self.assertEqual(len(e.peaks), 1)
        self.assertIsInstance(e.peaks[0], Peak)
        self.assertEqual(e.peaks[0].area, 2.0)

    def test_s1_helper_method(self):
        e = Event.empty_event()
        e.peaks.append(Peak({'area': 2.0,
                             'index_of_maximum': 0,
                             'type': 'S1',
                             'detector':         'tpc', }))

        self.assertEqual(len(e.S1s()), 1)
        self.assertIsInstance(e.S1s()[0], Peak)
        self.assertEqual(e.S1s()[0].area, 2.0)

    def test_s1_helper_method_sort(self):
        areas = [3.0, 1.0, 2.0, 1.2]

        e = Event.empty_event()
        for area in areas:
            e.peaks.append(Peak({'area': area,
                                 'type': 'S2',
                                 'detector': 'tpc'}))

        s2s = e.S2s()
        self.assertEqual(len(s2s), len(areas))

        # Please note the areas should come out in reverse order (largest first)
        areas = sorted(areas, reverse=True)

        for i, area in enumerate(areas):
            self.assertIsInstance(s2s[i], Peak)
            self.assertEqual(s2s[i].area, area)

    def test_waveform_string_name(self):
        w = SumWaveform()
        self.assertIsInstance(w, SumWaveform)

        w.name_of_filter = "blahblah"
        self.assertEqual(w.name_of_filter, "blahblah")

    def test_numpy_array_type(self):
        samples = [3, 4, 5]

        w = SumWaveform()
        self.assertIsInstance(w, SumWaveform)

        self.assertIsInstance(w.samples, np.ndarray)

        with self.assertRaises(TypeError):
            w.samples = samples

        w.samples = np.array(samples, dtype=w.samples.dtype)

        self.assertEqual(len(w.samples),
                         len(samples))
        self.assertIsInstance(w.samples, np.ndarray)
        self.assertEqual(w.samples.dtype, np.float32)

        w.samples = np.array(samples, dtype=np.float32)

        self.assertEqual(len(w.samples),
                         len(samples))
        self.assertIsInstance(w.samples, np.ndarray)
        self.assertEqual(w.samples.dtype, np.float32)


if __name__ == '__main__':
    unittest.main()

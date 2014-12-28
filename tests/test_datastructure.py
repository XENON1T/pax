#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pax
----------------------------------

Tests for `pax` module.
"""
import numpy as np
import unittest

from pax.datastructure import Event, Peak, Waveform


class TestDatastructure(unittest.TestCase):

    def setUp(self):
        pass

    def test_does_not_exist(self):
        e = Event()

        with self.assertRaises(AttributeError):
            e.does_not_exist = 3.0

    def test_good(self):
        e = Event()
        e.event_number = 3.0
        self.assertEqual(e.event_number, 3.0)

    def test_to_json(self):
        e = Event()
        e.to_json()

    def test_to_dict(self):
        e = Event()
        e.to_dict()

    def test_default(self):
        e = Event()
        self.assertEqual(e.event_number, 0)
        self.assertEqual(e.peaks, [])

    def test_wrong_type(self):
        e = Event()

        with self.assertRaises(ValueError):
            e.event_number = "abc"

    def test_peaks(self):
        e = Event()
        e.peaks = [{'area': 0,
                    'index_of_maximum': 0}]

        self.assertEqual(len(e.peaks), 1)
        self.assertIsInstance(e.peaks[0], Peak)
        self.assertEqual(e.peaks[0].area, 0)

    def test_peak_instantiation(self):
        p = Peak({'area': 3.0,
                  'index_of_maximum': 0})
        self.assertIsInstance(p, Peak)
        self.assertEqual(p.area, 3.0)

    def test_peaks_append(self):
        e = Event()
        e.peaks.append(Peak({'area': 2.0,
                             'index_of_maximum': 0,
                             'type': 'S1'}))

        self.assertEqual(len(e.peaks), 1)
        self.assertIsInstance(e.peaks[0], Peak)
        self.assertEqual(e.peaks[0].area, 2.0)

    def test_s1_helper_method(self):
        e = Event()
        e.peaks.append(Peak({'area': 2.0,
                             'index_of_maximum': 0,
                             'type': 'S1'}))

        self.assertEqual(len(e.S1s()), 1)
        self.assertIsInstance(e.S1s()[0], Peak)
        self.assertEqual(e.S1s()[0].area, 2.0)

    def test_s1_helper_method_sort(self):
        areas = [3.0, 1.0, 2.0, 1.2]

        e = Event()
        for area in areas:
            e.peaks.append(Peak({'area': area,
                                 'type': 'S2'}))

        s2s = e.S2s()
        self.assertEqual(len(s2s), len(areas))

        # Please note the areas should come out in reverse order (largest first)
        areas = sorted(areas, reverse=True)

        for i, area in enumerate(areas):
            self.assertIsInstance(s2s[i], Peak)
            self.assertEqual(s2s[i].area, area)

    def test_waveform_string_name(self):
        w = Waveform()
        self.assertIsInstance(w, Waveform)

        w.name_of_filter = "blahblah"
        self.assertEqual(w.name_of_filter, "blahblah")

    def test_numpy_array_type(self):
        samples = [3, 4, 5]

        w = Waveform()
        self.assertIsInstance(w, Waveform)

        self.assertIsInstance(w.samples, np.ndarray)

        # Will trigger a warning from line 87 in fields.py from micromodels:
        # 'converting list to numpy array'
        with self.assertRaises(TypeError):
            w.samples = samples

        w.samples = np.array(samples, dtype=w.samples.dtype)

        self.assertEqual(len(w.samples),
                         len(samples))
        self.assertIsInstance(w.samples, np.ndarray)
        self.assertEqual(w.samples.dtype, np.float64)

        w.samples = np.array(samples, dtype=np.float64)

        self.assertEqual(len(w.samples),
                         len(samples))
        self.assertIsInstance(w.samples, np.ndarray)
        self.assertEqual(w.samples.dtype, np.float64)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()

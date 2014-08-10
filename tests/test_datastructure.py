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

    def test_default(self):
        e = Event()
        self.assertEqual(e.event_number, 0)
        self.assertEqual(e.S1s, [])

    def test_wrong_type(self):
        e = Event()

        with self.assertRaises(ValueError):
            e.event_number = "abc"

    def test_s1s(self):
        e = Event()
        e.S1s = [{'area' : 0,
                      'index_of_maximum': 0}]

        self.assertEqual(len(e.S1s), 1)
        self.assertIsInstance(e.S1s[0], Peak)
        self.assertEqual(e.S1s[0].area, 0)

    def test_peak_instantiation(self):
        p = Peak({'area' : 3.0,
                           'index_of_maximum': 0})
        self.assertIsInstance(p, Peak)
        self.assertEqual(p.area, 3.0)


    def test_s1s_append(self):
        e = Event()
        e.S1s.append(Peak({'area' : 0,
                          'index_of_maximum': 0}))

        self.assertEqual(len(e.S1s), 1)
        self.assertIsInstance(e.S1s[0], Peak)
        self.assertEqual(e.S1s[0].area, 0)

    def test_waveform_string_name(self):
        w = Waveform()
        self.assertIsInstance(w, Waveform)

        w.name_of_filter = "blahblah"
        self.assertEqual(w.name_of_filter, "blahblah")


    def test_numpy_array_type(self):
        samples = [3,4,5]

        w = Waveform()
        self.assertIsInstance(w, Waveform)

        self.assertIsInstance(w.samples, np.ndarray)

        w.samples = samples

        print(w.samples)

        self.assertEqual(len(w.samples),
                         len(samples))
        self.assertIsInstance(w.samples, np.ndarray)
        self.assertEqual(w.samples.dtype, np.int16)

        w.samples = np.array(samples, dtype=np.int16)

        self.assertEqual(len(w.samples),
                         len(samples))
        self.assertIsInstance(w.samples, np.ndarray)
        self.assertEqual(w.samples.dtype, np.int16)

    def tearDown(self):
        pass


if __name__ == '__main__':
	unittest.main()

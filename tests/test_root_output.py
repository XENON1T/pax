import unittest
import numpy as np
from pax import core
import glob

import os
import ROOT


class TestRootOutput(unittest.TestCase):

    def test_root_output(self):
        # Get an event
        mypax = core.Processor(config_names='XENON100', config_dict={'pax': {
            'events_to_process': [0],
            'output': 'Dummy.DummyOutput',
            'encoder_plugin': None}})
        mypax.run()
        event = mypax.get_plugin_by_name('DummyOutput').last_event
        del mypax

        # Write same event to ROOT
        mypax = core.Processor(config_names='XENON100', config_dict={'pax': {
            'events_to_process': [0],
            'output_name': 'test_root_output'}})
        mypax.run()
        del mypax

        self.assertTrue(os.path.exists('test_root_output.root'))
        self.assertTrue(hasattr(ROOT, 'Peak'))

        # Can't test event class loading, event class already loaded during writing
        # ROOTClass.load_pax_event_class_from_root('test_root_output.root')

        f = ROOT.TFile('test_root_output.root')
        t = f.Get('tree')
        t.GetEntry(0)
        root_event = t.events
        self.assertEqual(len(root_event.peaks), len(event.peaks))
        for i in range(len(event.peaks)):
            peak = event.peaks[i]
            root_peak = root_event.peaks[i]
            self.assertEqual(peak.type, root_peak.type)

            # 5th or 6th significant figure appears to be different.. float precision difference?
            self.assertAlmostEqual(peak.area, root_peak.area,
                                   delta=0.0001 * max(1, peak.area))

            # Check area per channel
            self.assertAlmostEqual(peak.area, peak.area_per_channel.sum())
            self.assertAlmostEqual(peak.area, sum(root_peak.area_per_channel),
                                   delta=0.0001 * max(1, peak.area))
            np.testing.assert_array_almost_equal(peak.area_per_channel,
                                                 np.array(list(root_peak.area_per_channel)),
                                                 decimal=4)

    def tearDown(self):
        if os.path.exists('test_root_output.root'):
            os.remove('test_root_output.root')
        for fn in glob.glob('pax_event_class*'):
            os.remove(fn)


if __name__ == '__main__':
    unittest.main()

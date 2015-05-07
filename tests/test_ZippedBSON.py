import unittest

import numpy as np

from pax.datastructure import Event, Pulse
from pax import core
from shutil import rmtree


class TestZippedBSON(unittest.TestCase):

    def test_write_event(self):
        self.addCleanup(rmtree, 'zippedbsontest_tempdir')

        config = {'pax': {
            'plugin_group_names': ['output'],
            'output': 'BSON.WriteZippedBSON',
        },
            'BSON': {'output_name': 'zippedbsontest_tempdir'}
        }

        proc = core.Processor(config_names='XENON100',
                              config_dict=config,
                              just_testing=True)

        write_plugin = proc.get_plugin_by_name('WriteZippedBSON')

        event = Event.empty_event()

        event.pulses = [Pulse(left=i,
                              raw_data=np.array([0, 1, 2, 3], dtype=np.int16),
                              channel=i) for i in range(10)]

        write_plugin.write_event(event)
        write_plugin.shutdown()         # Needed to close the file in time for cleanup to remove it

    def test_write_read(self):
        self.addCleanup(rmtree, 'zippedbsontest_tempdir2')
        config = {'pax': {
            'events_to_process': [0],
            'plugin_group_names': ['input', 'output'],
            'input': 'XED.XedInput',
            'output': 'BSON.WriteZippedBSON',
        },
            'BSON': {
            'output_name': 'zippedbsontest_tempdir2',
            'input_name': 'zippedbsontest_tempdir2'
        }
        }

        pax_xed_to_zippedbson = core.Processor(config_names='XENON100',
                                               config_dict=config)
        pax_xed_to_zippedbson.run()

        config = {'pax': {
            'events_to_process': [0],
            'plugin_group_names': ['input'],
            'input': 'BSON.ReadZippedBSON',
        },
            'BSON': {
            'input_name': 'zippedbsontest_tempdir2'
        }
        }

        pax_zippedbson = core.Processor(config_names='XENON100',
                                        config_dict=config)
        self.read_plugin = pax_zippedbson.input_plugin

        events = list(self.read_plugin.get_events())
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertEqual(len(event.pulses), 1942)

        pulse = event.pulses[0]

        self.assertEqual(pulse.channel, 1)
        self.assertListEqual(pulse.raw_data[0:10].tolist(),
                             [16006, 16000, 15991, 16004, 16004, 16006, 16000, 16000,
                              15995, 16010])
        self.read_plugin.shutdown()    # Needed to close the file in time before dir gets removed

if __name__ == '__main__':
    unittest.main()

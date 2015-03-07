import unittest

import numpy as np

from pax.datastructure import Event, Occurrence
from pax import core
from shutil import rmtree


class TestAvro(unittest.TestCase):

    def test_write_event(self):
        self.addCleanup(rmtree, 'avrotest_tempdir')

        config = {'pax': {
            'plugin_group_names': ['output'],
            'output': 'Avro.WriteAvro',
        },
            'Avro': {'output_name': 'avrotest_tempdir'}
        }

        proc = core.Processor(config_names='XENON100',
                              config_dict=config,
                              just_testing=True)

        write_plugin = proc.get_plugin_by_name('WriteAvro')

        event = Event.empty_event()

        event.occurrences = [Occurrence(left=i,
                                        raw_data=np.array(
                                            [0, 1, 2, 3], dtype=np.int16),
                                        channel=i) for i in range(10)]

        write_plugin.write_event(event)
        write_plugin.shutdown()         # Needed to close the file in time for cleanup to remove it

    def test_write_read(self):
        self.addCleanup(rmtree, 'avrotest_tempdir2')
        config = {'pax': {
            'events_to_process': [0],
            'plugin_group_names': ['input', 'output'],
            'input': 'XED.XedInput',
            'output': 'Avro.WriteAvro',
        },
            'Avro': {
            'output_name': 'avrotest_tempdir2',
            'input_name': 'avrotest_tempdir2'
        }
        }

        pax_xed_to_avro = core.Processor(config_names='XENON100',
                                         config_dict=config)
        pax_xed_to_avro.run()

        config = {'pax': {
            'events_to_process': [0],
            'plugin_group_names': ['input'],
            'input': 'Avro.ReadAvro',
        },
            'Avro': {
            'input_name': 'avrotest_tempdir2'
        }
        }

        pax_avro = core.Processor(config_names='XENON100',
                                  config_dict=config)
        self.read_plugin = pax_avro.input_plugin

        events = list(self.read_plugin.get_events())
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertEqual(len(event.occurrences), 1942)

        occurrence = event.occurrences[0]

        self.assertEqual(occurrence.channel,
                         1)
        self.assertListEqual(occurrence.raw_data[0:10].tolist(),
                             [16006, 16000, 15991, 16004, 16004, 16006, 16000, 16000,
                              15995, 16010])
        self.read_plugin.shutdown()    # Needed to close avro file in time before dir gets removed

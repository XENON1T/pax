import unittest
import tempfile

import numpy as np

from pax.datastructure import Event, Occurrence
from pax import core, utils


class TestAvro(unittest.TestCase):

    def test_write_event(self):
        with tempfile.NamedTemporaryFile() as file:
            config = {'pax': {
                'plugin_group_names': ['output'],
                'output': 'Avro.WriteAvro',
            },
                'Avro': {'output_name': file.name}
            }

            proc = core.Processor(config_names='XENON100',
                                  config_dict=config,
                                  just_testing=True)

            write_plugin = proc.get_plugin_by_name('WriteAvro')

            event = utils.empty_event()

            event.occurrences = [Occurrence(left=i,
                                            raw_data=np.array(
                                                [0, 1, 2, 3], dtype=np.int16),
                                            channel=i) for i in range(10)]

            write_plugin.write_event(event)

    def test_write_read(self):
        config = {'pax': {
            'events_to_process': [0],
            'plugin_group_names': ['input', 'output'],
            'input': 'XED.XedInput',
            'output': 'Avro.WriteAvro',
        },
            'Avro': {
            'output_name': 'test.avro',
            'input_name': 'test.avro'
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
            'input_name': 'test.avro'
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

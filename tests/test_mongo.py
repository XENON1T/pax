import unittest
import os

from pax import core

class TestMongo(unittest.TestCase):
    def setUp(self):  # noqa
        os.system("mongorestore tests/dump")

    def test_triggered(self):
        config = {
            'pax': {
                'plugin_group_names': ['input'],
                'input': 'MongoDB.MongoDBInputTriggered',
            },
            'MongoDB.MongoDBInputTriggered': {
                'collection': 'dataset000002',
            }
        }
        
        pax_avro = core.Processor(config_names='NikhefLab',
                                  config_dict=config)
        self.read_plugin = pax_avro.input_plugin

        events = list(self.read_plugin.get_events())
        self.assertEqual(len(events), 146)

        event = events[0]
        self.assertEqual(len(event.occurrences), 8)

        occurrence = event.occurrences[0]

        self.assertEqual(occurrence.channel,
                         0)
        self.assertListEqual(occurrence.raw_data[0:10].tolist(),
                             [5648, 5647, 5648, 5643, 5648, 5646, 5648, 5647, 5647, 5641])
                             

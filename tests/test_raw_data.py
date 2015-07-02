import unittest
import tempfile

from pax import core

plugins_to_test = [
    {
        'name':         'BSON',
        'read_plugin':  'BSON.ReadBSON',
        'write_plugin': 'BSON.WriteBSON',
    },
    {
        'name':         'ZippedBSON',
        'read_plugin':  'BSON.ReadZippedBSON',
        'write_plugin': 'BSON.WriteZippedBSON',
    },
    {
        'name':         'JSON',
        'read_plugin':  'BSON.ReadJSON',
        'write_plugin': 'BSON.WriteJSON',
    },
    {
        'name':         'XED',
        'read_plugin':  'XED.ReadXED',
        'write_plugin': 'XED.WriteXED',
    },
    {
        'name':         'Avro',
        'read_plugin':  'Avro.ReadAvro',
        'write_plugin': 'Avro.WriteAvro',
    },
    {
        'name':         'ZippedPickles',
        'read_plugin':  'Pickle.ReadZippedPickles',
        'write_plugin': 'Pickle.WriteZippedPickles',
    },
]


class TestRawData(unittest.TestCase):

    def test_write_read(self):
        """Tests a pair of raw data read&write plugins by writing 2 events from XED, then reading them back
        """
        for plugin_info in plugins_to_test:

            # If this test errors in a strange way, the directory may not get deleted.
            # So make it somewhere the os knows to delete it sometime
            tempdir = tempfile.TemporaryDirectory()

            # This print statement is necessary to help user figure out which plugin failed, if any does fail.
            print("\n\nNow testing %s\n" % plugin_info['name'])

            config = {'pax': {'events_to_process': [0, 1],
                              'plugin_group_names': ['input', 'output'],
                              'input': 'XED.ReadXED',
                              'output': plugin_info['write_plugin']},
                      plugin_info['write_plugin']: {'output_name': tempdir.name,
                                                    'overwrite_output': True}}

            mypax = core.Processor(config_names='XENON100', config_dict=config)

            # Wrap this in a try-except, to ensure the read plugin shutdown is run BEFORE the tempdir shutdown
            try:
                mypax.run()
            except Exception as e:
                mypax.shutdown()
                raise e

            config = {'pax': {'events_to_process': [0, 1],
                              'plugin_group_names': ['input'],
                              'input': plugin_info['read_plugin']},
                      plugin_info['read_plugin']: {'input_name': tempdir.name}}

            mypax2 = core.Processor(config_names='XENON100', config_dict=config)
            self.read_plugin = mypax2.input_plugin

            try:
                events = list(self.read_plugin.get_events())
            except Exception as e:
                mypax2.shutdown()
                raise e

            self.assertEqual(len(events), 2)

            event = events[0]
            # If you run the full processing, the number of pulses will change
            # due to concatenation of adjacent pulses
            self.assertEqual(len(event.pulses), 1942)

            pulse = event.pulses[0]

            self.assertEqual(pulse.channel, 1)
            self.assertListEqual(pulse.raw_data[0:10].tolist(),
                                 [16006, 16000, 15991, 16004, 16004, 16006, 16000, 16000,
                                  15995, 16010])

            mypax2.shutdown()    # Needed to close the file in time before dir gets removed

            # Cleaning up the temporary dir explicitly (otherwise tempfile gives warning):
            tempdir.cleanup()

if __name__ == '__main__':
    unittest.main()

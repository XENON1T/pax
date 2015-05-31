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
        'name':         'JSON',
        'read_plugin':  'BSON.ReadJSON',
        'write_plugin': 'BSON.WriteJSON',
    },
    {
        'name':         'ZippedBSON',
        'read_plugin':  'BSON.ReadZippedBSON',
        'write_plugin': 'BSON.WriteZippedBSON',
    },
]


class TestBSON(unittest.TestCase):

    def test_write_read(self):
        # If this test errors in a strange way, the directory may not get deleted.
        # So make it somewhere the os knows to delete it sometime
        tempdir = tempfile.TemporaryDirectory()

        for plugin_info in plugins_to_test:

            # This print statement is necessary to help user figure out which plugin failed, if any does fail.
            print("\n\nNow testing %s\n" % plugin_info['name'])

            config = {'pax': {'events_to_process': [0, 1],
                              'input': 'XED.XedInput',
                              'output': plugin_info['write_plugin']},
                      plugin_info['write_plugin']: {'output_name': tempdir.name,
                                                    'overwrite_output': True}}

            pax_xed_to_bson = core.Processor(config_names='XENON100',
                                             config_dict=config)

            # Wrap this in a try-except, to ensure the read plugin shutdown is run BEFORE the tempdir shutdown
            try:
                pax_xed_to_bson.run()
            except Exception as e:
                pax_xed_to_bson.stop()
                raise e

            config = {'pax': {'events_to_process': [0, 1],
                              'input': plugin_info['read_plugin']},
                      plugin_info['read_plugin']: {'input_name': tempdir.name}}

            pax_bson = core.Processor(config_names='XENON100',
                                      config_dict=config)
            self.read_plugin = pax_bson.input_plugin

            try:
                events = list(self.read_plugin.get_events())
            except Exception as e:
                pax_bson.stop()
                raise e

            self.assertEqual(len(events), 2)

            event = events[0]
            self.assertEqual(len(event.pulses), 1942)

            pulse = event.pulses[0]

            self.assertEqual(pulse.channel, 1)
            self.assertListEqual(pulse.raw_data[0:10].tolist(),
                                 [16006, 16000, 15991, 16004, 16004, 16006, 16000, 16000,
                                  15995, 16010])

            pax_bson.stop()    # Needed to close the file in time before dir gets removed

        # Cleaning up the temporary dir explicitly (otherwise tempfile gives warning):
        tempdir.cleanup()

if __name__ == '__main__':
    unittest.main()

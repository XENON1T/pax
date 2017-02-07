import unittest
import tempfile
import shutil
import six

from pax import core

plugins_to_test = [
    {
        'name':         'ZippedBSON',
        'read_plugin':  'Zip.ReadZipped',
        'decoder':      'BSON.DecodeZBSON',
        'encoder':      'BSON.EncodeZBSON',
        'write_plugin': 'Zip.WriteZipped',
    },
    {
        'name':         'JSON',
        'read_plugin':  'BSON.ReadJSON',
        'write_plugin': 'BSON.WriteJSON',
        'encoder':      None,
        'decoder':      None,
    },
    {
        'name':         'XED',
        'read_plugin':  'XED.ReadXED',
        'decoder':      'XED.DecodeXED',
        'write_plugin': 'XED.WriteXED',
    },
    {
        'name':         'ZippedPickles',
        'read_plugin':  'Zip.ReadZipped',
        'decoder':      'Pickle.DecodeZPickle',
        'encoder':      'Pickle.EncodeZPickle',
        'write_plugin': 'Zip.WriteZipped',
    },
    {
        'name':         'MessagePack',
        'read_plugin':  'Zip.ReadZipped',
        'decoder':      'MessagePack.DecodeMessagePack',
        'encoder':      'MessagePack.EncodeMessagePack',
        'write_plugin': 'Zip.WriteZipped',
    },
]


class TestRawData(unittest.TestCase):

    def test_write_read(self):
        """Tests a pair of raw data read&write plugins by writing 2 events from XED, then reading them back
        """
        for plugin_info in plugins_to_test:

            # Disable the avro test on py2
            if six.PY2 and plugin_info['name'] == 'Avro':
                continue

            # If this test errors in a strange way, the directory may not get deleted.
            # So make it somewhere the os knows to delete it sometime
            tempdir = tempfile.mkdtemp()

            # This print statement is necessary to help user figure out which plugin failed, if any does fail.
            print("\n\nNow testing %s\n" % plugin_info['name'])

            config = {'pax': {'events_to_process': [0, 1],
                              'plugin_group_names': ['input', 'output'],
                              'input': 'XED.ReadXED',
                              'decoder_plugin': 'XED.DecodeXED',
                              'output': plugin_info['write_plugin']},
                      plugin_info['write_plugin']: {'output_name': tempdir,
                                                    'overwrite_output': True}}
            config['pax']['encoder_plugin'] = plugin_info.get('encoder', None)

            mypax = core.Processor(config_names='XENON100', config_dict=config)

            mypax.run()

            config = {'pax': {'events_to_process': [0, 1],
                              'plugin_group_names': ['input'],
                              'encoder_plugin': None,
                              'input': plugin_info['read_plugin']},
                      plugin_info['read_plugin']: {'input_name': tempdir}}
            config['pax']['decoder_plugin'] = plugin_info.get('decoder', None)

            mypax2 = core.Processor(config_names='XENON100', config_dict=config)
            self.read_plugin = mypax2.input_plugin

            try:
                events = list(self.read_plugin.get_events())
                if plugin_info.get('decoder'):
                    dp = mypax2.get_plugin_by_name(plugin_info['decoder'].split('.')[1])
                    events = [dp.transform_event(event) for event in events]
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
            shutil.rmtree(tempdir)


if __name__ == '__main__':
    unittest.main()

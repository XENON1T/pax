import unittest
import tempfile

from pax import core, exceptions


class TestBlinding(unittest.TestCase):

    def test_write_read(self):
        # If this test errors in a strange way, the directory may not get deleted.
        # So make it somewhere the os knows to delete it sometime
        tempdir = tempfile.TemporaryDirectory()

        # Write the first 5 XED events, blind the even events (0, 2, 4)
        config = {'pax': {'stop_after': 5,
                          'plugin_group_names': ['input', 'output'],
                          'output': 'BSON.WriteZippedBSON'},
                  'BSON': {'output_name': tempdir.name,
                           'overwrite_output': True,
                           'encryption_key_file': 'blinding_example_public_key.pklz',
                           # Blind all even events:
                           'blinding_condition': 'lambda event, config: event.event_number % 2 == 0',
                           }}
        mypax = core.Processor(config_names='XENON100', config_dict=config)
        mypax.run()
        del mypax

        # Try to read a blind event without a key: should fail
        config = {'pax': {'plugin_group_names': ['input'],
                          'input': 'BSON.ReadZippedBSON',
                          'events_to_process': [0]},
                  'BSON': {'input_name': tempdir.name}}
        unauthorized_pax = core.Processor(config_names='XENON100', config_dict=config)
        with self.assertRaises(exceptions.CantReadBlindedEvent):
            unauthorized_pax.run()

        # Try to read a nonblind event without a key: should succeed
        config = {'pax': {'plugin_group_names': ['input', 'output'],
                          'input': 'BSON.ReadZippedBSON',
                          'output': 'Dummy.DummyOutput',
                          'events_to_process': [1]},
                  'BSON': {'input_name': tempdir.name}}
        unauthorized_pax = core.Processor(config_names='XENON100', config_dict=config)
        unauthorized_pax.run()
        dummy_out = unauthorized_pax.get_plugin_by_name('DummyOutput')
        self.assertEqual(dummy_out.last_event.event_number, 1)

        # Try to reading all events from a file without a key: should skip blinded events
        config = {'pax': {'plugin_group_names': ['input', 'output'],
                          'input': 'BSON.ReadZippedBSON',
                          'output': 'Dummy.DummyOutput'},
                  'BSON': {'input_name': tempdir.name}}
        unauthorized_pax = core.Processor(config_names='XENON100', config_dict=config)
        unauthorized_pax.run()
        dummy_out = unauthorized_pax.get_plugin_by_name('DummyOutput')
        self.assertEqual(dummy_out.last_event.event_number, 3)
        del unauthorized_pax

        # Try to read a blind event with key
        config = {'pax': {'plugin_group_names': ['input', 'output'],
                          'input': 'BSON.ReadZippedBSON',
                          'output': 'Dummy.DummyOutput',
                          'events_to_process': [0]},
                  'BSON': {'input_name': tempdir.name,
                           'decryption_key_file': 'blinding_example_private_key.pklz'}}
        authorized_pax = core.Processor(config_names='XENON100', config_dict=config)
        authorized_pax.run()
        dummy_out = authorized_pax.get_plugin_by_name('DummyOutput')
        self.assertEqual(dummy_out.last_event.event_number, 0)

        # Try to iterating over events with key
        config = {'pax': {'plugin_group_names': ['input', 'output'],
                          'input': 'BSON.ReadZippedBSON',
                          'output': 'Dummy.DummyOutput'},
                  'BSON': {'input_name': tempdir.name,
                           'decryption_key_file': 'blinding_example_private_key.pklz'}}
        authorized_pax = core.Processor(config_names='XENON100', config_dict=config)
        authorized_pax.run()
        dummy_out = authorized_pax.get_plugin_by_name('DummyOutput')
        self.assertEqual(dummy_out.last_event.event_number, 4)
        del authorized_pax

        # Clean up the temporary dir explicitly (otherwise tempfile gives warning):
        try:
            tempdir.cleanup()
        except PermissionError:
            # Somehow it is still busy??
            pass

if __name__ == '__main__':
    unittest.main()

import unittest
import os

import h5py

from pax import core
import gc

plugins_to_test = [
    {
        'name':         'Table',
        'read_plugin':  'Table.TableReader',
        'write_plugin': 'Table.TableWriter',
        'extension':    'hdf5',
    },
    {
        'name':         'ROOTClass',
        'read_plugin':  'ROOTClass.ReadROOTClass',
        'write_plugin': 'ROOTClass.WriteROOTClass',
        'extension':    'root',
    },

]


class TestReprocessing(unittest.TestCase):

    def test_reprocessing(self):

        for plugin_info in plugins_to_test:
            print("\n\nNow testing %s\n" % plugin_info['name'])

            # Process the first event from the XED file to the format to test
            output1_filename = 'output1.' + plugin_info['extension']
            self.assertFalse(os.path.exists(output1_filename))
            mypax = core.Processor(config_names='XENON100', config_dict={'pax': {
                'events_to_process': [0],
                'output': plugin_info['write_plugin'],
                'output_name': 'output1'}})
            mypax.run()
            del mypax

            # Reprocess from the format to test to an HDF5 file
            output2_filename = 'output_after_%s' % plugin_info['name']
            mypax = core.Processor(config_names='reclassify', config_dict={'pax': {
                'input_name':  'output1.' + plugin_info['extension'],
                'input': plugin_info['read_plugin'],
                'output_name': output2_filename}})
            mypax.run()
            del mypax
            output2_filename += '.hdf5'

            gc.collect()        # Somehow this is necessary to really close all files file...
            os.remove(output1_filename)

            # Open the resulting HDF5
            self.assertTrue(os.path.exists(output2_filename))
            store = h5py.File(output2_filename)
            self.assertTrue('Event' in store)
            self.assertEqual(store['Event'].len(), 1)
            # TODO: the values below change if we change pax!
            self.assertEqual(store['Peak'].len(), 50)
            self.assertEqual(store['Interaction'].len(), 8)

            store.close()
            gc.collect()        # Somehow this is necessary to really close all files file...
            os.remove(output2_filename)


if __name__ == '__main__':
    unittest.main()

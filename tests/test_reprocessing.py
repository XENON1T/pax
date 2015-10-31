import unittest
import os

import h5py
import six

from pax import core

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

    # TODO: delete the HDF5 files
    def test_reprocessing(self):

        for plugin_info in plugins_to_test:
            # Skip the ROOT test until we have Python 2 in Travis
            if plugin_info['extension'] == 'root' and six.PY2:
                continue

            print("\n\nNow testing %s\n" % plugin_info['name'])

            # Process the first event from the XED file to the format to test
            mypax = core.Processor(config_names='XENON100', config_dict={'pax': {
                'events_to_process': [0],
                'output': plugin_info['write_plugin'],
                'output_name': 'output1'}})
            mypax.run()
            self.assertTrue(os.path.exists('output1.' + plugin_info['extension']))

            # Reprocess from the format to test to an HDF5 file
            output2_filename = 'output_after_%s' % plugin_info['name']
            mypax = core.Processor(config_names='reclassify', config_dict={'pax': {
                'input_name':  'output1.' + plugin_info['extension'],
                'input': plugin_info['read_plugin'],
                'output_name': output2_filename}})
            mypax.run()
            output2_filename += '.hdf5'

            # Open both HDF5 files.
            # TODO: This only works if the table writer is the first plugin tested!
            self.assertTrue(os.path.exists('output1.hdf5'))
            self.assertTrue(os.path.exists(output2_filename))
            store1 = h5py.File('output1.hdf5')      # This takes the first output from the HDF5 reprocessing!
            store2 = h5py.File(output2_filename)
            self.assertTrue('Event' in store1)
            self.assertTrue('Event' in store2)

            # Verify both have same number of events, peaks etc
            for dname in ('Event', 'Peak', 'ReconstructedPosition', 'Interaction'):
                self.assertEqual(store1[dname].len(), store2[dname].len())

        # TODO: Clean up
        # Somehow this doesn't work:
        # store1.close()
        # store2.close()
        # os.remove('output1.hdf5')
        # os.remove('output2.hdf5')


if __name__ == '__main__':
    unittest.main()

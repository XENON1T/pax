import unittest

import six
import h5py

from pax import core

plugins_to_test = [
    {
        'name':         'Table',
        'read_plugin':  'Table.TableReader',
        'write_plugin': 'Table.TableWriter',
        'extension':    'hdf5',
    },
    {
        'name':         'ROOT',
        'read_plugin':  'ROOT.ReadROOTClass',
        'write_plugin': 'ROOT.WriteROOTClass',
        'extension':    'root',
    },

]


class TestReprocessing(unittest.TestCase):

    # TODO: delete the HDF5 files
    def test_reprocessing(self):

        # For now, skip this test on  python 2
        if six.PY2:
            return

        for plugin_info in plugins_to_test:
            print("\n\nNow testing %s\n" % plugin_info['name'])

            # Process the first event from the XED file to the format to test
            mypax = core.Processor(config_names='XENON100', config_dict={'pax': {
                'events_to_process': [0],
                'output': plugin_info['write_plugin'],
                'output_name': 'output1'}})
            mypax.run()

            # Reprocess from the format to test to an HDF5 file
            mypax = core.Processor(config_names='reclassify', config_dict={'pax': {
                'input_name':  'output1.' + plugin_info['extension'],
                'input': plugin_info['read_plugin'],
                'output_name': 'output_after_%s' % plugin_info['name']}})
            mypax.run()

            # Open both HDF5 files.
            # TODO: This only works if the table writer is the first plugin tested!
            store1 = h5py.File('output1.hdf5')
            store2 = h5py.File('output2.hdf5')

            # Verify both have same number of peaks etc
            # NOT Hit, it's not read by default (for speed)
            for dname in ('Event', 'Peak', 'ReconstructedPosition'):
                self.assertEqual(store1[dname].len(), store2[dname].len())

        # TODO: Clean up
        # Somehow this doesn't work:
        # store1.close()
        # store2.close()
        # os.remove('output1.hdf5')
        # os.remove('output2.hdf5')


if __name__ == '__main__':
    unittest.main()

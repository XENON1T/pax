import unittest

import h5py

from pax import core


class TestReprocessing(unittest.TestCase):

    # TODO: delete the HDF5 files
    def test_reprocessing(self):

        # Process the first event from the XED file.
        mypax = core.Processor(config_names='XED', config_dict={'pax': {
            'events_to_process': [0],
            'output_name': 'output1'}})
        mypax.run()

        # Reprocess
        mypax = core.Processor(config_names='Reprocess', config_dict={'pax': {
            'input_name':  'output1.hdf5',
            'output_name': 'output2'}})
        mypax.run()

        # Open both hdf5 files
        store1 = h5py.File('output1.hdf5')
        store2 = h5py.File('output2.hdf5')

        # Verify both have same number of peaks etc
        # NOT channelPeak, it's not read by default (for speed)
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

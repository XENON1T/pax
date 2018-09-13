"""Strax data output to files
"""

import numpy as np

from pax.FolderIO import WriteToFolder


class WriteStrax(WriteToFolder):
    """Write raw data to a folder of strax files
    """
    file_extension = 'npy'

    def open(self, filename):
        self.current_file = open(filename, mode='wb')

    def write_event_to_current_file(self, event):
        np.save(self.current_file, event.to_strax(self))

    def close(self):
        self.current_file.close()

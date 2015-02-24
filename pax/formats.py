try:
    import ROOT  # noqa
except ImportError:
    print("You still don't have root. Nice!")
import numpy as np
import os

import h5py
import logging

log = logging.getLogger('BulkOutputInitialization')

try:
    import pandas
except ImportError:
    log.warning("You don't have pandas -- if you use the pandas output, pax will crash!")


class BulkOutputFormat(object):
    """Base class for bulk output formats
    """
    supports_append = False
    supports_write_in_chunks = False
    supports_array_fields = False
    supports_read_back = False
    file_extension = 'DIRECTORY'   # Leave to None for database insertion or something

    def __init__(self, config, log):
        self.config = config
        self.log = log

    def open(self, name, mode):
        # Dir formats don't need to do anything here
        pass

    def close(self):
        # Dir formats don't need to do anything here
        pass

    def read_data(self, df_name, start, end):
        raise NotImplementedError

    def write_data(self, data):
        # Should be overridden by child class
        raise NotImplementedError

    def get_number_of_events(self):
        raise NotImplementedError


class NumpyDump(BulkOutputFormat):
    file_extension = 'npz'
    supports_array_fields = True
    supports_read_back = True
    f = None

    def open(self, name, mode):
        self.filename = name
        if mode == 'r':
            self.f = np.load(self.filename)

    def close(self):
        if self.f is not None:
            self.f.close()

    def write_data(self, data):
        np.savez_compressed(self.filename, **data)

    def read_data(self, df_name, start, end):
        return self.f[df_name][start:end]

    def n_in_data(self, df_name):
        return len(self.f[df_name])


class HDF5Dump(BulkOutputFormat):
    file_extension = 'hdf5'
    supports_array_fields = True
    supports_write_in_chunks = True
    supports_read_back = True

    def open(self, name, mode):
        self.f = h5py.File(name, mode)

    def close(self):
        self.f.close()

    def write_data(self, data):
        for name, records in data.items():
            dataset = self.f.get(name)
            if dataset is None:
                self.f.create_dataset(name, data=records, maxshape=(None,),
                                      compression="gzip",   # hdfview doesn't like lzf?
                                      shuffle=True,
                                      fletcher32=True)
            else:
                oldlen = dataset.len()
                dataset.resize((oldlen + len(records),))
                dataset[oldlen:] = records

    def read_data(self, df_name, start, end):
        return self.f.get(df_name)[start:end]

    def n_in_data(self, df_name):
        return self.f[df_name].len()


class ROOTDump(BulkOutputFormat):
    """Write data to ROOT file

    Convert numpy structered array, every array becomes a TTree.
    Every record becomes a TBranch.
    For the first event the structure of the tree and branches is
    determined, for each branch the proper datatype is determined
    by converting the numpy types to their respective ROOT types.
    This is """
    file_extension = 'root'
    supports_array_fields = True
    supports_write_in_chunks = False
    supports_read_back = False

    # This line makes sure all TTree objects are NOT owned
    # by python, avoiding segfault when garbage collection
    ROOT.TTree.__init__._creates = False

    def open(self, name, mode):
        self.f = ROOT.TFile(name, "RECREATE")
        self.trees = {}
        self.branch_buffers = {}
        # Lookup dictionary for converting python numpy types to
        # ROOT types, strings are handled seperately!
        self.root_type = {'float64': '/D',
                          'int64': '/I',
                          'bool': '/O',
                          'float': '/D',
                          'float32': '/D',
                          'int': '/I',
                          'S': '/C',
                          }

    def close(self):
        self.f.Close()

    def write_data(self, data):
        for treename, records in data.items():

            # Create tree first time write data is called
            if treename not in self.trees:
                self.log.debug("Creating tree: %s" % treename)
                self.trees[treename] = ROOT.TTree(treename, treename)
                self.branch_buffers[treename] = {}
                for fieldname in records.dtype.names:
                    field_data = records[fieldname]
                    dtype = field_data.dtype
                    # Handle array types
                    if len(field_data.shape) > 1:
                        array_len = field_data.shape[1]
                        # Create buffer structure for arrays
                        self.branch_buffers[treename][fieldname] = np.zeros(1,
                                                                            dtype=[('roothell', dtype, (array_len,),)])
                        # Set buffer to use this structure
                        self.trees[treename].Branch(fieldname,
                                                    self.branch_buffers[treename][fieldname],
                                                    '%s[%d]%s' % (fieldname, array_len, self.root_type[str(dtype)]))

                    # Handle all other types (int, float, string, bool)
                    else:
                        sdtype = str(dtype)
                        if sdtype.startswith('|S'):
                            sdtype = 'S'
                        # Store a single element in a buffer of the correct type
                        self.branch_buffers[treename][fieldname] = np.zeros(1, dtype=records[fieldname].dtype)
                        # Set the branch to use this buffer
                        self.trees[treename].Branch(fieldname,
                                                    self.branch_buffers[treename][fieldname],
                                                    fieldname + self.root_type[sdtype])

            # Fill branches
            for record in records:
                for fieldname in record.dtype.names:
                    # Store one record in branch buffer
                    self.branch_buffers[treename][fieldname][0] = record[fieldname]
                # Fill appends the actual data to the branches
                self.trees[treename].Fill()

        # Write to file
        self.f.Write()

##
# Pandas output formats
##


class PandasFormat(BulkOutputFormat):

    pandas_format_key = None

    def write_data(self, data):
        for name, records in data.items():
            # Write pandas dataframe to container
            self.write_pandas_dataframe(name,
                                        pandas.DataFrame.from_records(records))

    def write_pandas_dataframe(self, df_name, df):
        # Write each DataFrame to file
        getattr(df, 'to_' + self.pandas_format_key)(
            os.path.join(self.config['output_name'], df_name + '.' + self.pandas_format_key))


class PandasCSV(PandasFormat):
    pandas_format_key = 'csv'


class PandasHTML(PandasFormat):
    pandas_format_key = 'html'


class PandasJSON(PandasFormat):
    pandas_format_key = 'json'

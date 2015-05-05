"""Formats that pax can read and write

Here are the definitions of how to serialize our data structure to various formats.  Both
read and write routines are required.
"""

import numpy as np
import os

import logging

base_logger = logging.getLogger('BulkOutput')

try:
    import ROOT  # noqa
except ImportError:
    base_logger.warning("ROOT module not imported - if you use the ROOT format, pax will crash!")
except SyntaxError:
    base_logger.warning("ROOT module not made for Python3? - if you use the ROOT format, pax will crash!")

try:
    import pandas
except ImportError:
    base_logger.warning("You don't have pandas -- if you use any of the pandas formats, pax will crash!")


try:
    import h5py
except ImportError:
    base_logger.warning("You don't have the HDF5 output -- if you use the hdf5 format, pax will crash!")


class BulkOutputFormat(object):
    """Base class for bulk output formats
    """
    supports_append = False
    supports_write_in_chunks = False
    supports_array_fields = False
    supports_read_back = False
    prefers_python_strings = False
    file_extension = 'DIRECTORY'   # Leave to None for database insertion or something

    def __init__(self, log=base_logger):
        """Initialize the output format
        log is optional, so you can load these in- and outside of the of pax
        """
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

    def read_data(self, df_name, start=0, end=None):
        if end is None:
            end = self.n_in_data(df_name)
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
                try:
                    self.f.create_dataset(name, data=records, maxshape=(None,),
                                          compression="gzip",   # hdfview doesn't like lzf?
                                          shuffle=True,
                                          fletcher32=True)
                except ValueError:
                    self.log.fatal("Fatal error in creating HDF5 table %s!" % name)
                    self.log.fatal("First record to write was was:")
                    self.log.fatal(records[0])
                    raise
            else:
                oldlen = dataset.len()
                dataset.resize((oldlen + len(records),))
                dataset[oldlen:] = records

    def read_data(self, df_name, start=0, end=None):
        if end is None:
            end = self.n_in_data(df_name)
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

    def __init__(self, *args, **kwargs):
        # This line makes sure all TTree objects are NOT owned
        # by python, avoiding segfaults when garbage collecting
        ROOT.TTree.__init__._creates = False
        super().init(*args, **kwargs)

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
                                                                            dtype=[('temp_name', dtype, (array_len,),)])
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
        self.log.debug("Writing out to TFile")
        self.f.Write()
        self.log.debug("Done writing")


##
# Pandas data formats
##

class PandasFormat(BulkOutputFormat):
    pandas_format_key = None
    prefers_python_strings = True

    def open(self, name, mode):
        self.filename = name

    def write_data(self, data):
        for name, records in data.items():
            # Write pandas dataframe to container
            df = pandas.DataFrame.from_records(records)
            self.write_pandas_dataframe(name, df)

    def write_pandas_dataframe(self, df_name, df):
        # Write each DataFrame to file
        getattr(df, 'to_' + self.pandas_format_key)(
            os.path.join(self.filename, df_name + '.' + self.pandas_format_key))


class PandasCSV(PandasFormat):
    pandas_format_key = 'csv'


class PandasHTML(PandasFormat):
    pandas_format_key = 'html'


class PandasJSON(PandasFormat):
    pandas_format_key = 'json'


class PandasHDF5(PandasFormat):
    supports_append = True
    supports_write_in_chunks = True
    supports_read_back = True
    string_data_length = 32     # HACK, should come from config...
    file_extension = 'hdf5'

    def open(self, name, mode):
        self.store = pandas.HDFStore(name, complevel=9, complib='blosc')

    def close(self):
        self.store.close()

    def write_pandas_dataframe(self, df_name, df):
        if df_name == 'pax_info':
            self.log.error("HACK: didn't write metadata. Some JSON->string->pandas hdf5 issue. Sorry...")
            return
            # Don't worry about pre-setting string field lengths
            # self.store.append(df_name, df, format='table')

        else:
            # Look for string fields (dtype=object), we should pre-set a length for them
            string_fields = df.select_dtypes(include=['object']).columns.values

            self.store.append(df_name, df, format='table',
                              min_itemsize={field_name: self.string_data_length
                                            for field_name in string_fields})

    def read_data(self, df_name, start=0, end=None):
        if end is None:
            end = self.n_in_data(df_name)
            if end == 0:
                return []
        return self.store[df_name][start:end+1].to_records(index=False)

    def n_in_data(self, df_name):
        if df_name not in self.store:
            print(self.store)
            self.log.warning("No %s present in HDF5 file... you sure this is good data?" % df_name)
            return 0
        return len(self.store[df_name])


# List of data formats, pax / analysis code can import this
flat_data_formats = {
    'hdf5':         HDF5Dump,
    'numpy':        NumpyDump,
    'hdf5_pandas':  PandasHDF5,
    'csv':          PandasCSV,
    'html':         PandasHTML,
    'json':         PandasJSON,
    'root':         ROOTDump
}

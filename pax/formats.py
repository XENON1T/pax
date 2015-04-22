"""Input/Output code1 for *processed* data

Here are the definitions of how to serialize our data structure to and from various formats.
Please be careful when editing this file:
 - Do not add any dependencies (e.g. imports at head of file without try-except), this file has to stay import-able
   even if not all the python modules for all the formats are installed.
 - Do not use python3-specific syntax, this file should be importable by python2 applications.
"""
import logging
import os
import re
import subprocess
import sys

import numpy as np

base_logger = logging.getLogger('BulkOutput')

try:
    import ROOT   # noqa
except ImportError:
    base_logger.warning("pyROOT didn't import - if you use the ROOT format, pax will crash!")
except SyntaxError:
    base_logger.warning("pyROOT not made for Python3? - if you use the ROOT format, pax will crash!")

try:
    import pandas
except ImportError:
    base_logger.warning("You don't have pandas -- if you use any of the pandas formats, pax will crash!")


try:
    import h5py
except ImportError:
    base_logger.warning("You don't have h5py -- if you use the hdf5 format, pax will crash!")


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
        raise NotImplementedError

    @property
    def data_types_present(self):
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

    @property
    def data_types_present(self):
        return list(self.f.keys())

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

    @property
    def data_types_present(self):
        return list(self.f.keys())

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
    supports_read_back = True

    # Lookup dictionary for converting python numpy types to
    # ROOT types, strings are handled seperately!
    root_type = {'float32': '/F',
                 'float64': '/D',
                 'int32': '/I',
                 'int64': '/L',
                 'bool': '/O',
                 'S': '/C'}

    numpy_type = {'F': np.float32,
                  'D': np.float64,
                  'I': np.int32,
                  'L': np.int64,
                  'O': np.bool,
                  'C': np.dtype('object')}

    def __init__(self, *args, **kwargs):
        # This line makes sure all TTree objects are NOT owned
        # by python, avoiding segfaults when garbage collecting
        ROOT.TTree.__init__._creates = False
        # DON'T use the Python3 super trick here, we want this code to run in python2 as well!
        BulkOutputFormat.__init__(self, *args, **kwargs)

    def open(self, name, mode):
        if mode == 'w':
            self.f = ROOT.TFile(name, "RECREATE")
            self.trees = {}
            self.branch_buffers = {}
        elif mode == 'r':
            self.f = ROOT.TFile(name)
        else:
            raise ValueError("Invalid mode")

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

    def read_data(self, df_name):
        self.log.warning("ROOT read support is experimental!")
        tree = self.f.Get(df_name)

        # Read the branch names and types
        dt = []
        for b in tree.GetListOfBranches():
            bdata = re.split(r'[\/\[\]]+', b.GetTitle())
            if len(bdata) == 2:
                # Normal field
                bname, btype = bdata
                # TODO HACK ignore string fields for now...
                if btype == 'S':
                    continue
                dt.append((bname, self.numpy_type[btype]))
            elif len(bdata) == 3:
                # Array field
                bname, blen, btype = bdata
                dt.append((bname, self.numpy_type[btype], (int(blen),)))
            else:
                raise ValueError("Strange branch %s in tree %s?" % (b.GetTitle(), df_name))
            if btype == 'S':
                continue

        # Read the data into a numpy record array
        n = self.n_in_data(df_name)
        data = np.zeros(n, dtype=dt)
        for i in range(self.n_in_data(df_name)):
            tree.GetEntry(i)
            for bdata in dt:
                value = getattr(tree, bdata[0])
                if len(bdata) == 3:
                    # Array field: must force this into numpy array to read it
                    # Try list(value) or value[:] if you want to have fun...
                    data[i][bdata[0]][:bdata[2][0]] = value
                else:
                    data[i][bdata[0]] = value

        return data

    @property
    def data_types_present(self):
        # After some trial and error...
        return [x.GetTitle() for x in self.f.GetListOfKeys()]

    def n_in_data(self, df_name):
        return self.f.Get(df_name).GetEntries()


##
# Pandas data formats
##

class PandasFormat(BulkOutputFormat):
    pandas_format_key = None
    supports_array_fields = True

    def open(self, name, mode):
        self.filename = name

    def write_data(self, data):
        for name, records in data.items():
            # Write pandas dataframe to container
            df_series_dict = {}
            for column_name in records.dtype.names:
                if len(records[column_name].shape) != 1:
                    # This is an array field. Pandas doesn't like this: we should convert it to a list of lists,
                    # then store it as an object-dtype Series
                    df_series_dict[column_name] = pandas.Series(records[column_name].tolist(),
                                                                dtype=np.dtype("object"))
                else:
                    df_series_dict[column_name] = pandas.Series(records[column_name],
                                                                dtype=records[column_name].dtype)
            df = pandas.DataFrame(df_series_dict)
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
    prefers_python_strings = True
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

    @property
    def data_types_present(self):
        return list(self.store.keys())

    def n_in_data(self, df_name):
        if df_name not in self.store:
            print(self.store)
            self.log.warning("No %s present in HDF5 file... you sure this is good data?" % df_name)
            return 0
        return len(self.store[df_name])


class ROOTViaPy2(NumpyDump):
    file_extension = 'root'
    supports_array_fields = True
    supports_read_back = False
    f = None

    def __init__(self, py2path=None, converter_script_path=None, **kwargs):
        self.log.warning("ROOT via py2 is experimental -- don't rely on it to discover dark matter yet")
        NumpyDump.__init__(self, **kwargs)
        if py2path is None:
            self.log.warning("Python2 executable path not specified: trying a few likely options")
            for attempt in ('/usr/bin/python',
                            'C:/Python27/python.exe', 'C:/Python26/python.exe',  'C:/Python28/python.exe'):
                if os.path.exists(attempt):
                    py2path = attempt
                    break
            else:
                raise ValueError('Need to pass path to python2 on initialization of RootViaPy2 format!')
        if converter_script_path is None:
            self.log.warning("convert_pax_formats path not passed: trying a few likely options")
            for attempt in (os.path.dirname(sys.executable), '.', '..'):
                attempt = os.path.join(attempt, 'convert_pax_formats')
                if os.path.exists(attempt):
                    converter_script_path = attempt
                    break
            else:
                raise ValueError('Need to pass path to convert_pax_formats on initialization of RootViaPy2 format!')
        self.py2path = py2path
        self.converter_script_path = converter_script_path

    def open(self, name, mode):
        # name is the file with '.root' added
        # We must also construct the temporary numpy filename
        self.root_filename = name
        self.numpy_filename = os.path.splitext(name)[0] + '_temp.npz'
        NumpyDump.open(self, self.numpy_filename, mode)

    def close(self):
        # Stop numpy writing
        NumpyDump.close(self)

        # Convert numpy dump to ROOT
        self.log.info("Converting temporary file to ROOT...")
        subprocess.call([self.py2path, self.converter_script_path,
                         self.numpy_filename, self.root_filename,
                         '--source_format', 'numpy',
                         '--destination_format', 'root',
                         '--pax_path', os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir)])

        # Delete temporary numpy file
        os.remove(self.filename)


# List of data formats, pax / analysis code can import this
flat_data_formats = {
    'hdf5':         HDF5Dump,
    'numpy':        NumpyDump,
    'hdf5_pandas':  PandasHDF5,
    'csv':          PandasCSV,
    'html':         PandasHTML,
    'json':         PandasJSON,
    'root':         ROOTDump,
    'root_via_py2': ROOTViaPy2,
}

import os
import json
import shutil
from collections import OrderedDict

import numpy as np

import time
import pandas
import pax
from pax import plugin, exceptions


class BulkOutput(plugin.OutputPlugin):

    """
    Convert our data structure to 'dataframes' (lists of dictionaries) then write to several output formats:
        numpy record array dump
        any container supported by pandas: HDF5, CSV, JSON, HTML, ...
        (soon: ROOT)
        (soon: several at same time?)

    We make a separate dataframe for each model type in our data structure
    (Event, Peak, ReconstructedPosition, ...)

    The data frames have a hierarchical multi-index, e.g. for the ReconstrucedPosition DataFrame
        the first  index is the event number
        the second index is the Peak's index in event.peaks
        the third  index is the ReconstructedPosition's index in peak.reconstructed_positions
    Each index is also named (in this example, 'Event', 'Peak', 'ReconstructedPosition') for clarity.

    Because pandas and numpy are optimized for working with large data structures, converting/appending each instance
    to pandas/numpy separately would take long. Hence, we store data in lists of 'dtaframes' first, then convert those
    to DataFrames or record arrays once we've collected a bunch of them.

    Available options:

     - output_format:      If hdf, will produce an HDF5 file with tables for each dataframe.
                           If csv, json, html, pickle, ... will produce a folder with files for each dataframe.
                           See http://pandas.pydata.org/pandas-docs/dev/io.html for all possible options;
                           I really just call pandas.to_FORMAT for each DataFrame.
     - output_name:        The name of the output file or folder, WITHOUT file extension.
     - fields_to_ignore:   Fields which will not be stored.

    Further options specific to some output formats:

     - append_data:        Append data to an existing file, if output format supports it
     - write_every:        Write data to disk after every nth event.
                           (If not supported, all data is kept in memory, then written to disk on shutdown.)
     - string_data_length: Maximum length of strings in string data fields.  If you try to store a longer string
                           in any but the first write pass, it will crash!

    """

    def startup(self):
        # Dictionary to contain the data, which will later become pandas DataFrames.
        # Keys are data frame names, values are lists of (index_tuple,dictionary) tuples
        self.dataframes = {}

        self.events_ready_to_be_written = 0

        # Init the output format
        # globals()[classname]() instantiates class named in classname, saves us an eval
        self.output_format = of = globals()[self.config['output_format']](self.config, self.log)

        # Check if options are supported
        if not of.supports_write_every and self.config['write_every'] != float('inf'):
            self.log.warning('Output format %s does not support write_every: will write all at end.' %
                             of.__class__.__name__)
            self.config['write_every'] = float('inf')

        if self.config['append_data'] and not of.supports_append:
            self.log.warning('Output format %s does not support append: setting to False.')
            self.config['append_data'] = False

        # Append extension to outfile, if this format has one
        if of.file_extension and of.file_extension != 'DIRECTORY':
            self.config['output_name'] += '.' + of.file_extension
            self.output_format.config['output_name'] = self.config['output_name']

        # Deal with existing files or non-existing dirs
        outfile = self.config['output_name']
        if os.path.exists(outfile):
            if self.config['append_data'] and of.supports_append:
                self.log.info('Output file/dir %s already exists: appending.' % outfile)
            elif self.config['overwrite_data']:
                print("HALLOOO", self.__class__.__name__, of.file_extension, of.supports_write_every)
                self.log.info('Output file/dir %s already exists, and you wanted to overwrite, so deleting it!'
                              % outfile)
                if self.output_format.file_extension == 'DIRECTORY':
                    shutil.rmtree(outfile)
                    os.mkdir(outfile)
                else:
                    os.remove(outfile)
            else:
                raise exceptions.OutputFileAlreadyExistsError(
                    '%s already exists, and you did not specify append or overwrite...' % outfile)
        elif of.file_extension is 'DIRECTORY':
            # We are a dir output format: dir must exist
            os.mkdir(outfile)

        # Write pax configuration and version to pax_info dataframe
        # Will be a dataframe with one row, indexed by timestamp
        # If you append to an existing HDF5 file, it will make a second row
        # TODO: However, it will probably crash if the configuration is a longer string...
        self.append_to_df('pax_info', ([('timestamp', time.time()), ], {
            'pax_version':             pax.__version__,
            'configuration_json':      json.dumps(self.processor.config),
        }))

        # Open the output file
        self.output_format.open()

    def write_event(self, event):
        # Store all the event data in self.dataframes
        self.model_to_dataframe(event, [('Event', event.event_number), ])

        # Write the data to file if needed
        self.events_ready_to_be_written += 1
        if self.events_ready_to_be_written == self.config['write_every']:
            self.output_format.write_dataframes(self.dataframes)

    def shutdown(self):
        if self.events_ready_to_be_written:
            self.output_format.write_dataframes(self.dataframes)

    def model_to_dataframe(self, m, index_trail):
        """Convert one of our data model instances to a 'dataframe' (data_dict with index_trail),
        handling its subcollections recursively.

        :param m: instance to convert
        :param index_trail: list of (index_name, index) tuples denoting multi-index trail
        """

        # Dict to contain data from this model instance, will be used in dataframe generation
        data_dict = OrderedDict()

        # Grab all data into data_dict -- and more importantly, handle subcollections
        for field_name, field_value in m.get_fields_data():

            if field_name in self.config['fields_to_ignore']:
                continue

            if isinstance(field_value, list):
                if not len(field_value):
                    continue

                # We'll ship model collections off to their own pre-dataframes
                # Convert each child_model to a dataframe, with a new index appended to the index trail
                element_type = type(field_value[0])
                for new_index, child_model in enumerate(field_value):
                    self.model_to_dataframe(child_model, index_trail + [(element_type.__name__,
                                                                         new_index)])

            elif isinstance(field_value, np.ndarray) and not self.output_format.supports_array_fields:
                # NumpyArrayFields must get their own dataframe -- assumes field names are unique!
                # dataframe columns = positions in the array
                self.append_to_df(field_name, (index_trail,
                                               {k: v for k, v in enumerate(field_value)}))
            else:
                data_dict[field_name] = field_value

        # Finally, append the instance we started with to its pre_dataframe
        self.append_to_df(m.__class__.__name__, (index_trail, data_dict))

    def append_to_df(self, dfname, index_and_data_tuple):
        """ Appends an (index, data_dict) tuple to self.dataframes[dfname]
        """
        if dfname in self.dataframes:
            self.dataframes[dfname].append(index_and_data_tuple)
        else:
            self.dataframes[dfname] = [index_and_data_tuple]


class BulkOutputFormat(object):
    """Base class for output formats
    """
    supports_append = False
    supports_write_every = False
    supports_array_fields = False
    file_extension = 'DIRECTORY'   # Leave to None for database insertion or something

    def __init__(self, config, log):
        self.config = config
        self.log = log

    def open(self):
        # Dir formats don't need to do anything here
        pass

    def write_dataframe(self, df_name, df):
        # Should be overridden by child class
        raise NotImplementedError

    def close(self):
        # Dir formats don't need to do anything here
        pass


class NumpyDump(BulkOutputFormat):
    file_extension = 'npz'
    supports_array_fields = True

    def write_dataframes(self, dataframes):

        recarrays = {}

        for df_name, df in dataframes.items():
            # Convert to numpy record array

            # Get the dtype from the first instance
            dtype = []
            index_trail, data = df[0]
            for name, value in index_trail:
                dtype.append(self.numpy_field_dtype(name, value))
            for name, value in data.items():
                dtype.append(self.numpy_field_dtype(name, value))

            # Make values list
            df_values = []
            for index_trail, data_dict in df:
                df_values.append(
                    tuple([x[1] for x in index_trail] + list(data_dict.values()))
                )

            try:
                recarrays[df_name] = np.array(df_values, dtype=dtype)
            except OverflowError:
                print(df_name, dtype)
                raise

        # Save all recarrays to file
        np.savez_compressed(self.config['output_name'], **recarrays)

    def numpy_field_dtype(self, name, x):
        if name == 'configuration_json':
            return name, 'O'
        if isinstance(x, int):
            return name, np.int64
        if isinstance(x, float):
            return name, 'f'
        if isinstance(x, str):
            return name, 'S' + str(self.config['string_data_length'])
        if isinstance(x, np.ndarray):
            return name, x.dtype, x.shape
        else:
            raise TypeError("Don't know numpy type code for %s" % type(x))


##
# Pandas output formats
##

class PandasFormat(BulkOutputFormat):

    pandas_format_key = None

    def write_dataframes(self, dataframes):
        for df_name, df in dataframes.items():
            self.log.debug("Converting %s to pandas DataFrame" % df_name)

            df_data = []
            df_index = []
            df_index_names = []

            for index_trail, data in df:
                df_data.append(data)
                df_index.append([ind[1] for ind in index_trail])
                if not df_index_names:
                    # Should be the same for all rows, so do just once
                    # ... too lazy to wrap & assert
                    df_index_names = [ind[0] for ind in index_trail]

            # Convert to pandas dataframe
            pandas_df = pandas.DataFrame(df_data,
                                         index=pandas.MultiIndex.from_tuples(df_index,
                                                                             names=df_index_names))
            # Write pandas dataframe to container
            self.write_pandas_dataframe(df_name, pandas_df)

    def write_pandas_dataframe(self, df_name, df):
        # Write each DataFrame to a file
        getattr(df, 'to_' + self.pandas_format_key)(
            os.path.join(self.config['output_name'], df_name + '.' + self.pandas_format_key))


class PandasCSV(PandasFormat):
    pandas_format_key = 'csv'


class PandasHTML(PandasFormat):
    pandas_format_key = 'html'


class PandasJSONPandasFormat(PandasFormat):
    pandas_format_key = 'json'


class PandasHDF5(PandasFormat):

    supports_append = True
    supports_write_every = True
    file_extension = 'hdf5'

    def open(self):
        self.store = pandas.HDFStore(self.config['output_name'], complevel=9, complib='blosc')

    def write_pandas_dataframe(self, df_name, df):
        # Write each pandas DataFrame to a table in the hdf
        if df_name == 'pax_info':
            # Don't worry about pre-setting string field lengths
            self.store.append(df_name, df, format='table')

        else:
            # Look for string fields (dtype=object), we should pre-set a length for them
            string_fields = df.select_dtypes(include=['object']).columns.values

            self.store.append(df_name, df, format='table',
                              min_itemsize={field_name: self.config['string_data_length']
                                            for field_name in string_fields})

    def close(self):
        self.store.close()
import os
import json
import shutil

import numpy as np

import time

import h5py
import pax
from pax import plugin, exceptions


class BulkOutput(plugin.OutputPlugin):

    """
    Convert our data structure to numpy record arrays, one for each class.
    Then output to one of several output formats:
        NumpyDump:  numpy record array dump (compressed)
        HDF5Dump:   h5py HDF5  (compressed)
        PandasCSV, PandasJSON, PandasHTML
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
        # Dictionary to contain the data
        # Every class in the datastructure is a key; values are dicts:
        # {
        #   tuples  :       list of data tuples not yet converted to records,
        #   records :       numpy record arrays,
        #   dtype   :       dtype of numpy record (includes field names),
        # }

        configdump = json.dumps(self.processor.config)

        self.data = {
            # Write pax configuration and version to pax_info dataframe
            # Will be a table with one row
            # If you append to an existing HDF5 file, it will make a second row
            # TODO: However, it will probably crash if the configuration is a longer string...
            'pax_info': {
                'tuples':       [(time.time(), str(pax.__version__), configdump)],
                'records':      None,
                'dtype':        [('timestamp', np.int),
                                 ('pax_version', 'S32'),
                                 ('configuration_json', 'S%d' % len(configdump))]}
        }

        self.events_ready_for_conversion = 0

        # Init the output format
        # globals()[classname]() instantiates class named in classname, saves us an eval
        self.output_format = of = globals()[self.config['output_format']](self.config, self.log)

        # Check if options are supported
        if not of.supports_write_in_chunks and self.config['write_in_chunks']:
            self.log.warning('Output format %s does not support write_in_chunks: will write all at end.' %
                             of.__class__.__name__)
            self.config['write_in_chunks'] = False

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

        # Open the output file
        self.output_format.open()

    def write_event(self, event):
        # Store all the event data internally, write to disk when appropriate
        self.model_to_tuples(event, index_fields=[('Event', event.event_number), ])
        self.events_ready_for_conversion += 1

        if self.events_ready_for_conversion == self.config['convert_every']:
            self.convert_to_records()
            if self.config['write_in_chunks']:
                self.write_to_disk()

    def convert_to_records(self):
        for dfname in self.data.keys():
            self.log.debug("Converting %s " % dfname)
            # Convert tuples to records
            newrecords = np.array(self.data[dfname]['tuples'], self.data[dfname]['dtype'])
            # Clear tuples. Enjoy the freed memory.
            self.data[dfname]['tuples'] = []
            # Append new records
            if self.data[dfname]['records'] is None:
                self.data[dfname]['records'] = newrecords
            else:
                self.data[dfname]['records'] = np.concatenate((self.data[dfname]['records'], newrecords))

    def write_to_disk(self):
        self.log.debug("Writing to disk...")
        # If any records are present, call output format to write records to disk
        if not 'Event' in self.data:
            # The processor crashed, don't want to make things worse!
            self.log.warning('No events to write: did you crash pax?')
            return
        if self.data['Event']['records'] is not None:
            self.output_format.write_data({k: v['records'] for k, v in self.data.items()})
        # Delete records we've just written to disk
        for d in self.data.keys():
            self.data[d]['records'] = None

    def shutdown(self):
        if self.events_ready_for_conversion:
            self.convert_to_records()
        self.write_to_disk()

    def model_to_tuples(self, m, index_fields):
        """Convert one of our data model instances to a tuple while storing its field names & dtypes,
           handling subcollections recursively, keeping track of index hierarchy

        :param m: instance to convert
        :param index_fields: list of (index_field_name, value) tuples denoting multi-index trail
        """

        # List to contain data from this model, will be made into tuple later
        m_name = m.__class__.__name__
        m_indices = [x[1] for x in index_fields]
        m_data = []

        # Have we seen this model before? If not, initialize stuff
        first_time_seen = False
        if not m_name in self.data:
            self.data[m_name] = {
                'tuples':         [],
                'records':        None,
                # Initialize dtype with the index fields
                'dtype':          [(x[0], np.int) for x in index_fields],
                'index_depth':    len(m_indices),
            }
            first_time_seen = True

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
                    self.model_to_tuples(child_model, index_fields + [(element_type.__name__,
                                                                       new_index)])

            elif isinstance(field_value, np.ndarray) and not self.output_format.supports_array_fields:
                # NumpyArrayFields must get their own dataframe -- assumes field names are unique!
                # dataframe columns = positions in the array

                # Is this the first time we see this numpy array field?
                if field_name not in self.data:
                    assert first_time_seen    # Must be the first time we see dataframe as well
                    self.data[field_name] = {
                        'tuples':         [],
                        'records':        None,
                        # Initialize dtype with the index fields + every column in array becomes a field.... :-(
                        'dtype':          [(x[0], np.int) for x in index_fields] +
                                          [(str(i), field_value.dtype) for i in range(len(field_value))],
                        'index_depth':    len(m_indices),
                    }

                self.data[field_name]['tuples'].append(tuple(m_indices + list(field_value)))

            else:
                m_data.append(field_value)
                if first_time_seen:
                    # Store this field's data type
                    self.data[m_name]['dtype'].append(self._numpy_field_dtype(field_name, field_value))

        # Store m_indices + m_data in self.data['tuples']
        self.data[m_name]['tuples'].append(tuple(m_indices + m_data))

    def _numpy_field_dtype(self, name, x):
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


class BulkOutputFormat(object):
    """Base class for output formats
    """
    supports_append = False
    supports_write_in_chunks = False
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

    def write_data(self, data):
        np.savez_compressed(self.config['output_name'], **data)



class HDF5Dump(BulkOutputFormat):
    file_extension = 'hdf5'
    supports_array_fields = True
    supports_write_in_chunks = True

    def open(self):
        self.f = h5py.File(self.config['output_name'], "w")

    def write_data(self, data):
        for name, records in data.items():
            dataset = self.f.get(name)
            if dataset is None:
                self.f.create_dataset(name, data=records, maxshape=(None,), compression="gzip")
            else:
                oldlen = dataset.len()
                dataset.resize((oldlen + len(records),))
                dataset[oldlen:] = records

    def close(self):
        self.f.close()




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


class PandasJSONPandasFormat(PandasFormat):
    pandas_format_key = 'json'
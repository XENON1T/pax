import os
import time
import json

import pandas

import pax
from pax import plugin
from pax.micromodels import fields as mm_fields


class WritePandas(plugin.OutputPlugin):

    """
    Convert our event to pandas DataFrames, then write to containers supported by pandas: HDF5, CSV, JSON, HTML, ...

    We make a separate DataFrame for each model type in our data structure
    (Event, Peak, reconstructed position, ...) and every NumpyArrayField as well.

    The data frames have a hierarchical multi-index, e.g. for the ReconstrucedPosition DataFrame
        the first  index is the event number
        the second index is the Peak's index in event.peaks
        the third  index is the ReconstructedPosition's index in peak.reconstructed_positions
    Each index is also named (in this example, 'Event', 'Peak', 'ReconstructedPosition') for clarity.

    Because pandas is optimised for working with large data structures, converting/appending each micromodels instance
    to a DataFrame separately would take an unacceptable amount of CPU time. Hence, we store data in lists of
    dictionaries first ("pre-dataframes"), then convert those to DataFrames once we are ready to write to disk.

    Available options:

     - output_format:      If hdf, will produce an HDF5 file with tables for each dataframe.
                           If csv, json, html, pickle, ... will produce a folder with files for each dataframe.
                           See http://pandas.pydata.org/pandas-docs/dev/io.html for all possible options;
                           I really just call pandas.to_FORMAT for each DataFrame.
     - output_name:        The name of the output file or folder, WITHOUT file extension.
     - fields_to_ignore:   Fields which will not be stored.

    Further options specific to output_format='hdf':

     - append_data:        Append data to an existing HDF5 file.
     - write_every:        Write data to disk after every nth event.
                           (For the other output_formats, all data is kept in memory, then written to disk on shutdown.)
     - string_data_length: Maximum length of strings in string data fields.  If you try to store a longer string
                           in any but the first write pass, it will crash!

    """

    def startup(self):
        # Dictionary to contain the data, which will later become pandas DataFrames.
        # Keys are data frame names, values are lists of (index_tuple,dictionary) tuples
        self.dataframes = {}

        self.outfile = outfile = self.config.get('output_name',         'output')
        self.fields_to_ignore = self.config.get('fields_to_ignore',     [])
        self.output_format = self.config.get('output_format',           'hdf')
        self.write_every = self.config.get('write_every',               10)
        self.append_data = self.config.get('append_data',               False)
        self.string_data_length = self.config.get('string_data_length', 32)

        self.events_ready_to_be_written = 0

        if self.output_format == 'hdf':
            outfile += '.hdf'

            if not self.append_data:
                # Delete earlier output file, if one exists
                if os.path.exists(outfile):
                    os.remove(outfile)

            self.store = pandas.HDFStore(outfile, complevel=9, complib='blosc')
        else:
            # Ensure the directory for the output files exists
            if not os.path.isdir(outfile):
                os.mkdir(outfile)

            # We need to write in one go at the end:
            self.write_every = float('inf')

        # Write pax configuration and version to pax_info dataframe
        # Will be a dataframe with one row, indexed by timestamp
        # If you append to an existing HDF5 file, it will make a second row
        # TODO: However, it will probably crash if the configuration is a longer string...
        def json_object_handler(obj):
            if isinstance(obj, set):
                return list(obj)
            raise TypeError

        self.append_to_df('pax_info', ([('timestamp', time.time()), ], {
            'pax_version':             pax.__version__,
            'configuration_json':      json.dumps(self.processor.config, default=json_object_handler),
        }))

    def write_event(self, event):

        # Store all the event data in self.dataframes
        self.mm_to_dataframe(event, [('Event', event.event_number)])

        # Write the data to file if needed
        self.events_ready_to_be_written += 1
        if self.events_ready_to_be_written == self.write_every:
            self.write_dataframes()

    def shutdown(self):

        if self.events_ready_to_be_written:
            self.write_dataframes()

        if self.output_format == 'hdf':
            self.store.close()

    def mm_to_dataframe(self, mm, index_trail):
        """Convert a MicroModels class instance to a pre-dataframe (data_dict with index_trail),
        handling its subcollections recursively.

        :param mm: instance to convert
        :param index_trail: list of (index_name, index) tuples denoting multi-index trail
        """
        # Dict to contain data from this mm instance, will be used in dataframe generation
        data_dict = {}

        # Grab all data into data_dict -- and more imporantly, handle subcollections
        for field_name, field_instance in mm.get_fields().items():

            field_value = getattr(mm, field_name)

            if field_name in self.fields_to_ignore:
                continue

            if isinstance(field_instance, mm_fields.ModelCollectionField):

                # We'll ship model collections off to their own pre-dataframes
                # Convert each child_mm to a dataframe, with a new index appended to the index trail
                for new_index, child_mm in enumerate(field_value):
                    self.mm_to_dataframe(child_mm, index_trail + [(child_mm.__class__.__name__, new_index)])

            elif isinstance(field_instance, mm_fields.NumpyArrayField):

                # NumpyArrayFields also get their own dataframe -- assumes field names are unique!
                # dataframe columns = positions in the array
                self.append_to_df(field_name, (index_trail,
                                               {k: v for k, v in enumerate(field_value)}))

            else:
                data_dict[field_name] = field_value

        # Finally, append the instance we started with to its pre_dataframe
        self.append_to_df(mm.__class__.__name__, (index_trail, data_dict))

    def append_to_df(self, dfname, index_and_data_tuple):
        """ Appends an (index, data_dict) tuple to self.dataframes[dfname]
        """
        if dfname in self.dataframes:
            self.dataframes[dfname].append(index_and_data_tuple)
        else:
            self.dataframes[dfname] = [index_and_data_tuple]

    def write_dataframes(self):

        # Convert pre-dataframes to actual pandas DataFrames
        for df_name, df in self.dataframes.items():

            self.log.debug("Converting %s to pandas DataFrame" % df_name)

            df_data = []
            df_index = []
            df_index_names = []

            for index_trail, data in self.dataframes[df_name]:
                df_data.append(data)
                df_index.append([ind[1] for ind in index_trail])
                if not df_index_names:
                    # Should be the same for all rows, so do just once
                    # ... too lazy to wrap & assert
                    df_index_names = [ind[0] for ind in index_trail]

            self.dataframes[df_name] = pandas.DataFrame(df_data,
                                                        index=pandas.MultiIndex.from_tuples(df_index,
                                                                                            names=df_index_names))

        # Write the DataFrames to disk
        for df_name, df in self.dataframes.items():

            self.log.debug("Writing %s" % df_name)

            if self.output_format == 'hdf':

                # Write each DataFrame to a table
                if df_name == 'pax_info':
                    # Don't worry about pre-setting string field lengths
                    self.store.append(df_name, df, format='table')

                else:
                    # Look for string fields (dtype=object), we should pre-set a length for them
                    string_fields = df.select_dtypes(include=['object']).columns.values

                    self.store.append(df_name, df, format='table',
                                      min_itemsize={field_name: self.string_data_length
                                                    for field_name in string_fields})

            else:
                # Write each DataFrame to a file
                getattr(df, 'to_' + self.output_format)(
                    os.path.join(self.outfile, df_name + '.' + self.output_format))

        # Clean up for the next batch of events
        self.events_ready_to_be_written = 0
        self.dataframes = {}

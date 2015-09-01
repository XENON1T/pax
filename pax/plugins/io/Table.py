import os
import shutil
from bson.json_util import dumps

import numpy as np

from pax import plugin, exceptions, datastructure
from pax.data_model import Model
from pax.formats import flat_data_formats


class TableWriter(plugin.OutputPlugin):

    """Output data to flat table formats
    Convert our data structure to numpy record arrays, one for each class (Event, Peak, ReconstructedPosition, ...).
    Then output to one of several output formats (see formats.py)

    For each index, an extra column is added (e.g. ReconstructedPosition has an extra column 'Event', 'Peak'
    and 'ReconstructedPosition', each restarting from 0 whenever the corresponding higher-level entity changes).

    The timestamp, pax configuration and version number are stored in a separate table/array: pax_info.

    Because numpy arrays are optimized for working with large data structures, converting/appending each instance
    separately would take long. Hence, we store data in lists first, then convert those once we've collected a bunch.

    Available configuration options:

     - output_format:      Name of output format to produce. Must be child class of TableFormat
     - output_name:        The name of the output file or folder, WITHOUT file extension.
     - fields_to_ignore:   Fields which will not be stored.
     - overwrite_data:     If True, overwrite if a file/directory with the same name exists
     - string_data_length: Maximum length of strings in string data fields; longer strings will be truncated.
                           (the pax configuration is always stored fully)
     - append_data:        Append data to an existing file, if output format supports it
     - buffer_size:        Convert to numpy record arrays after every nth event.
     - write_in_chunks:    Write to disk every time after converting to numpy record arrays, if the output format
                           supports it. Else all data is kept in memory, then written to disk on shutdown.
    """

    def startup(self):
        # Check if user forgot to specify some fields in fields_to_ignore
        if 'hits' not in self.config['fields_to_ignore'] and 'all_hits' not in self.config['fields_to_ignore']:
            raise ValueError("You must ignore either (peak.)hits or (event.)all_hits to avoid duplicating"
                             "the hit info in the tabular output.")
        if 'sum_waveforms' not in self.config['fields_to_ignore']:
            self.log.warning("You did not ignore the (event.)sum_waveforms field. This means you're trying to dump the"
                             "entire event sum waveform to the tabular output. "
                             "I'll try, but if it fails, you have been warned...")
        if 'raw_data' not in self.config['fields_to_ignore']:
            self.log.warning("You did not ignore the (pulse.)raw_data field. This means you're trying to dump the"
                             "entire raw data for every pulse to the tabular output!!! "
                             "I'll try, but if it fails, you have been warned...")

        metadata_dump = dumps(self.processor.get_metadata())

        # Dictionary to contain the data
        # Every class in the datastructure is a key; values are dicts:
        # {
        #   tuples  :       list of data tuples not yet converted to records,
        #   records :       numpy record arrays,
        #   dtype   :       dtype of numpy record (includes field names),
        # }
        self.data = {
            # Write pax configuration and version to pax_info dataframe
            # Will be a table with one row
            # If you append to an existing HDF5 file, it will make a second row
            # TODO: However, it will probably crash if the configuration is a
            # longer string...
            'pax_info': {
                'tuples':       [(metadata_dump,)],
                'records':      None,
                'dtype':        [('metadata_json', 'S%d' % len(metadata_dump))]}
        }

        self.events_ready_for_conversion = 0

        # Init the output format
        self.output_format = of = flat_data_formats[self.config['output_format']](log=self.log)

        if self.config['append_data'] and self.config['overwrite_data']:
            raise ValueError('Invalid configuration for TableWriter: Cannot both'
                             ' append and overwrite')

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

        # Deal with existing files or non-existing dirs
        outfile = self.config['output_name']
        if os.path.exists(outfile):
            if self.config['append_data'] and of.supports_append:
                self.log.info('Output file/dir %s already exists: appending.' % outfile)
            elif self.config['overwrite_data']:
                self.log.info('Output file/dir %s already exists, and you '
                              'wanted to overwrite, so deleting it!' % outfile)
                if self.output_format.file_extension == 'DIRECTORY':
                    self.log.warning('Deleting recursively %s...',
                                     outfile)
                    shutil.rmtree(outfile)  # Deletes directory and contents
                    os.mkdir(outfile)
                else:
                    os.remove(outfile)
            else:
                raise exceptions.OutputFileAlreadyExistsError('%s already exists, and you did not specify append or '
                                                              'overwrite...' % outfile)
        elif of.file_extension is 'DIRECTORY':
            # We are a dir output format: dir must exist
            os.mkdir(outfile)

        # Open the output file
        self.log.info("Opening output file/directory %s" % self.config['output_name'])
        self.output_format.open(self.config['output_name'], mode='w')

    def write_event(self, event):
        """Receive event and determine what to do with it

        Store all the event data internally, write to disk when appropriate.
        This function follows the plugin API.
        """
        # Hack to convert s1, s2 in interaction objects to numbers
        for i in range(len(event.interactions)):
            for q in ('s1', 's2'):
                peak = getattr(event.interactions[i], q)
                object.__setattr__(event.interactions[i], q, event.peaks.index(peak))

        self._model_to_tuples(event,
                              index_fields=[('Event', event.event_number), ])
        self.events_ready_for_conversion += 1

        if self.events_ready_for_conversion >= self.config['buffer_size']:
            self._convert_to_records()
            if self.config['write_in_chunks']:
                self._write_to_disk()

    def _convert_to_records(self):
        """Convert buffer data to numpy record arrays
        """
        for dfname in self.data.keys():
            self.log.debug("Converting %s " % dfname)
            # Set index at which next set of tuples begins
            self.data[dfname]['first_index'] = len(self.data[dfname]['tuples']) + \
                self.data[dfname].get('first_index', 0)
            # Convert tuples to records
            newrecords = np.array(self.data[dfname]['tuples'],
                                  self.data[dfname]['dtype'])
            # Clear tuples. Enjoy the freed memory.
            self.data[dfname]['tuples'] = []
            # Append new records
            if self.data[dfname]['records'] is None:
                self.data[dfname]['records'] = newrecords
            else:
                self.data[dfname]['records'] = np.concatenate((self.data[dfname]['records'],
                                                               newrecords))
        self.events_ready_for_conversion = 0

    def _write_to_disk(self):
        """Write buffered data to disk
        """
        self.log.debug("Writing to disk...")
        # If any records are present, call output format to write records to
        # disk
        if 'Event' not in self.data:
            # The processor crashed, don't want to make things worse!
            self.log.warning('No events to write: did you crash pax?')
            return
        if self.data['Event']['records'] is not None:
            self.output_format.write_data({k: v['records'] for k,
                                           v in self.data.items()})
        # Delete records we've just written to disk
        for d in self.data.keys():
            self.data[d]['records'] = None

    def shutdown(self):
        if self.events_ready_for_conversion:
            self._convert_to_records()
        self._write_to_disk()
        self.output_format.close()

    def get_index_of(self, mname):
        # Returns index +1 of last last entry in self.data[mname]. Returns -1
        # if no mname seen before.
        if mname not in self.data:
            return 0
        else:
            return self.data[mname]['first_index'] + len(self.data[mname]['tuples'])

    def _model_to_tuples(self, m, index_fields):
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
        if m_name not in self.data:
            self.data[m_name] = {
                'tuples':               [],
                'records':              None,
                # Initialize dtype with the index fields
                'dtype':                [(x[0], np.int) for x in index_fields],
                'index_depth':          len(m_indices),
                # Dictionary of collection field's {field_names: collection
                # class name}
                'subcollection_fields': m.get_list_field_info(),
                'first_index':          0
            }
            first_time_seen = True

        # Handle the subcollection fields first
        collection_field_name = {}    # Maps child types to collection field names
        for field_name, field_type in self.data[m_name]['subcollection_fields'].items():
            if field_name in self.config['fields_to_ignore']:
                continue

            field_value = getattr(m, field_name)
            child_class_name = field_type.__name__
            collection_field_name[child_class_name] = field_name

            # Store the absolute start index & number of children
            child_start = self.get_index_of(child_class_name)
            n_children = len(field_value)
            if first_time_seen:
                # Add data types for n_x (unless already present, e.g. peak.n_hits) and x_start field names.
                # Will have int type.
                if not hasattr(m, 'n_%s' % field_name):
                    self.data[m_name]['dtype'].append(self._numpy_field_dtype('n_%s' % field_name, 0))
                self.data[m_name]['dtype'].append(self._numpy_field_dtype('%s_start' % field_name, 0))
            if not hasattr(m, 'n_%s' % field_name):
                m_data.append(n_children)
            m_data.append(child_start)

            # We'll ship model collections off to their own tuples (later record arrays)
            # Convert each child_model to a dataframe, with a new index
            # appended to the index trail
            for new_index, child_model in enumerate(field_value):
                self._model_to_tuples(child_model,
                                      index_fields + [(type(child_model).__name__,
                                                       new_index)])

        # Handle the ordinary (non-subcollection) fields
        for field_name, field_value in m.get_fields_data():

            if field_name in self.config['fields_to_ignore'] or isinstance(field_value, list):
                continue

            elif isinstance(field_value, Model):
                # Individual child model: store the child number instead
                # Note: only works if collection is in the same model!
                child_class_name = field_value.__class__.__name__

                if first_time_seen:
                    self.data[m_name]['dtype'].append(self._numpy_field_dtype(field_name, 0))

                # We know the number the next child should get. What number is this one?
                for child_i_from_back, child in enumerate(reversed(getattr(m,
                                                                           collection_field_name[child_class_name]))):
                    # Note the is instead of ==, we really want the same child, not just one that looks the same
                    if child is field_value:
                        break
                else:
                    # Assume fake child, fallthrough in datastructure (e.g. event without S1)
                    m_data.append(datastructure.INT_NAN)
                    continue

                child_i = self.get_index_of(child_class_name) - 1 - child_i_from_back
                m_data.append(child_i)

            elif isinstance(field_value, np.ndarray) and not self.output_format.supports_array_fields:
                # Hack for formats without array field support: NumpyArrayFields must get their own dataframe
                #  -- assumes field names are unique!
                # dataframe columns = str(positions in the array) ('0', '1',
                # '2', ...)

                # Is this the first time we see this numpy array field?
                if field_name not in self.data:
                    # Must be the first time we see dataframe as well
                    assert first_time_seen
                    self.data[field_name] = {
                        'tuples':          [],
                        'records':         None,
                        # Initialize dtype with the index fields + every column
                        # in array becomes a field.... :-(
                        'dtype':           [(x[0], np.int) for x in index_fields] +
                                           [(str(i), field_value.dtype)
                                            for i in range(len(field_value))],
                        'index_depth':     len(m_indices),
                    }
                self.data[field_name]['tuples'].append(tuple(m_indices + field_value.tolist()))

            else:
                m_data.append(field_value)
                if first_time_seen:
                    # Store this field's data type
                    self.data[m_name]['dtype'].append(self._numpy_field_dtype(field_name,
                                                                              field_value))

        # Store m_indices + m_data in self.data['tuples']
        self.data[m_name]['tuples'].append(tuple(m_indices + m_data))

    def _numpy_field_dtype(self, name, x):
        """Return field dtype of numpy record with field name name and value (of type of) x
        """
        if name == 'metadata_dump':
            return name, 'O'
        if isinstance(x, int):
            return name, np.int64
        if isinstance(x, float):
            return name, 'f'
        if isinstance(x, str):
            if self.output_format.prefers_python_strings:
                return name, 'O'
            else:
                return name, 'S' + str(self.config['string_data_length'])
        if isinstance(x, np.ndarray):
            return name, x.dtype, x.shape
        else:
            # Some weird numpy type, hopefully
            return name, type(x)


class TableReader(plugin.InputPlugin):

    """Read data from TableWriter for reprocessing

    'Reprocessing' means: reading in old processed data, then start somewhere in middle of processing chain,
    (e.g. redo classification), and finally write to new file.

    Reprocessing WITHOUT reading in the individual hits is very fast. This is fine for re-doing
    peak classification and anything higher-level we may come up with.

    For re-doing clustering and/or peak property computation you must read in the hits, which is slow.
    The speed is set by overhead in the datastructure: 1/3 of runtime due to type checking, rest due to
    converting between all the internal formats). We can try to optimize this at some point.

    However, for this kind of reprocessing we may eventually need to read the raw data and build a sum waveform
    as well, which takes time too.

    TODO: Check if all the tables / dnames we want to read in are present in the file, else give error
    """

    def startup(self):
        self.chunk_size = self.config['chunk_size']
        self.read_hits = self.config['read_hits']
        self.read_recposes = self.config['read_recposes']
        self.read_interactions = self.config['read_interactions']

        self.output_format = of = flat_data_formats[self.config['format']](log=self.log)
        if not of.supports_read_back:
            raise NotImplementedError("Output format %s does not "
                                      "support reading data back in!" % self.config['format'])

        of.open(name=self.config['input_name'], mode='r')

        self.dnames = ['Event', 'Peak']
        if self.read_hits:
            self.dnames.append('Hit')
        if self.read_recposes:
            self.dnames.append('ReconstructedPosition')
        if self.read_interactions:
            self.dnames.append('Interaction')

        # Dict of numpy record arrays just read from disk, waiting to be sorted
        self.cache = {}
        self.max_n = {x: of.n_in_data(x) for x in self.dnames}
        self.number_of_events = self.max_n['Event']
        self.current_pos = {x: 0 for x in self.dnames}

    def get_events(self):
        """Get events from processed data source
        """
        of = self.output_format

        for event_i in range(self.number_of_events):

            in_this_event = {}

            # Check if we should fill the cache.
            for dname in self.dnames:     # dname -> event, peak, etc.

                # Check if what is stored in the cache for 'dname' is either
                # nonexistent, empty, or incomplete.  If so, keep reading new
                # chunks of data to populate the cache.
                while dname not in self.cache or len(self.cache[dname]) == 0 \
                        or self.cache[dname][0]['Event'] == self.cache[dname][-1]['Event']:

                    # If no data of this dname left in the file, we of course
                    # stop filling the cache
                    if self.current_pos[dname] == self.max_n[dname]:
                        break

                    new_pos = min(self.max_n[dname],
                                  self.current_pos[dname] + self.chunk_size)
                    new_chunk = of.read_data(dname,
                                             self.current_pos[dname],
                                             new_pos)
                    self.current_pos[dname] = new_pos

                    # Add new chunk to cache
                    bla = np.concatenate((self.cache.get(dname,
                                                         np.empty(0,
                                                                  dtype=new_chunk.dtype)),
                                          new_chunk))
                    self.cache[dname] = bla

            # What number is the next event?
            this_event_i = self.cache['Event'][0]['Event']

            # Get all records belonging to this event:
            for dname in self.dnames:
                mask = self.cache[dname]['Event'] == this_event_i
                in_this_event[dname] = self.cache[dname][mask]

                # Chop records in this event from cache
                inverted_mask = True ^ mask  # XOR
                self.cache[dname] = self.cache[dname][inverted_mask]

            # Convert records to pax data
            assert len(in_this_event['Event']) == 1
            e_record = in_this_event['Event'][0]
            peaks = in_this_event['Peak']
            peak_numbers = peaks['Peak'].tolist()      # Needed to build interaction objects

            event = self.convert_record(datastructure.Event, e_record)

            if self.config.get('read_hits_only', False):
                # Read in only hits for reclustering
                for hit_record in in_this_event['Hit']:
                    cp = self.convert_record(datastructure.Hit, hit_record)
                    event.all_hits.append(cp)
            else:

                for peak_i, p_record in enumerate(peaks):
                    peak = self.convert_record(datastructure.Peak, p_record)

                    if self.read_recposes:
                        for rp_record in in_this_event['ReconstructedPosition'][
                                in_this_event['ReconstructedPosition']['Peak'] == peak_i]:
                            peak.reconstructed_positions.append(
                                self.convert_record(datastructure.ReconstructedPosition,
                                                    rp_record)
                            )

                    if self.read_hits:
                        for hit_record in in_this_event['Hit'][(in_this_event['Hit']['Peak'] == peak_i)]:
                            cp = self.convert_record(datastructure.Hit,
                                                     hit_record)
                            peak.hits.append(cp)
                            event.all_hits.append(cp)

                    event.peaks.append(peak)

                if self.read_interactions:
                    interactions = in_this_event['Interaction']
                    for intr_i, intr_record in enumerate(interactions):
                        intr = self.convert_record(datastructure.Interaction, intr_record, ignore_type_checks=True)
                        # Hack to build s1 and s2 attributes from peak numbers
                        for q in ('s1', 's2'):
                            peak = event.peaks[peak_numbers.index(getattr(intr, q))]
                            object.__setattr__(intr, q, peak)
                        event.interactions.append(intr)

            yield event

    def convert_record(self, class_to_load_to, record, ignore_type_checks=False):
        # We defined a nice custom init for event... ahem... now we have to do
        # cumbersome stuff...
        if class_to_load_to == datastructure.Event:
            result = datastructure.Event(n_channels=self.config['n_channels'],
                                         start_time=record['start_time'],
                                         stop_time=record['stop_time'],
                                         sample_duration=record['sample_duration'])
        else:
            result = class_to_load_to()
        for k, v in self._numpy_record_to_dict(record).items():
            # If result doesn't have this attribute, ignore it
            # This happens for n_peaks etc. and attributes that have been
            # removed
            if hasattr(result, k):
                if ignore_type_checks:
                    object.__setattr__(result, k, v)
                else:
                    setattr(result, k, v)
        return result

    def _numpy_record_to_dict(self, record):
        """Convert a single numpy record to a dict (keys=field names, values=values)
        """
        names = record.dtype.names
        result = {}
        for k, v in zip(names, record):
            # Skip index fields, if present
            if k in ('Event', 'Peak', 'Hit', 'ReconstructedPosition'):
                continue
            if isinstance(v, np.bytes_):
                v = v.decode("utf-8")
            result[k] = v
        return result

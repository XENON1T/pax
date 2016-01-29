"""Interfacing to MongoDB

MongoDB is used as a data backend within the DAQ.  For example, 'kodiaq', which
reads out the digitizers, will write data to MongoDB.  This data from kodiaq can
either be triggered or untriggered. In the case of untriggered, an event builder
must be run on the data and will result in triggered data.  Input and output
classes are provided for MongoDB access.  More information is in the docstrings.
"""
import time

import numpy as np
import numba
import snappy

from pax.datastructure import Event, Pulse
from pax import plugin, units


class MongoDBReader:
    use_monary = False

    def startup(self):
        self.sample_duration = self.config['sample_duration']

        # Setup keys and options for finding pulses
        self.start_key = self.config['start_key']
        self.stop_key = self.config['stop_key']
        self.sort_key = ([(self.start_key, 1), (self.stop_key, 1)])
        self.mongo_find_options = {'sort': self.sort_key}
        # Used to have 'cursor_type': pymongo.cursor.CursorType.EXHAUST here as well

        # Load the digitizer channel -> PMT index mapping
        self.detector = self.config['detector']
        self.pmts = self.config['pmts' if self.detector == 'tpc' else 'pmts_muon_veto']
        self.pmt_mappings = {(x['digitizer']['module'],
                              x['digitizer']['channel']): x['pmt_position'] for x in self.pmts}

        # Connect to the runs db
        mm = self.processor.mongo_manager
        self.runs = mm.get_database('run').get_collection('runs')
        self.update_run_doc()

        # Connect to the input database
        # Input database connection settings are specified in the run doc
        # TODO: can username, host, password, settings different from standard db access info?
        # Then we have to pass extra args to MongoManager
        self.input_info = nfo = self.run_doc['detectors'][self.detector]['mongo_buffer']
        if self.use_monary:
            if not mm.monary_enabled:
                self.log.error("Use of monary was requested, but monary did not import. Reverting to pymongo.")
                self.use_monary = False

        if self.use_monary:
            self.monary_client = mm.get_database(database_name=nfo['database'],
                                                 uri=nfo['address'],
                                                 monary=True)

        else:
            self.input_collection = mm.get_database(database_name=nfo['database'],
                                                    uri=nfo['address']).get_collection(nfo['collection'])
            self.input_collection.ensure_index(self.sort_key)

    def do_monary_query(self, query, fields, types):
        if not self.use_monary:
            raise RuntimeError("use_monary is false, monary query isn't going to happen")
        database = self.input_info['database']
        collection = self.input_info['collection']
        return self.monary_client.query(database, collection, query, fields, types)

    def update_run_doc(self):
        """Update the internal run doc within this class

        This is useful for example checking if a run has ended.
        """
        self.run_doc = self.runs.find_one({'_id': self.config['run_doc_id']})
        self.data_taking_ended = 'endtimestamp' in self.run_doc

    def _to_mt(self, x):
        # Takes time in ns and converts to samples
        # makes number small by 10
        return int(x * units.ns // self.sample_duration)

    def _from_mt(self, x):
        # Takes time in samples and converts to ns
        # Makes number bigger by 10
        return int(x * self.sample_duration // units.ns)


class MongoDBReadUntriggered(plugin.InputPlugin, MongoDBReader):

    """Read from MongoDB and build events
    This will perform a sliding window trigger on the pulse midpoint times.
    No PMT pulse data is read in this class to ensure speed.
    """
    last_event_number = 0
    use_monary = True

    def startup(self):
        MongoDBReader.startup(self)

        # Load a few more config settings
        self.detector = self.config['detector']
        self.search_window = self.config.get('search_window', 60 * units.s)

        # Initialize the buffer to hold the pulse ranges
        self.pulse_ranges_buffer = np.ones((self.config.get('pulse_ranges_buffer_size', int(1e7)), 2),
                                           dtype=np.int64) * -1

        self.log.info("Building events with:")
        self.log.info("\tSliding window: %0.2f us", self.config['window'] / units.us)
        self.log.info("\tMultiplicity: %d coincident pulses", self.config['multiplicity'])
        self.log.info("\tLeft extension: %0.2f us", self.config['left_extension'] / units.us)
        self.log.info("\tRight extension: %0.2f us", self.config['right_extension'] / units.us)
        self.log.info("\tSearch window: %s s", self.search_window / units.s)

    def get_events(self):
        self.last_time_searched = 0  # ns

        # Used to timeout if DAQ crashes and no data will come
        time_of_last_daq_response = time.time()

        if not self.use_monary:
            self.log.debug("Total number of pulses in collection: %d" % self.input_collection.count())

        while True:
            # Update the run document, so we know if the run ended.
            # This must happen before querying for more data, to avoid a race condition where the run ends
            # while the query is in process.
            self.update_run_doc()

            # Query for pulse start & stop times within a large search window
            search_after = self.last_time_searched   # TODO: add configurable delay?
            self.log.info("Searching for pulses after %s", sampletime_fmt(search_after))
            query = {self.start_key: {'$gt': self._to_mt(search_after),
                                      '$lt': self._to_mt(search_after + self.search_window)}}

            if self.use_monary:
                # TODO: pass sort key
                start_times, stop_times = self.do_monary_query(query=query,
                                                               fields=[self.start_key, self.stop_key],
                                                               types=['int64', 'int64'])
                x = np.round(0.5 * (start_times + stop_times) * self.sample_duration).astype(np.int64)

            else:
                times = list(self.input_collection.find(query,
                                                        projection=[self.start_key, self.stop_key],
                                                        **self.mongo_find_options))
                # Convert response from list of dictionaries of start & stop time in samples
                # to numpy array of pulse midpoint times in pax time units (ns)
                x = np.zeros(len(times), dtype=np.int64)
                for i, doc in enumerate(times):
                    x[i] = int(0.5 * (self._from_mt(doc[self.start_key]) + self._from_mt(doc[self.stop_key])))

            self.last_time_searched += self.search_window

            if not len(x):
                # No more pulse data found. Did the run end?
                # self.data_taking_ended is updated in self.update_run_doc(), called right after we started the loop
                if self.data_taking_ended:
                    self.log.info("No pulses found, and data taking ended.")
                    status = self.runs.update_one({'_id': self.config['run_doc_id']},
                                                  {'$set': {'detectors.%s.trigger.status' % self.detector: 'processed',
                                                            'detectors.%s.trigger.ended' % self.detector: True}})
                    self.log.debug("Answer from updating rundb doc: %s" % status)
                    break
                if time.time() - time_of_last_daq_response > 60:  # seconds   # TODO: configure
                    raise RuntimeError('Timed out waiting for new data (DAQ crash?)')
                self.log.info("No data found, but run is still ongoing. Waiting for new data.")
                time.sleep(1)  # TODO: configure
                continue

            self.log.info("Acquired pulse time data in range [%s, %s]", sampletime_fmt(x[0]), sampletime_fmt(x[-1]))

            if self.config['mega_event']:
                self.log.info("Building mega-event with all data in search range.")
                event_ranges = [[x[0], x[-1]]]

            else:
                # Do the sliding window coindidence trigger
                event_ranges = self.sliding_window(x,
                                                   window=self.config['window'],
                                                   multiplicity=self.config['multiplicity'],
                                                   left=self.config['left_extension'],
                                                   right=self.config['right_extension'])
                self.log.info("Found %d event ranges", len(event_ranges))
                self.log.debug(event_ranges)

            for i, (t0, t1) in enumerate(event_ranges):
                self.last_event_number += 1
                yield Event(n_channels=self.config['n_channels'],
                            start_time=t0,
                            sample_duration=self.sample_duration,
                            stop_time=t1,
                            dataset_name=self.run_doc['name'],
                            event_number=self.last_event_number)

            # Update time of last DAQ reponse
            # Better to do this here than before building: if the building * processing takes a long time
            # we don't want to think the DAQ has crashed
            time_of_last_daq_response = time.time()

    def sliding_window(self, x, window=1000, multiplicity=3, left=-10, right=7):
        """Sliding window cluster finder (with range extension)

        x is a list of times.  A window will slide over the values in x and
        this function will return all ranges with more than 'multiplicity'
        of pulses.
        Also, left and right will be added to ranges, where left can be the
        drift length.
        """
        if left > 0:
            raise ValueError("Left offset must be negative!")
        self.pulse_ranges_buffer *= -1
        n_ranges_found = _sliding_window_numba(x, window, multiplicity,
                                               left, right, self.pulse_ranges_buffer,
                                               len(self.pulse_ranges_buffer))
        if n_ranges_found >= len(self.pulse_ranges_buffer):
            self.log.fatal("Too many pulse ranges to fit in buffer... :-(?")
        return self.pulse_ranges_buffer[:n_ranges_found]


class MongoDBReadUntriggeredFiller(plugin.TransformPlugin, MongoDBReader):

    """Read pulse data from untriggered MongoDB into event ranges provided by trigger MongoDBReadUntriggered
    This is a separate plugin, since reading the raw pulse data is the expensive operation we want to parallelize.
    """
    def startup(self):
        MongoDBReader.startup(self)

    def transform_event(self, event):
        t0, t1 = event.start_time, event.stop_time  # ns
        self.log.debug("Fetching pulse data for event in range [%s, %s]", sampletime_fmt(t0), sampletime_fmt(t1))
        self.log.debug("Total number of pulses in collection: %d" % self.input_collection.count())

        self.mongo_iterator = self.input_collection.find({self.start_key: {"$gte": self._to_mt(t0),
                                                                           "$lte": self._to_mt(t1)}},
                                                         **self.mongo_find_options)
        for i, pulse_doc in enumerate(self.mongo_iterator):
            digitizer_id = (pulse_doc['module'], pulse_doc['channel'])
            if digitizer_id in self.pmt_mappings:
                # Fetch the raw data
                data = pulse_doc['data']
                if self.input_info['compressed']:
                    data = snappy.decompress(data)

                time_within_event = self._from_mt(pulse_doc[self.start_key]) - t0  # ns

                event.pulses.append(Pulse(left=self._to_mt(time_within_event),
                                          raw_data=np.fromstring(data,
                                                                 dtype="<i2"),
                                          channel=self.pmt_mappings[digitizer_id]))
            else:
                self.log.warning("Found data from digitizer module %d, channel %d,"
                                 "which doesn't exist according to PMT mapping! Ignoring...",
                                 pulse_doc['module'], pulse_doc['channel'])

        self.log.debug("%d pulses in event %s" % (len(event.pulses), event.event_number))
        return event


class MongoDBWriteTriggered(plugin.OutputPlugin):
    """Write raw or processed events to MongoDB

    These events are already triggered.  We first convert event class to a dict,
    then pymongo converts that to BSON.  We have to convert the numpy arrays to
    bytes because otherwise we get type errors since pymongo doesn't know about
    numpy.
    """

    def startup(self):
        mm = self.processor.mongo_manager
        # TODO: grab output_name from somewhere???
        self.output_collection = mm.get_database('output').get_collection(self.config['output_name'])

    def write_event(self, event_class):
        # Convert event class to pymongo-able dictionary, then insert into the database
        event_dict = event_class.to_dict(convert_numpy_arrays_to='bytes')
        self.output_collection.write(event_dict)


# TODO: This needs tests!!!
@numba.jit(numba.int64(numba.int64[:],
                       numba.int64, numba.int64, numba.int64, numba.int64,
                       numba.int64[:, :], numba.int64), nopython=True)
def _sliding_window_numba(x,
                          window, multiplicity, left, right,
                          ranges_buffer, ranges_buffer_size):
    current_range = -1
    i = 0  # Start of range to test
    j = 0  # End of range to test

    while j < x.size:  # For every pulse... extend end
        if x[j] - x[i] > window:  # If time diff greater than window, form new cluster
            if j - i > multiplicity:  # If more than 100 pulses, trigger
                if current_range != -1 and ranges_buffer[current_range, 1] + window > x[i]:
                    ranges_buffer[current_range, 1] = x[j - 1] + right
                else:
                    current_range += 1
                    if current_range > ranges_buffer_size - 1:
                        break
                    ranges_buffer[current_range, 0] = x[i] + left
                    ranges_buffer[current_range, 1] = x[j - 1] + right
            i += 1
        else:
            j += 1

    if j - i > multiplicity and current_range < ranges_buffer_size - 1:
        current_range += 1
        ranges_buffer[current_range, 0] = x[i] + left
        ranges_buffer[current_range, 1] = x[j - 1] + right

    return current_range + 1


def sampletime_fmt(num):
    """num is in 1s of ns"""
    for x in ['ns', 'us', 'ms', 's', 'ks', 'Ms', 'G', 'T']:
        if num < 1000.0:
            return "%3.3f %s" % (num, x)
        num /= 1000.0
    return "%3.1f %s" % (num, 's')

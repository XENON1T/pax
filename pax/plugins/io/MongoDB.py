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
import pymongo

from pax.datastructure import Event, Pulse, EventProxy
from pax import plugin, units


class MongoDBReader:
    use_monary = False

    def startup(self):
        self.sample_duration = self.config['sample_duration']
        self.secret_mode = self.config['secret_mode']

        # Setup keys and options for finding pulses
        self.start_key = self.config['start_key']
        self.stop_key = self.config['stop_key']
        self.sort_key = ([(self.start_key, 1), (self.stop_key, 1)])
        self.mongo_find_options = {'sort': self.sort_key}
        # Used to have 'cursor_type': pymongo.cursor.CursorType.EXHAUST here as well

        # Load the digitizer channel -> PMT index mapping
        self.detector = self.config['detector']
        self.pmts = self.config['pmts']
        self.pmt_mappings = {(x['digitizer']['module'],
                              x['digitizer']['channel']): x['pmt_position'] for x in self.pmts}
        self.ignored_channels = []

        # Connect to the runs db
        mm = self.processor.mongo_manager
        self.runs = mm.get_database('run').get_collection('runs_new')
        self.refresh_run_doc()

        # Do we need to use monary?
        # Only relevant for readuntriggered... code stink
        if self.config['mega_events']:
            # In this mode the "trigger" just checks if there are any actually pulses, nothing more
            self.use_monary = False

        if self.use_monary:
            if not mm.monary_enabled:
                self.log.error("Use of monary was requested, but monary did not import. Reverting to pymongo.")
                self.use_monary = False

        # Connect to the input database
        # Input database connection settings are specified in the run doc
        # TODO: can username, host, password, settings different from standard db access info?
        # Then we have to pass extra args to MongoManager
        self.input_info = None
        for doc in self.run_doc['data']:
            if doc['type'] == 'untriggered':
                self.input_info = doc
                break
        else:
            raise ValueError("Invalid run document: none of the 'data' entries contain untriggered data!")

        nfo = self.input_info     # shorthand
        self.input_info['database'] = nfo['location'].split('/')[-1]
        if self.use_monary:
            self.monary_client = mm.get_database(database_name=nfo['database'],
                                                 uri=nfo['location'],
                                                 monary=True)

        self.log.debug("Grabbing collection %s" % nfo['collection'])
        self.input_collection = mm.get_database(database_name=nfo['database'],
                                                uri=nfo['location']).get_collection(nfo['collection'])
        self.log.debug("Creating index in input collection")
        self.input_collection.create_index(self.sort_key, background=True)
        self.log.debug("Succesfully grabbed collection %s" % nfo['collection'])

    def do_monary_query(self, query, fields, types, **kwargs):
        if not self.use_monary:
            raise RuntimeError("use_monary is false, monary query isn't going to happen")
        database = self.input_info['database']
        collection = self.input_info['collection']
        return self.monary_client.query(database, collection, query, fields, types, **kwargs)

    def refresh_run_doc(self):
        """Update the internal run doc within this class
        (does not change anything in the runs database)

        This is useful for example checking if a run has ended.
        """
        self.log.debug("Updating run doc")
        self.run_doc = self.runs.find_one({'_id': self.config['run_doc_id']})
        self.data_taking_ended = 'end' in self.run_doc

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
    do_output_check = False

    def startup(self):
        MongoDBReader.startup(self)

        # Load a few more config settings
        self.detector = self.config['detector']
        self.search_window = self.config['search_window']

        # Initialize the buffer to hold the pulse ranges
        self.pulse_ranges_buffer = np.ones((self.config.get('pulse_ranges_buffer_size', int(1e7)), 2),
                                           dtype=np.int64) * -1

        self.log.info("Building events with:")
        self.log.info("\tSearch window: %s s", self.search_window / units.s)
        if self.config['mega_events']:
            self.log.info("\t'Mega-event' mode: no trigger, just segment data in fixed chunks (size of search window)")
        else:
            self.log.info("\tCoincidence trigger mode")
            self.log.info("\tMultiplicity: %d coincident pulses", self.config['multiplicity'])
            self.log.info("\tSliding window: %0.2f us", self.config['window'] / units.us)
            self.log.info("\tLeft extension: %0.2f us", self.config['left_extension'] / units.us)
            self.log.info("\tRight extension: %0.2f us", self.config['right_extension'] / units.us)

    def get_events(self):
        last_time_searched = 0  # ns
        pulses_read = 0         # Number of pulses read by the event builder
        pulses_in_events = 0    # Number of pulses placed inside events
        pulses_in_collection = self.input_collection.count()
        total_event_length = 0  # ns
        total_pulse_length = 0  # ns

        # Used to timeout if DAQ crashes and no data will come
        time_of_last_daq_response = time.time()
        last_time_to_search = float('inf')
        if pulses_in_collection == 0:
            raise ValueError("This run does not even contain one pulse!")

        self.log.info("Total number of pulses in collection: %d" % pulses_in_collection)

        while True:
            # If the run is still ongoing, update the run document, so we know if the run ended.
            # This must happen before querying for more data, to avoid a race condition where the run ends
            # while the query is in process.
            if not self.data_taking_ended:
                self.refresh_run_doc()

            if self.data_taking_ended and last_time_to_search == float('inf'):
                # If the run ended, find the "last" pulse
                # best would be to query for the last stop time, but don't want to trigger a re-sort!
                cu = self.input_collection.find().sort(self.start_key, direction=pymongo.DESCENDING).limit(1)
                last_time_to_search = self._from_mt(list(cu)[0][self.stop_key])
                self.log.info("The DAQ has stopped, last pulse time is %s" % pax_to_human_time(last_time_to_search))

                # Does this correspond roughly to the run end time? If not, warn, DAQ may have crashed.
                end_of_run_t = (self.run_doc['end'].timestamp() -
                                self.run_doc['start'].timestamp()) * units.s
                if end_of_run_t > last_time_to_search + 60 * units.s:
                    self.log.warning("Run is %s long according to run db, but last pulse in collection starts at %s. "
                                     "Did the DAQ crash?" % (pax_to_human_time(end_of_run_t),
                                                             pax_to_human_time(last_time_to_search)))

                # Add 1 search window margin (in case the pulse is long, or another pulse started sooner but ends later)
                last_time_to_search += self.search_window

            # Query for pulse start & stop times within a large search window
            search_after = last_time_searched   # TODO: add configurable delay?

            if self.config['mega_events']:
                if not self.data_taking_ended:
                    raise ValueError("Mega event mode not available if run is still ongoing")

                # Segment the data in fixed-size chunks
                event_ranges = np.array([[search_after, search_after + self.search_window]])

                # Advance an entire search window
                last_time_searched += self.search_window

            else:
                # Run the coincidence trigger algorithm
                self.log.info("Searching for pulses after %s", pax_to_human_time(search_after))

                query = {self.start_key: {'$gt': self._to_mt(search_after),
                                          '$lt': self._to_mt(search_after + self.search_window)}}

                if self.use_monary:
                    # TODO: pass sort key
                    start_times, stop_times = self.do_monary_query(query=query,
                                                                   fields=[self.start_key, self.stop_key],
                                                                   sort=self.start_key,
                                                                   types=['int64', 'int64'])
                    # Note: no fence post +1, pulse right boundary should be inclusive of the last sample
                    total_pulse_length += np.sum(stop_times - start_times) * self.sample_duration
                    x = np.round(0.5 * (start_times + stop_times) * self.sample_duration).astype(np.int64)

                else:
                    times = list(self.input_collection.find(query,
                                                            projection=[self.start_key, self.stop_key],
                                                            **self.mongo_find_options))
                    # Convert response from list of dictionaries of start & stop time in samples
                    # to numpy array of pulse midpoint times in pax time units (ns)
                    x = np.zeros(len(times), dtype=np.int64)
                    for i, doc in enumerate(times):
                        start = self._from_mt(doc[self.start_key])
                        stop = self._from_mt(doc[self.stop_key])
                        total_pulse_length += start - stop
                        x[i] = int(0.5 * (start + stop))

                pulses_read += len(x)

                if len(x):
                    self.log.info("Acquired pulse time data in range [%s, %s]",
                                  pax_to_human_time(x[0]),
                                  pax_to_human_time(x[-1]))
                    # Compute the last pulse start time, for search window advancement
                    if self.use_monary:
                        last_start_time = self._from_mt(start_times[-1])
                    else:
                        last_start_time = self._from_mt(times[-1][0][-1])

                elif not self.data_taking_ended:
                    # We may be running ahead of the DAQ, then sleep a bit
                    # TODO: this wait condition isalso triggered if there is a big hole in the data.
                    # In that case we won't continue until the run has ended...
                    self.log.info("No pulses in search window, but run is still ongoing: "
                                  "sleeping a bit, not advancing search window.")
                    if time.time() - time_of_last_daq_response > 60:  # seconds
                        if not self.secret_mode:
                            status = self.runs.update_one(
                                {'_id': self.config['run_doc_id']},
                                {'$set': {'trigger.status': 'timeout',
                                          'trigger.ended': True}})
                            self.log.debug("Answer from updating rundb doc: %s" % status)
                        raise RuntimeError('Timed out waiting for new data (DAQ crash?)')
                    time.sleep(1)  # TODO: configure
                    continue

                else:
                    self.log.info("No pulse data found")

                # Do the sliding window coindidence trigger
                event_ranges = self.sliding_window(x,
                                                   window=self.config['window'],
                                                   multiplicity=self.config['multiplicity'],
                                                   left=self.config['left_extension'],
                                                   right=self.config['right_extension'])
                self.log.info("Found %d event ranges", len(event_ranges))
                self.log.debug(event_ranges)

                # Advance the search window
                if self.data_taking_ended:
                    last_time_searched += self.search_window
                else:
                    # We can't advance the entire search window, since the DAQ may not have filled it all the way.
                    # The condition len(x) == 0 and not self.data_taking_ended has already been dealt with.
                    # Notice we can't use x[-1], but need last_start_time, to avoid boundary problems
                    # since the query is on the start_time, but x is the pulse midpoint time.
                    last_time_searched = last_start_time

            if last_time_searched > last_time_to_search:
                self.log.info("Searched beyond last pulse in run collection: stopping event builder")
                break

            for i, (t0, t1) in enumerate(event_ranges):
                total_event_length += t1 - t0   # Note: no fence post +1, t1 is stop time of last sample
                yield EventProxy(event_number=self.last_event_number, data=[t0, t1])
                self.last_event_number += 1

            # Update time of last DAQ reponse
            # This must be done here: building events and getting them all into the queue can take a long time
            # especially since the queue must maintain a fixed size.
            # We don't want to think the DAQ has crashed just because we're using too few CPUS.
            time_of_last_daq_response = time.time()

        # All went well - print out status information
        events_built = self.last_event_number + 1
        end_of_run_info = {'trigger.events_built': events_built,
                           'trigger.ended': True,
                           'trigger.pulses_in_collection': pulses_in_collection,
                           'trigger.pulses_read': pulses_read,
                           'trigger.pulses_in_events': pulses_in_events,
                           'trigger.mean_event_length': total_event_length / events_built,
                           'trigger.mean_event_occupancy': total_pulse_length / total_event_length,
                           'trigger.config': {k: self.config.get(k, '<<option not present>>')
                                              for k in ['window', 'left_extension', 'right_extension', 'search_window',
                                                        'multiplicity', 'mega_events', 'detector',
                                                        #'user', 'password',   # :-)
                                                        'host', 'port']}}
        if not self.secret_mode:
            self.runs.update_one({'_id': self.config['run_doc_id']},
                                 {'$set': end_of_run_info})
        self.log.info("Event building complete. Information: %s" % end_of_run_info)

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
    do_input_check = False

    def startup(self):
        MongoDBReader.startup(self)
        self.time_of_run_start = int(self.run_doc['start'].timestamp() * units.s)

    def transform_event(self, event_proxy):
        t0, t1 = event_proxy.data  # ns
        self.log.debug("Fetching data for event with range [%s, %s]", pax_to_human_time(t0), pax_to_human_time(t1))

        event = Event(n_channels=self.config['n_channels'],
                      start_time=t0 + self.time_of_run_start,
                      sample_duration=self.sample_duration,
                      stop_time=t1 + self.time_of_run_start,
                      dataset_name=self.run_doc['name'],
                      event_number=event_proxy.event_number)

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
            elif digitizer_id not in self.ignored_channels:
                self.log.warning("Found data from digitizer module %d, channel %d,"
                                 "which doesn't exist according to PMT mapping! Ignoring...",
                                 pulse_doc['module'], pulse_doc['channel'])
                self.ignored_channels.append(digitizer_id)

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


def pax_to_human_time(num):
    """num is in 1s of ns"""
    for x in ['ns', 'us', 'ms', 's', 'ks', 'Ms', 'G', 'T']:
        if num < 1000.0:
            return "%3.3f %s" % (num, x)
        num /= 1000.0
    return "%3.1f %s" % (num, 's')

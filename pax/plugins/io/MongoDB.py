"""Interfacing to MongoDB

MongoDB is used as a data backend within the DAQ.  For example, 'kodiaq', which
reads out the digitizers, will write data to MongoDB.  This data from kodiaq can
either be triggered or untriggered. In the case of untriggered, an event builder
must be run on the data and will result in triggered data.  Input and output
classes are provided for MongoDB access.  More information is in the docstrings.
"""
import time

import numpy as np
import snappy
import pymongo

from pax.datastructure import Event, Pulse, EventProxy
from pax import plugin, trigger, units


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

        # Can we use monary?
        # Only relevant for readuntriggered... code stink
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

        self.input_info['database'] = self.input_info['location'].split('/')[-1]
        if self.input_info['database'] == 'admin':
            self.log.warning("According to the runs db, the data resides in the 'admin' database... "
                             "Guessing 'untriggered' instead.")
            self.input_info['database'] = 'untriggered'
        if self.use_monary:
            self.monary_client = mm.get_database(database_name=self.input_info['database'],
                                                 uri=self.input_info['location'],
                                                 monary=True)

        self.log.debug("Grabbing collection %s" % self.input_info['collection'])
        self.input_collection = mm.get_database(
            database_name=self.input_info['database'],
            uri=self.input_info['location']).get_collection(self.input_info['collection'])
        self.log.debug("Creating index in input collection")
        self.input_collection.create_index(self.sort_key, background=True)
        self.log.debug("Succesfully grabbed collection %s" % self.input_info['collection'])

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
        self.log.info("Starting event builder")

        self.trigger = trigger.Trigger(pax_config=self.processor.config)

    def get_events(self):
        last_time_searched = 0  # ns
        next_event_number = 0

        # Used to timeout if DAQ crashes and no data will come
        time_of_last_daq_response = time.time()
        last_time_to_search = float('inf')
        more_data_coming = True

        while more_data_coming:
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
            search_after = last_time_searched
            self.log.info("Searching for pulses after %s", pax_to_human_time(search_after))

            query = {self.start_key: {'$gt': self._to_mt(search_after),
                                      '$lt': self._to_mt(search_after + self.search_window)}}

            if self.use_monary:
                times, modules, channels = self.do_monary_query(query=query,
                                                                fields=[self.start_key, 'module', 'channel'],
                                                                types=['int64', 'int32', 'int32'],
                                                                **self.mongo_find_options)
                times = times * self.sample_duration

            else:
                time_docs = list(self.input_collection.find(query,
                                                            projection=[self.start_key],
                                                            **self.mongo_find_options))
                # Convert response from list of dictionaries to numpy arrays
                # Pulse times must be converted to pax time units (ns)
                times = np.zeros(len(time_docs), dtype=np.int64)
                channels = np.zeros(len(time_docs), dtype=np.int32)
                modules = np.zeros(len(time_docs), dtype=np.int32)
                for i, doc in enumerate(time_docs):
                    times[i] = self._from_mt(doc[self.start_key])
                    channels[i] = doc['channel']
                    modules[i] = doc['module']

            if len(times):
                self.log.info("Acquired pulse time data in range [%s, %s]",
                              pax_to_human_time(times[0]),
                              pax_to_human_time(times[-1]))

                if not self.data_taking_ended and last_time_searched - times[0] < 0.1 * self.search_window:
                    self.log.info("Most of search window seems empty, probably we're running ahead of DAQ, sleep a sec")
                    time.sleep(1)

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

            # Record advancement of the search window
            if self.data_taking_ended:
                last_time_searched += self.search_window
            else:
                # We can't advance the entire search window, since the DAQ may not have filled it all the way.
                # The condition len(x) == 0 and not self.data_taking_ended has already been dealt with, so ignore
                # your editor's warning that last_start_time may not have been set yet.
                last_time_searched = times[-1]

            more_data_coming = not last_time_searched > last_time_to_search
            if not more_data_coming:
                self.log.info("Searched to %s, which is beyond %s. This is the last batch of data" % (
                    pax_to_human_time(last_time_searched), pax_to_human_time(last_time_to_search)))

            # Send the new data to the trigger
            for data in self.trigger.run(last_time_searched=last_time_searched,
                                         start_times=times,
                                         channels=channels,
                                         modules=modules,
                                         last_data=not more_data_coming):
                self.log.debug("Sending off event %d with data %s" % (next_event_number, data))
                yield EventProxy(event_number=next_event_number, data=data)
                next_event_number += 1

            # Trigger stopped giving event ranges: if more data is coming,
            # we'll stay in loop and get more data for the trigger.

            # Update time of last DAQ reponse
            # This must be done here: building events and getting them all into the queue can take a long time
            # especially since the queue must maintain a fixed size.
            # We don't want to think the DAQ has crashed just because we're using too few CPUS.
            time_of_last_daq_response = time.time()

        # All went well - print out status information
        # Get the end of run info from the trigger, and add the 'trigger.' prefix
        # Also add some MongoDB specific stuff
        end_of_run_info = self.trigger.shutdown()
        end_of_run_info = {'trigger.%s' % k: v for k, v in end_of_run_info.items()}
        end_of_run_info['trigger.ended'] = True
        end_of_run_info['trigger.status'] = 'processed'

        if not self.secret_mode:
            self.runs.update_one({'_id': self.config['run_doc_id']},
                                 {'$set': end_of_run_info})
        self.log.info("Event building complete. Trigger information: %s" % end_of_run_info)


class MongoDBReadUntriggeredFiller(plugin.TransformPlugin, MongoDBReader):

    """Read pulse data from untriggered MongoDB into event ranges provided by trigger MongoDBReadUntriggered
    This is a separate plugin, since reading the raw pulse data is the expensive operation we want to parallelize.
    """
    do_input_check = False

    def startup(self):
        MongoDBReader.startup(self)
        self.time_of_run_start = int(self.run_doc['start'].timestamp() * units.s)

    def transform_event(self, event_proxy):
        (t0, t1), trigger_signals = event_proxy.data  # ns
        self.log.debug("Fetching data for event with range [%s, %s]",
                       pax_to_human_time(t0),
                       pax_to_human_time(t1))

        event = Event(n_channels=self.config['n_channels'],
                      start_time=t0 + self.time_of_run_start,
                      sample_duration=self.sample_duration,
                      stop_time=t1 + self.time_of_run_start,
                      dataset_name=self.run_doc['name'],
                      event_number=event_proxy.event_number,
                      trigger_signals=trigger_signals)

        # Convert trigger signal times to ns since start of event (now in ns since start of run)
        event.trigger_signals['left_time'] -= t0
        event.trigger_signals['right_time'] -= t0
        event.trigger_signals['time_mean'] -= t0

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


def pax_to_human_time(num):
    """num is in 1s of ns"""
    for x in ['ns', 'us', 'ms', 's', 'ks', 'Ms', 'G', 'T']:
        if num < 1000.0:
            return "%3.3f %s" % (num, x)
        num /= 1000.0
    return "%3.1f %s" % (num, 's')

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
        self.input_database = mm.get_database(database_name=self.input_info['database'],
                                              uri=self.input_info['location'])
        self.input_collection = self.input_database.get_collection(self.input_info['collection'])
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
        self.log.debug("Retrieving run doc")
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
        self.batch_window = self.config['batch_window']

        # Initialize the buffer to hold the pulse ranges
        self.pulse_ranges_buffer = np.ones((self.config.get('pulse_ranges_buffer_size', int(1e7)), 2),
                                           dtype=np.int64) * -1
        self.log.info("Starting event builder")

        # Initialize the trigger
        # For now, make a collection in trigger_monitor on the same eb as the untriggered collection
        if not self.secret_mode:
            self.uri_for_monitor = self.input_info['location'].replace('untriggered', 'trigger_monitor')
            trig_mon_db = self.processor.mongo_manager.get_database('trigger_monitor', uri=self.uri_for_monitor)
            trig_mon_coll = trig_mon_db.get_collection(self.run_doc['name'])
        else:
            trig_mon_coll = None
            self.uri_for_monitor = 'nowhere, because secret mode was used'
        self.trigger = trigger.Trigger(pax_config=self.processor.config,
                                       trigger_monitor_collection=trig_mon_coll)

    def get_last_pulse_time(self):
        """Returns time (in pax units, i.e. ns) at which the pulse which starts last in the run stops
        It would have been nicer to know the last stop time, but pulses are sorted by start time.
        """
        cu = self.input_collection.find().sort(self.start_key, direction=pymongo.DESCENDING).limit(1)
        cu = list(cu)
        if not len(cu):
            return 0

        value = cu[0]
        if self.stop_key in value:
            value = value[self.stop_key]
        else:
            value = value[self.start_key]

        return self._from_mt(value)

    def get_events(self):
        self.refresh_run_doc()

        last_time_searched = 0  # ns
        last_time_to_search = None
        next_event_number = 0

        more_data_coming = True

        if self.data_taking_ended:
            last_pulse_time = self.get_last_pulse_time()
            self.log.info("The DAQ has stopped, last pulse time is %s" % pax_to_human_time(last_pulse_time))
            # Does this correspond roughly to the run end time? If not, warn, DAQ may have crashed.
            end_of_run_t = (self.run_doc['end'].timestamp() -
                            self.run_doc['start'].timestamp()) * units.s
            if end_of_run_t > last_pulse_time + 60 * units.s:
                self.log.warning("Run is %s long according to run db, but last pulse in collection starts at %s. "
                                 "Did the DAQ crash?" % (pax_to_human_time(end_of_run_t),
                                                         pax_to_human_time(last_pulse_time)))

            # Add 1 batch window margin (in case the pulse is long, or another pulse started sooner but ends later)
            last_time_to_search = last_pulse_time + self.batch_window

        while more_data_coming:

            if not self.data_taking_ended:
                # Update the run document, so we know if the run ended.
                # This must happen before querying for more data, to avoid a race condition where the run ends
                # while the query is in process.
                self.refresh_run_doc()
                last_pulse_time = self.get_last_pulse_time()
                if self.data_taking_ended:
                    # Data taking just ended, we can define the time we should stop searching.
                    last_time_to_search = last_pulse_time + self.batch_window
                else:
                    # Make sure we only query data that is at least one batch window away from the last pulse time,
                    # to avoid querying near the ragged edge (where readers are inserting asynchronously).
                    # Also make sure we only query once a full batch window of such safe data is available (to avoid
                    # mini-queries). So if batch_window = 60 sec, we'll wait for 2 mins of data before our first query.
                    if last_pulse_time - 2 * self.batch_window < last_time_searched:
                        self.log.info("DAQ has not taken sufficient data to continue. Sleeping 5 sec...")
                        time.sleep(5)

            # Query for pulse start & stop times within a large batch window
            self.log.info("Searching for pulses after %s", pax_to_human_time(last_time_searched))

            query = {self.start_key: {'$gt': self._to_mt(last_time_searched),
                                      '$lt': self._to_mt(last_time_searched + self.batch_window)}}

            if self.use_monary:
                if self.config['can_get_area']:
                    times, modules, channels, areas = self.do_monary_query(
                        query=query,
                        fields=[self.start_key, 'module', 'channel', 'integral'],
                        types=['int64', 'int32', 'int32', 'float64'],
                        **self.mongo_find_options
                    )
                else:
                    times, modules, channels = self.do_monary_query(
                        query=query,
                        fields=[self.start_key, 'module', 'channel'],
                        types=['int64', 'int32', 'int32'],
                        **self.mongo_find_options
                    )
                    areas = np.zeros(len(times), dtype=np.float64)
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
                areas = np.zeros(len(time_docs), dtype=np.int32)
                can_get_area = self.config['can_get_area']
                for i, doc in enumerate(time_docs):
                    times[i] = self._from_mt(doc[self.start_key])
                    channels[i] = doc['channel']
                    modules[i] = doc['module']
                    if can_get_area:
                        areas[i] = doc['area']

            if len(times):
                self.log.info("Acquired pulse time data in range [%s, %s]",
                              pax_to_human_time(times[0]),
                              pax_to_human_time(times[-1]))
            else:
                self.log.info("No pulse data found.")

            # Record advancement of the batch window
            # We've ensured above that a full batch window is always queried, even in live mode.
            last_time_searched += self.batch_window

            # Check if more data is coming
            more_data_coming = (not self.data_taking_ended) or (last_time_searched < last_time_to_search)
            if not more_data_coming:
                self.log.info("Searched to %s, which is beyond %s. This is the last batch of data" % (
                    pax_to_human_time(last_time_searched), pax_to_human_time(last_time_to_search)))

            # Check if we've passed the user-specified stop (if so configured)
            if last_time_searched > self.config.get('stop_after_sec', float('inf')) * units.s:
                self.log.warning("Searched to %s, which is beyond the user-specified stop at %d sec."
                                 "This is the last batch of data" % (last_time_searched,
                                                                     self.config['stop_after_sec']))
                more_data_coming = False

            # Check if the collection has grown beyond the burn threshold
            max_size_gb = self.config.get('burn_data_if_collection_exceeds_gb', float('inf'))
            if max_size_gb != float('inf'):
                coll_size_gb = self.input_database.command("collstats", self.run_doc['name'])['size'] / 1e9
                if coll_size_gb > max_size_gb:
                    self.log.critical("Untriggered collection is %0.2f GB large, but you limited it to %0.2f GB. "
                                      "ALL DATA from this collection will now be DELETED!!!" % (coll_size_gb,
                                                                                                max_size_gb))
                    # From the Mongo docs: "To remove all documents from a collection, it may be more efficient to
                    # drop the entire collection, including the indexes, and then recreate the collection
                    # and rebuild the indexes"
                    self.input_database.drop_collection(self.run_doc['name'])
                    self.input_collection = self.input_database.get_collection(self.run_doc['name'])
                    self.input_collection.create_index(self.sort_key, background=True)

                    # Jump last_time_searched to just before next pulse which is inserted
                    # Without this we waste precious moments wading through dead time
                    while True:
                        last_pulse_time = self.get_last_pulse_time()
                        if last_pulse_time != 0:
                            break
                        self.log.info("Sleeping one second, waiting for DAQ to insert a new pulse after the deletion")
                        time.sleep(1)
                    self.last_time_searched = last_pulse_time - self.batch_window

            # Send the new data to the trigger
            for data in self.trigger.run(last_time_searched=last_time_searched,
                                         start_times=times,
                                         channels=channels,
                                         modules=modules,
                                         areas=areas,
                                         last_data=not more_data_coming):
                yield EventProxy(event_number=next_event_number, data=data)
                next_event_number += 1

            # Trigger stopped giving event ranges; if more data is coming,
            # we'll stay in loop and get more data for the trigger.

        # All went well - print out status information
        # Get the end of run info from the trigger, and add the 'trigger.' prefix
        # Also add some MongoDB specific stuff
        end_of_run_info = self.trigger.shutdown()
        end_of_run_info = {'trigger.%s' % k: v for k, v in end_of_run_info.items()}
        end_of_run_info['trigger.ended'] = True
        end_of_run_info['trigger.status'] = 'processed'
        end_of_run_info['trigger.trigger_monitor_data_location'] = self.uri_for_monitor

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

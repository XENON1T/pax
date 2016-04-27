"""Interfacing to MongoDB

MongoDB is used as a data backend within the DAQ.  For example, 'kodiaq', which
reads out the digitizers, will write data to MongoDB.  This data from kodiaq can
either be triggered or untriggered. In the case of untriggered, an event builder
must be run on the data and will result in triggered data.  Input and output
classes are provided for MongoDB access.  More information is in the docstrings.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import snappy
import pymongo

from pax.datastructure import Event, Pulse, EventProxy
from pax import plugin, trigger, units


class MongoDBReader:

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

        self.log.debug("Grabbing collection %s" % self.input_info['collection'])
        self.input_database = mm.get_database(database_name=self.input_info['database'],
                                              uri=self.input_info['location'])
        self.input_collection = self.input_database.get_collection(self.input_info['collection'])
        self.log.debug("Creating index in input collection")
        self.input_collection.create_index(self.sort_key, background=True)
        self.log.debug("Succesfully grabbed collection %s" % self.input_info['collection'])

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
        self.max_query_workers = self.config.get('max_query_workers')
        self.last_pulse_time = 0  # time (in pax units, i.e. ns) at which the pulse which starts last in the run stops
        # It would have been nicer to know the last stop time, but pulses are sorted by start time...

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

    def refresh_run_info(self):
        """Refreshes the run doc and last pulse time information.
        """
        self.refresh_run_doc()

        # Get the last pulse time
        cu = self.input_collection.find().sort(self.start_key, direction=pymongo.DESCENDING).limit(1)
        cu = list(cu)
        if not len(cu):
            # No pulses in collection yet?
            last_pulse_time = 0
        else:
            last_pulse_time = cu[0][self.start_key]
        self.last_pulse_time = self._from_mt(last_pulse_time)

        if self.data_taking_ended:
            self.log.info("The DAQ has stopped, last pulse time is %s" % pax_to_human_time(self.last_pulse_time))
            # Does this correspond roughly to the run end time? If not, warn, DAQ may have crashed.
            end_of_run_t = (self.run_doc['end'].timestamp() -
                            self.run_doc['start'].timestamp()) * units.s
            if end_of_run_t > self.last_pulse_time + 60 * units.s:
                self.log.warning("Run is %s long according to run db, but last pulse in collection starts at %s. "
                                 "Did the DAQ crash?" % (pax_to_human_time(end_of_run_t),
                                                         pax_to_human_time(self.last_pulse_time)))

    def get_events(self):
        self.refresh_run_info()
        last_time_searched = 0  # ns
        next_event_number = 0
        more_data_coming = True

        # Do we know when to stop searching?
        if self.data_taking_ended:
            # Add 1 batch window margin (in case the pulse is long, or another pulse started sooner but ends later)
            end_of_search_for_this_run = self.last_pulse_time + self.batch_window
        else:
            end_of_search_for_this_run = float('inf')

        while more_data_coming:
            # Refresh the run info, to find out if data taking has ended
            if not self.data_taking_ended:
                self.refresh_run_info()

            # What is the last time we need to search?
            if self.data_taking_ended:
                end_of_search_for_this_run = self.last_pulse_time + self.batch_window
            else:
                end_of_search_for_this_run = float('inf')

            # What is the earliest time we still need to search?
            next_time_to_search = last_time_searched
            if next_time_to_search != 0:
                next_time_to_search += self.batch_window * self.config['skip_ahead']

            # How many batch windows can we search now?
            if self.data_taking_ended:
                batches_to_search = int((end_of_search_for_this_run - next_time_to_search) / self.batch_window) + 1
            else:
                # Make sure we only query data that is at least one batch window away from the last pulse time,
                # to avoid querying near the ragged edge (where readers are inserting asynchronously).
                # Also make sure we only query once a full batch window of such safe data is available (to avoid
                # mini-queries). So if batch_window = 60 sec, we'll wait for 2 mins of data before our first query.
                batches_to_search = int((self.last_pulse_time - next_time_to_search) / self.batch_window) - 1
                if batches_to_search < 1:
                    self.log.info("DAQ has not taken sufficient data to continue. Sleeping 5 sec...")
                    time.sleep(5)
                    continue
            batches_to_search = min(batches_to_search, self.max_query_workers)

            with ThreadPoolExecutor(max_workers=batches_to_search) as executor:

                # Start the queries in separate processes
                futures = []
                for batch_i in range(batches_to_search):
                    monary_client = self.processor.mongo_manager.get_database(database_name=self.input_info['database'],
                                                                              uri=self.input_info['location'],
                                                                              monary=True)
                    start = next_time_to_search + batch_i * self.batch_window
                    stop = start + self.batch_window
                    self.log.info("Submitting query for batch %d, time range [%s, %s)" % (
                        batch_i, pax_to_human_time(start), pax_to_human_time(stop)))
                    future = executor.submit(get_pulses,
                                             monary_client=monary_client,
                                             run_name=self.run_doc['run_name'],
                                             start_mongo_time=self._to_mt(start),
                                             stop_mongo_time=self._to_mt(stop),
                                             get_area=self.config['can_get_area'])
                    futures.append(future)

                # Record advancement of the batch window
                last_time_searched = next_time_to_search + batches_to_search * self.batch_window

                # Check if more data is coming
                more_data_coming = (not self.data_taking_ended) or (last_time_searched < end_of_search_for_this_run)
                if not more_data_coming:
                    self.log.info("Searched to %s, which is beyond %s. This is the last batch of data" % (
                        pax_to_human_time(last_time_searched), pax_to_human_time(end_of_search_for_this_run)))

                # Check if we've passed the user-specified stop (if so configured)
                if last_time_searched > self.config.get('stop_after_sec', float('inf')) * units.s:
                    self.log.warning("Searched to %s, which is beyond the user-specified stop at %d sec."
                                     "This is the last batch of data" % (last_time_searched,
                                                                         self.config['stop_after_sec']))
                    more_data_coming = False

                # Retrieve results from the queries, then build events (which must happen serially).
                for i, future in enumerate(futures):
                    times, modules, channels, areas = future.result()
                    times = times * self.sample_duration

                    if len(times):
                        self.log.info("Batch %d: acquired pulses in range [%s, %s]",
                                      pax_to_human_time(times[0]),
                                      pax_to_human_time(times[-1]))
                    else:
                        self.log.info("Batch %d: No pulse data found.")

                    # Send the new data to the trigger, which will build events from it
                    for data in self.trigger.run(last_time_searched=next_time_to_search + (i + 1) * self.batch_window,
                                                 start_times=times,
                                                 channels=channels,
                                                 modules=modules,
                                                 areas=areas,
                                                 last_data=(not more_data_coming and i == len(futures) - 1)):
                        yield EventProxy(event_number=next_event_number, data=data)
                        next_event_number += 1

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


def get_pulses(monary_client, run_name, start_mongo_time, stop_mongo_time, get_area=False):
    results = monary_client.query(database='untriggered',
                                  collection=run_name,
                                  query={'time': {'$get': start_mongo_time,
                                                  '$lt': stop_mongo_time}},
                                  fields=['time', 'module', 'channel'] + (['integral'] if get_area else []),
                                  types=['int64', 'int32', 'int32'] + (['area'] if get_area else []),
                                  select_fields=True)
    monary_client.close()

    if get_area:
        times, modules, channels, areas = results
    else:
        times, modules, channels = results
        areas = np.zeros(len(times), dtype=np.float64)

    # Sort ourselves, to spare Mongo the trouble
    sort_order = np.argsort(times)
    times = times[sort_order]
    modules = modules[sort_order]
    channels = channels[sort_order]
    return times, modules, channels, areas
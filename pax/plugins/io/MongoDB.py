"""Interfacing to MongoDB

MongoDB is used as a data backend within the DAQ.  For example, 'kodiaq', which
reads out the digitizers, will write data to MongoDB.  This data from kodiaq can
either be triggered or untriggered. In the case of untriggered, an event builder
must be run on the data and will result in triggered data.  Input and output
classes are provided for MongoDB access.  More information is in the docstrings.
"""
import time
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from monary import Monary
import numpy as np
import snappy
import pymongo

from pax.datastructure import Event, Pulse, EventProxy
from pax import plugin, trigger, units


class MongoBase:

    def startup(self):
        self.sample_duration = self.config['sample_duration']
        self.secret_mode = self.config['secret_mode']

        # Setup keys and options for finding pulses
        self.start_key = self.config['start_key']
        self.stop_key = self.config['stop_key']
        self.sort_key = ([(self.start_key, 1), (self.stop_key, 1)])
        self.mongo_find_options = {'sort': self.sort_key}

        # Connect to the runs db
        self.cm = ClientMaker(self.processor.config['MongoDB'])
        self.run_client = self.cm.get_client('run')
        self.runs = self.run_client['run'].get_collection('runs_new')
        self.refresh_run_doc()

        # Connect to the input database
        # Input database connection settings are specified in the run doc
        # TODO: can username, host, password, settings differer from run db access info? Let's hope not...
        # If this case does occur, we have to pass extra args to the ClientMaker
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
        self.input_client = self.cm.get_client(database_name=self.input_info['database'],
                                               uri=self.input_info['location'],
                                               w=0)         # w=0 ensures fast deletes. We're not going to write.
        self.input_db = self.input_client[self.input_info['database']]
        self.input_collection = self.input_db.get_collection(self.input_info['collection'])

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


class MongoDBReadUntriggered(plugin.InputPlugin, MongoBase):

    """Read from MongoDB and build events
    This will perform a sliding window trigger on the pulse midpoint times.
    No PMT pulse data is read in this class to ensure speed.
    """
    use_monary = True
    do_output_check = False

    def startup(self):
        MongoBase.startup(self)

        # Load a few more config settings
        self.detector = self.config['detector']
        self.batch_window = self.config['batch_window']
        self.max_query_workers = self.config['max_query_workers']
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
            trig_mon_db = self.cm.get_client('trigger_monitor',
                                             uri=self.uri_for_monitor)['trigger_monitor']
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
                # Make sure we only query data that is edge_safety_margin away from the last pulse time.
                # This is because the readers are inserting the pulse data slightly asynchronously.
                # Also make sure we only query once a full batch window of such safe data is available (to avoid
                # mini-queries).
                duration_of_searchable = self.last_pulse_time - self.config['edge_safety_margin'] - next_time_to_search
                batches_to_search = int(duration_of_searchable / self.batch_window)
                if batches_to_search < 1:
                    self.log.info("DAQ has not taken sufficient data to continue. Sleeping 5 sec...")
                    time.sleep(5)
                    continue
            batches_to_search = min(batches_to_search, self.max_query_workers)

            # Start new queries in separate processes
            with ThreadPoolExecutor(max_workers=batches_to_search) as executor:
                futures = []
                for batch_i in range(batches_to_search):
                    start = next_time_to_search + batch_i * self.batch_window
                    stop = start + self.batch_window
                    self.log.info("Submitting query for batch %d, time range [%s, %s)" % (
                        batch_i, pax_to_human_time(start), pax_to_human_time(stop)))
                    future = executor.submit(get_pulses,
                                             client_maker_config=self.cm.config,
                                             input_info=self.input_info,
                                             run_name=self.run_doc['name'],
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
                stop_after_sec = self.config.get('stop_after_sec', None)
                if stop_after_sec and 0 < stop_after_sec < float('inf'):
                    if last_time_searched > stop_after_sec * units.s:
                        self.log.warning("Searched to %s, which is beyond the user-specified stop at %d sec."
                                         "This is the last batch of data" % (last_time_searched,
                                                                             self.config['stop_after_sec']))
                        more_data_coming = False

                # Retrieve results from the queries, then build events (which must happen serially).
                for i, future in enumerate(futures):
                    times, modules, channels, areas = future.result()
                    times = times * self.sample_duration

                    if len(times):
                        self.log.info("Batch %d: acquired pulses in range [%s, %s]" % (
                                      i,
                                      pax_to_human_time(times[0]),
                                      pax_to_human_time(times[-1])))
                    else:
                        self.log.info("Batch %d: No pulse data found." % i)

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
        end_of_run_info['trigger.mongo_reader_config'] = {k: v for k, v in self.config.items()
                                                          if k != 'password' and
                                                          k not in self.processor.config['DEFAULT']}

        if not self.secret_mode:
            self.runs.update_one({'_id': self.config['run_doc_id']},
                                 {'$set': end_of_run_info})
        self.log.info("Event building complete. Trigger information: %s" % end_of_run_info)


class MongoDBReadUntriggeredFiller(plugin.TransformPlugin, MongoBase):

    """Read pulse data from untriggered MongoDB into event ranges provided by trigger MongoDBReadUntriggered
    This is a separate plugin, since reading the raw pulse data is the expensive operation we want to parallelize.
    """
    do_input_check = False

    def startup(self):
        MongoBase.startup(self)
        self.ignored_channels = []
        self.time_of_run_start = int(self.run_doc['start'].timestamp() * units.s)

        # Load the digitizer channel -> PMT index mapping
        self.detector = self.config['detector']
        self.pmts = self.config['pmts']
        self.pmt_mappings = {(x['digitizer']['module'],
                              x['digitizer']['channel']): x['pmt_position'] for x in self.pmts}

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


class MongoDBClearUntriggered(plugin.TransformPlugin, MongoBase):

    """Keeps track of which time is safe to delete, then deletes data from the collection in batches.
    At shutdown, drop the collection.

    Will do NOTHING unless delete_data = True in config.

    This must run as part of the output group, so it gets the events in order.
    """
    do_input_check = False
    do_output_check = False

    def startup(self):
        MongoBase.startup(self)
        self.time_of_run_start = int(self.run_doc['start'].timestamp() * units.s)
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_query_workers'])
        self.last_time_deleted = 0

    def transform_event(self, event_proxy):
        if not self.config['delete_data']:
            return event_proxy

        time_since_start = event_proxy.data['stop_time'] - self.time_of_run_start
        if time_since_start > self.last_time_deleted + self.config['batch_window']:
            self.log.info("Seen event at %s, clearing all data until then." % (
                pax_to_human_time(time_since_start), pax_to_human_time(time_since_start)))
            self.executor.submit(delete_pulses,
                                 self.input_collection,
                                 start_mongo_time=self._to_mt(self.last_time_deleted),
                                 stop_mongo_time=self._to_mt(time_since_start))
            self.last_time_deleted = time_since_start

        return event_proxy

    def shutdown(self):
        if not self.config['delete_data']:
            return

        self.log.info("Shutting down delete executor / waiting for deletes")
        self.executor.shutdown(wait=True)

        self.log.info("Dropping data collection")
        self.input_db.drop_collection(self.run_doc['name'])
        self.log.info("Collection has been dropped")


def pax_to_human_time(num):
    """Converts a pax time to a human-readable representation"""
    for x in ['ns', 'us', 'ms', 's', 'ks', 'Ms', 'G', 'T']:
        if num < 1000.0:
            return "%3.3f %s" % (num, x)
        num /= 1000.0
    return "%3.1f %s" % (num, 's')


def get_pulses(client_maker_config, input_info, run_name,
               start_mongo_time, stop_mongo_time,
               get_area=False,):
    """Query pulses between start_mongo_time and stop_mongo_time (in mongo time units)
    Uses, then closes the monary_client to do the query.

    Returns four numpy arrays: times, modules, channels, areas.
    Areas consists of zeros unless get_area = True, in which we also fetch 'integral' from MongoDB.
    """
    query = {'time': {'$gte': start_mongo_time,
                      '$lt': stop_mongo_time}}

    client_maker = ClientMaker(client_maker_config)

    monary_client = client_maker.get_client(database_name=input_info['database'],
                                            uri=input_info['location'],
                                            monary=True)
    fields = ['time', 'module', 'channel'] + (['integral'] if get_area else [])
    types = ['int64', 'int32', 'int32'] + (['area'] if get_area else [])
    results = list(monary_client.block_query('untriggered', run_name, query, fields, types,
                                             block_size=int(1e7),
                                             select_fields=True))
    monary_client.close()

    if not len(results) or not len(results[0]):
        times = np.zeros(0, dtype=np.int64)
        modules = np.zeros(0, dtype=np.int32)
        channels = np.zeros(0, dtype=np.int32)
        areas = np.zeros(0, dtype=np.float64)

    else:
        # Concatenate results from multiple blocks, in case multiple blocks were needed
        results = [np.concatenate([results[i][j]
                                   for i in range(len(results))])
                   for j in range(len(results[0]))]

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
        areas = areas[sort_order]

        # modules = np.zeros(len(times), dtype=np.int32)
        # channels = np.zeros(len(times), dtype=np.int32)
        # areas = np.zeros(len(times), dtype=np.float64)

    return times, modules, channels, areas


def delete_pulses(collection, start_mongo_time, stop_mongo_time):
    """Deletes pulse between start_mongo_time (inclusive) and stop_mongo_time (exclusive), both in mongo time units.
    """
    query = {'time': {'$gte': start_mongo_time,
                      '$lt': stop_mongo_time}}
    collection.delete_many(query)


class ClientMaker:
    """Helper class to create MongoDB clients

    On __init__, you can specify options that will be used to format mongodb uri's,
    in particular user, password, host and port.
    """
    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger('Mongo client maker')

    def get_client(self, database_name=None, uri=None, monary=False, **kwargs):
        """Get a Mongoclient. Returns Mongo database object.
        If you provide a mongodb connection string uri, we will insert user & password into it,
        otherwise one will be built from the configuration settings.
        If database_name=None, will connect to the default database of the uri. database=something
        overrides event the uri's specification of a database.
        """
        # Pattern of URI's we expect from database (without user & pass)
        uri_pattern = r'mongodb://([^:]+):(\d+)/(\w+)'

        # Format of URI we should eventually send to mongo
        full_uri_format = 'mongodb://{user}:{password}@{host}:{port}/{database}'

        if uri is None:
            # Construct the entire URI from default settings
            uri = full_uri_format.format(database=database_name, **self.config)
        else:
            m = re.match(uri_pattern, uri)
            if m:
                # URI was provided, but without user & pass.
                host, port, _database_name = m.groups()
                if database_name is None:
                    database_name = _database_name
                uri = full_uri_format.format(database=database_name, host=host, port=port,
                                             user=self.config['user'], password=self.config['password'])
            else:
                # Some other URI was provided. Maybe works...
                self.log.warning("Unexpected Mongo URI %s, expected format %s. Trying anyway..." % (uri, uri_pattern))

        if monary:
            # Monary clients are not cached
            self.log.debug("Connecting to Mongo via monary using uri %s" % uri)
            return Monary(uri, **kwargs)

        else:
            self.log.debug("Connecting to Mongo using uri %s" % uri)
            client = pymongo.MongoClient(uri, **kwargs)
            client.admin.command('ping')        # raises pymongo.errors.ConnectionFailure on failure
            self.log.debug("Succesfully pinged client")
            return client

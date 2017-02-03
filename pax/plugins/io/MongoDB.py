"""Interfacing to MongoDB

MongoDB is used as a data backend within the DAQ.  For example, 'kodiaq', which
reads out the digitizers, will write data to MongoDB.  This data from kodiaq can
either be triggered or untriggered. In the case of untriggered, an event builder
must be run on the data and will result in triggered data.  Input and output
classes are provided for MongoDB access.  More information is in the docstrings.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
import datetime
import time

import pytz
import numpy as np
import pymongo
import snappy
import pickle
import monary

from pax.MongoDB_ClientMaker import ClientMaker, parse_passwordless_uri
from pax.datastructure import Event, Pulse, EventProxy
from pax import plugin, trigger, units, exceptions


class MongoBase:

    _cached_subcollection_handles = {}

    def startup(self):
        self.sample_duration = self.config['sample_duration']
        self.secret_mode = self.config['secret_mode']

        # Connect to the runs db
        self.cm = ClientMaker(self.processor.config['MongoDB'])
        self.run_client = self.cm.get_client('run', autoreconnect=True)
        self.runs_collection = self.run_client['run']['runs_new']
        self.refresh_run_doc()

        self.split_collections = self.run_doc['reader']['ini'].get('rotating_collections', 0)

        if self.split_collections:
            self.batch_window = int(self.sample_duration * (2 ** 31))
            self.log.debug("Split collection mode: batch window forced to %s sec" % (self.batch_window / units.s))
        else:
            self.batch_window = self.config['batch_window']

        # Retrieve information for connecting to the input database from the run doc
        self.input_info = None
        for doc in self.run_doc['data']:
            if doc['type'] == 'untriggered':
                self.input_info = doc
                break
        else:
            raise ValueError("Invalid run document: none of the 'data' entries contain untriggered data!")

        if ';' in self.input_info['location']:
            self.split_hosts = True
            self.input_info['location'] = self.input_info['location'].split(';')[0]
        else:
            self.split_hosts = False

        self.input_info['database'] = self.input_info['location'].split('/')[-1]
        if not self.input_info['database'] == 'untriggered' and self.config['detector'] == 'tpc':
            raise ValueError("TPC data is expected in the 'untriggered' database,"
                             " but this run is in %s?!" % self.input_info['database'])

        start_datetime = self.run_doc['start'].replace(tzinfo=pytz.utc).timestamp()
        self.time_of_run_start = int(start_datetime * units.s)

        # Connect to the input database on the primary host
        self.input_client = self.cm.get_client(database_name=self.input_info['database'],
                                               uri=self.input_info['location'],
                                               w=0)         # w=0 ensures fast deletes. We're not going to write.
        self.input_db = self.input_client[self.input_info['database']]

        if not self.split_collections:
            self.input_collection = self.input_db.get_collection(self.input_info['collection'])
        # In split collections mode, we use the subcollection methods (see below) to get the input collections

        if self.split_hosts:
            self.hosts = [parse_passwordless_uri(x)[0]
                          for x in set(self.run_doc['reader']['ini']['mongo']['hosts'].values())]
        else:
            self.hosts = [parse_passwordless_uri(self.input_info['location'])[0]]

        # Make pymongo db handles for all hosts. Double work if not split_hosts, but avoids double code later
        self.dbs = [self.cm.get_client(database_name=self.input_info['database'],
                                       host=host,
                                       uri=self.input_info['location'],
                                       w=0)[self.input_info['database']] for host in self.hosts]

        # Get the database in which the acquisition monitor data resides.
        if not self.split_hosts:
            # If we haven't split hosts, just take the one host we have.
            self.aqm_db = self.dbs[0]
        else:
            aqm_host = self.config.get('acquisition_monitor_host', 'eb0')
            db_i = self.hosts.index(aqm_host)
            self.aqm_db = self.dbs[db_i]

        if not self.split_collections:
            self.input_collections = [db.get_collection(self.input_info['collection']) for db in self.dbs]

    def refresh_run_doc(self):
        """Update the internal run doc within this class
        (does not change anything in the runs database)

        This is useful for example checking if a run has ended.
        """
        self.log.debug("Retrieving run doc")
        self.run_doc = self.runs_collection.find_one({'_id': self.config['run_doc_id']})
        self.log.debug("Run doc retrieved")
        self.data_taking_ended = 'end' in self.run_doc

    def subcollection_name(self, number):
        """Return name of subcollection number in the run"""
        assert self.split_collections
        return '%s_%s' % (self.run_doc['name'], number)

    def subcollection(self, number, host_i=None):
        """Return pymongo collection object for subcollection number in the run
        Caches subcollection handles for you, since it seems to take time to ask for the collection
        every event
        Actually this turned out to be some other bug... probably we can remove collection caching now.
        """
        if host_i is None:
            db = self.input_db
        else:
            db = self.dbs[host_i]
        assert self.split_collections
        cache_key = (number, host_i)
        if cache_key in self._cached_subcollection_handles:
            return self._cached_subcollection_handles[cache_key]
        else:
            coll = db.get_collection(self.subcollection_name(number))
            self._cached_subcollection_handles[cache_key] = coll
            return coll

    def subcollection_with_time(self, time):
        """Returns the number of the subcollection which contains pulses which start at time
        time: pax units (ns) since start of run
        """
        assert self.split_collections
        return int(time / self.batch_window)

    def time_range_query(self, start=None, stop=None):
        """Returns Mongo query to find pulses that START in [start, stop)
        Start and stop are each specified in pax units since start of the run.
        """
        return {'time': {'$gte': self._to_mt(start),
                         '$lt': self._to_mt(stop)}}

    def _to_mt(self, x):
        """Converts the time x from pax units to mongo units"""
        return int(x // self.sample_duration)

    def _from_mt(self, x):
        """Converts the time x from mongo units to pax units"""
        return int(x * self.sample_duration)


class MongoDBReadUntriggered(plugin.InputPlugin, MongoBase):

    """Read pulse times from MongoDB, pass them to the trigger,
    and send off EventProxy's for MongoDBReadUntriggeredFiller.
    """
    do_output_check = False
    latest_subcollection = 0           # Last subcollection that was found to contain some data, last time we checked

    def startup(self):
        self.log.info("Eventbuilder input starting up")
        MongoBase.startup(self)
        self.detector = self.config['detector']
        self.max_query_workers = self.config['max_query_workers']
        self.last_pulse_time = 0  # time (in pax units, i.e. ns) at which the pulse which starts last in the run stops
        # It would have been nicer to simply know the last stop time, but pulses are sorted by start time...

        # Initialize the trigger
        # For now, make a collection in trigger_monitor on the same eb as the untriggered collection
        if not self.secret_mode:
            self.uri_for_monitor = self.config['trigger_monitor_mongo_uri']
            trig_mon_db = self.cm.get_client('trigger_monitor', uri=self.uri_for_monitor)['trigger_monitor']
            trig_mon_coll = trig_mon_db.get_collection(self.run_doc['name'])
        else:
            trig_mon_coll = None
            self.uri_for_monitor = 'nowhere, because secret mode was used'

        self.log.info("Trigger starting up")
        self.trigger = trigger.Trigger(pax_config=self.processor.config,
                                       trigger_monitor_collection=trig_mon_coll)
        self.log.info("Trigger startup successful")

        # For starting event building in the middle of a run:
        self.initial_start_time = self.config.get('start_after_sec', 0) * units.s
        if self.initial_start_time:
            self.latest_subcollection = self.initial_start_time // self.batch_window
            self.log.info("Starting at %0.1f sec, subcollection %d" % (self.initial_start_time,
                                                                       self.latest_subcollection))

        self.pipeline_status_collection = self.run_client['run'][self.config.get('pipeline_status_collection_name',
                                                                                 'pipeline_status')]
        self.log.info("Eventbuilder input startup successful")

    def refresh_run_info(self):
        """Refreshes the run doc and last pulse time information.
        Also updates the pipeline status info with the current queue length
        """
        self.refresh_run_doc()

        # Find the last collection with data in it
        self.log.debug("Finding last collection")
        if self.split_collections:
            if self.data_taking_ended:
                # Get all collection names, find the last subcollection with some data that belongs to the current run.
                subcols_with_stuff = [int(x.split('_')[-1]) for x in self.input_db.collection_names()
                                      if x.startswith(self.run_doc['name']) and
                                      self.input_db.get_collection(x).count()]
                if not len(subcols_with_stuff):
                    self.log.error("Run contains no collection(s) with any pulses!")
                    self.last_pulse_time = 0
                    # This should only happen at the beginning of a run, otherwise something is very wrong with the
                    # collection clearing algorithm!
                    assert self.latest_subcollection == 0
                    return
                else:
                    self.latest_subcollection = max(subcols_with_stuff)
                check_collection = self.subcollection(self.latest_subcollection)
            else:
                # While the DAQ is running, we can't use this method, as the reader creates empty collections
                # ahead of the insertion point.
                if self.config.get('use_run_status_doc'):
                    # Dan made a doc with the approximate insertion point of each digitizer: the min of these should
                    # be safe to use (more or less.. a slight delay is still advisable. ask Dan for details)
                    status_doc = self.input_db.get_collection('status').find_one({'collection': self.run_doc['name']})
                    if status_doc is None:
                        raise RuntimeError("Missing run status doc!")
                    safe_col = float('inf')
                    for k, v in status_doc:
                        if isinstance(v, int):
                            safe_col = min(v, safe_col)
                    safe_col -= 1
                    if safe_col < 0 or safe_col == float('inf'):
                        self.log.info("No subcollection is safe for triggering yet")
                        self.last_pulse_time = 0
                        return
                    self.latest_subcollection = safe_col
                    self.log.info("First safe subcollection is %d" % self.latest_subcollection)
                else:
                    # Old method: find the last collection with some data, rely on large safety margin
                    # Keep fingers crossed. Instead, move forward in subcollections until we find one without data.
                    # If there is a large gap in the data, we won't progress beyond it until the run ends.
                    while True:
                        if not self.subcollection(self.latest_subcollection + 1).count():
                            break
                        self.latest_subcollection += 1
                    self.log.info("Last subcollection with data is %d" % self.latest_subcollection)

                check_collection = self.subcollection(self.latest_subcollection)
        else:
            check_collection = self.input_collection

        # Find the last pulse in the collection
        cu = list(check_collection.find().sort('time', direction=pymongo.DESCENDING).limit(1))
        if not len(cu):
            if self.split_collections:
                if not self.latest_subcollection == 0:
                    self.log.warning("Latest subcollection %d seems empty now, but wasn't before... Race condition/edge"
                                     " case in mongodb, bug in clearing code, or something else weird? Investigate if "
                                     "this occurs often!!" % self.latest_subcollection)
                last_pulse_time = self.latest_subcollection * self.batch_window
            else:
                # Apparently the DAQ has not taken any pulses yet?
                last_pulse_time = 0
        else:
            last_pulse_time = cu[0]['time']

        self.last_pulse_time = self._from_mt(last_pulse_time)

        if self.data_taking_ended:
            self.log.info("The DAQ has stopped, last pulse time is %s" % pax_to_human_time(self.last_pulse_time))
            if self.split_collections:
                self.log.info("The last subcollection number is %d" % self.latest_subcollection)

            # Does this correspond roughly to the run end time? If not, warn, DAQ may have crashed.
            end_of_run_t = (self.run_doc['end'].timestamp() - self.run_doc['start'].timestamp()) * units.s
            if not (0 <= end_of_run_t - self.last_pulse_time <= 60 * units.s):
                self.log.warning("Run is %s long according to run db, but last pulse starts at %s. "
                                 "Did the DAQ crash?" % (pax_to_human_time(end_of_run_t),
                                                         pax_to_human_time(self.last_pulse_time)))

        # Insert some status info into the pipeline info
        if not self.secret_mode:
            if hasattr(self, 'last_time_searched'):
                lts = self.last_time_searched
            else:
                lts = 0
            self.pipeline_status_collection.insert({'name': 'eventbuilder_info',
                                                    'time': datetime.datetime.utcnow(),
                                                    'pax_id': self.config.get('pax_id', 'no_pax_id_set'),
                                                    'last_pulse_so_far_in_run': self.last_pulse_time,
                                                    'latest_subcollection': self.latest_subcollection,
                                                    'last_time_searched': lts,
                                                    'working_on_run': True,
                                                    })

    def get_events(self):
        self.log.info("Eventbuilder get_events starting up")
        self.refresh_run_info()
        self.log.info("Fetched runs db info successfully")

        # Last time (ns) searched, exclusive. ie we searched [something, last_time_searched)
        self.last_time_searched = self.initial_start_time
        self.log.info("self.initial_start_time: %s", pax_to_human_time(self.initial_start_time))
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
            next_time_to_search = self.last_time_searched
            if next_time_to_search != self.initial_start_time:
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
            batches_to_search = min(batches_to_search, self.max_query_workers // len(self.hosts))

            # Start new queries in separate processes
            with ThreadPoolExecutor(max_workers=self.max_query_workers) as executor:
                futures = []
                for batch_i in range(batches_to_search):
                    futures_per_host = []

                    # Get the query, and collection name needed for it
                    start = next_time_to_search + batch_i * self.batch_window
                    if self.split_collections:
                        subcol_i = self.subcollection_with_time(next_time_to_search) + batch_i
                        # Prep the query -- not a very difficult one :-)
                        query = {}
                        collection_name = self.subcollection_name(subcol_i)
                        self.log.info("Submitting query for subcollection %d" % subcol_i)
                    else:
                        collection_name = self.run_doc['name']
                        stop = start + self.batch_window
                        query = self.time_range_query(start, stop)
                        self.log.info("Submitting query for batch %d, time range [%s, %s)" % (
                            batch_i, pax_to_human_time(start), pax_to_human_time(stop)))

                    # Do the query on each host
                    for host in self.hosts:
                        future = executor.submit(get_pulses,
                                                 client_maker_config=self.cm.config,
                                                 query=query,
                                                 input_info=self.input_info,
                                                 collection_name=collection_name,
                                                 host=host,
                                                 get_area=self.config['can_get_area'])
                        futures_per_host.append(future)
                    futures.append(futures_per_host)

                # Record advancement of the batch window
                self.last_time_searched = next_time_to_search + batches_to_search * self.batch_window

                # Check if there is more data
                more_data_coming = (not self.data_taking_ended) or (self.last_time_searched <
                                                                    end_of_search_for_this_run)
                if not more_data_coming:
                    self.log.info("Searched to %s, which is beyond %s. This is the last batch of data" % (
                        pax_to_human_time(self.last_time_searched), pax_to_human_time(end_of_search_for_this_run)))

                # Check if we've passed the user-specified stop (if so configured)
                stop_after_sec = self.config.get('stop_after_sec', None)
                if stop_after_sec and 0 < stop_after_sec < float('inf'):
                    if self.last_time_searched > stop_after_sec * units.s:
                        self.log.warning("Searched to %s, which is beyond the user-specified stop at %d sec."
                                         "This is the last batch of data" % (self.last_time_searched,
                                                                             self.config['stop_after_sec']))
                        more_data_coming = False

                # Retrieve results from the queries, then pass everything to the trigger
                for i, futures_per_host in enumerate(futures):
                    if len(futures_per_host) == 1:
                        assert not self.split_hosts
                        times, modules, channels, areas = futures_per_host[0].result()
                    else:
                        assert self.split_hosts
                        times = []
                        modules = []
                        channels = []
                        areas = []
                        for f in futures_per_host:
                            ts, ms, chs, ars = f.result()
                            times.append(ts)
                            modules.append(ms)
                            channels.append(chs)
                            areas.append(ars)
                        times = np.concatenate(times)
                        modules = np.concatenate(modules)
                        channels = np.concatenate(channels)
                        areas = np.concatenate(areas)

                    times = times * self.sample_duration

                    if len(times):
                        self.log.info("Batch %d: acquired pulses in range [%s, %s]" % (
                                      i,
                                      pax_to_human_time(times[0]),
                                      pax_to_human_time(times[-1])))
                    else:
                        self.log.info("Batch %d: No pulse data found." % i)

                    # Send the new data to the trigger, which will build events from it
                    # Note the data is still unsorted: the trigger will take care of sorting it.
                    for data in self.trigger.run(last_time_searched=next_time_to_search + (i + 1) * self.batch_window,
                                                 start_times=times,
                                                 channels=channels,
                                                 modules=modules,
                                                 areas=areas,
                                                 last_data=(not more_data_coming and i == len(futures) - 1)):
                        yield EventProxy(event_number=next_event_number, data=data, block_id=-1)
                        next_event_number += 1

        # We've built all the events for this run!
        # Compile the end of run info for the run doc and for display
        trigger_end_info = self.trigger.shutdown()
        trigger_end_info.update(dict(ended=True,
                                     status='processed',
                                     trigger_monitor_data_location=self.uri_for_monitor,
                                     mongo_reader_config={k: v for k, v in self.config.items()
                                                          if k != 'password' and
                                                          k not in self.processor.config['DEFAULT']}))

        if not self.secret_mode:
            end_of_run_info = {'trigger.%s' % k: v for k, v in trigger_end_info.items()}
            self.runs_collection.update_one({'_id': self.config['run_doc_id']},
                                            {'$set': end_of_run_info})
        self.log.info("Event building complete. Trigger information: %s" % trigger_end_info)


class MongoDBReadUntriggeredFiller(plugin.TransformPlugin, MongoBase):

    """Read pulse data into event ranges provided by trigger MongoDBReadUntriggered.
    This is a separate plugin, since reading the raw pulse data is the expensive operation we want to parallelize.
    """
    do_input_check = False

    def startup(self):
        MongoBase.startup(self)
        self.ignored_channels = []
        self.max_pulses_per_event = self.config.get('max_pulses_per_event', float('inf'))
        self.high_energy_prescale = self.config.get('high_energy_prescale', 0.1)

        self.log.info("Software HEV settings: %s max pulses per event, %s prescale" % (self.max_pulses_per_event,
                                                                                       self.high_energy_prescale))

        # Load the digitizer channel -> PMT index mapping
        self.detector = self.config['detector']
        self.pmts = self.config['pmts']
        self.pmt_mappings = {(x['digitizer']['module'],
                              x['digitizer']['channel']): x['pmt_position'] for x in self.pmts}

    def _get_cursor_between_times(self, start, stop, subcollection_number=None):
        """Returns count, cursor over all pulses that start in [start, stop) (both pax units since start of run).
        Order of pulses is not defined.
        count is 0 if max_pulses_per_event is float('inf'), since we don't care about it in that case.
        Does NOT deal with time ranges split between subcollections, but does deal with split hosts.
        """
        cursors = []
        count = 0
        for host_i, host in enumerate(self.hosts):
            if subcollection_number is None:
                assert not self.split_collections
                collection = self.input_collections[host_i]
            else:
                assert self.split_collections
                collection = self.subcollection(subcollection_number, host_i)
            query = self.time_range_query(start, stop)
            cursor = collection.find(query)
            # Ask for a large batch size: the default is 101 documents or 1MB. This results in a very small speed
            # increase (when I measured it on a normal dataset)
            cursor.batch_size(int(1e7))
            cursors.append(cursor)
            if self.max_pulses_per_event != float('inf'):
                count += collection.count(query)
        if len(self.hosts) == 1:
            return count, cursors[0]
        else:
            return count, chain(*cursors)

    def transform_event(self, event_proxy):
        # t0, t1 are the start, stop time of the event in pax units (ns) since the start of the run
        (t0, t1), trigger_signals = event_proxy.data
        self.log.debug("Fetching data for event with range [%s, %s]",
                       pax_to_human_time(t0),
                       pax_to_human_time(t1))

        event = Event(n_channels=self.config['n_channels'],
                      block_id=event_proxy.block_id,
                      start_time=t0 + self.time_of_run_start,
                      sample_duration=self.sample_duration,
                      stop_time=t1 + self.time_of_run_start,
                      dataset_name=self.run_doc['name'],
                      event_number=event_proxy.event_number,
                      trigger_signals=trigger_signals)

        # Convert trigger signal times to time since start of event
        event.trigger_signals['left_time'] -= t0
        event.trigger_signals['right_time'] -= t0
        event.trigger_signals['time_mean'] -= t0

        if self.split_collections:
            start_col = self.subcollection_with_time(t0)
            end_col = self.subcollection_with_time(t1)
            if start_col == end_col:
                count, mongo_iterator = self._get_cursor_between_times(t0, t1, start_col)
                if count > self.max_pulses_per_event:
                    # Software "veto" the event to prevent overloading the event builder
                    if np.random.rand() > self.high_energy_prescale:
                        self.log.debug("VETO: %d pulses in event %s" % (len(event.pulses), event.event_number))
                        event.n_pulses = int(count)
                        return event
            else:
                self.log.info("Found event [%s-%s] which straddles subcollection boundary." % (
                    pax_to_human_time(t0), pax_to_human_time(t1)))
                # Ignore the software-HEV in this case
                mongo_iterator = chain(self._get_cursor_between_times(t0, t1, start_col)[1],
                                       self._get_cursor_between_times(t0, t1, end_col)[1])
        else:
            mongo_iterator = self._get_cursor_between_times(t0, t1)

        data_is_compressed = self.input_info['compressed']
        for i, pulse_doc in enumerate(mongo_iterator):
            digitizer_id = (pulse_doc['module'], pulse_doc['channel'])
            pmt = self.pmt_mappings.get(digitizer_id)
            if pmt is not None:
                # Fetch the raw data
                data = pulse_doc['data']
                if data_is_compressed:
                    data = snappy.decompress(data)

                time_within_event = self._from_mt(pulse_doc['time']) - t0  # ns

                event.pulses.append(Pulse(left=self._to_mt(time_within_event),
                                          raw_data=np.fromstring(data,
                                                                 dtype="<i2"),
                                          channel=pmt,
                                          do_it_fast=True))
            elif digitizer_id not in self.ignored_channels:
                self.log.warning("Found data from digitizer module %d, channel %d,"
                                 "which doesn't exist according to PMT mapping! Ignoring...",
                                 pulse_doc['module'], pulse_doc['channel'])
                self.ignored_channels.append(digitizer_id)

        self.log.debug("%d pulses in event %s" % (len(event.pulses), event.event_number))
        return event


class MongoDBClearUntriggered(plugin.TransformPlugin, MongoBase):
    """Clears data whose events have been built from MongoDB>
    Will do NOTHING unless delete_data = True in config.
    This must run as part of the output group, so it gets the events in order.

    If split_collections:
        Drop sub collections when events from subsequent collections start arriving.
        Drop all remaining subcollections on shutdown

    Else (single collection mode):
        Keeps track of which time is safe to delete, then deletes data from the collection in batches.
        At shutdown, drop the collection

    """
    do_input_check = False
    do_output_check = False
    last_time_deleted = 0
    last_subcollection_not_yet_deleted = 0

    def startup(self):
        MongoBase.startup(self)
        self.executor = ThreadPoolExecutor(max_workers=self.config['max_query_workers'])
        if not self.config['delete_data']:
            self.log.info("Will NOT rescue acquisition monitor pulses, need delete_data enabled for this")
            return
        aqm_file_path = self.config.get('acquisition_monitor_file_path')
        if aqm_file_path is None:
            self.log.info("Will NOT rescue acquisition monitor pulses!")
            self.aqm_output_handle = None
        else:
            # Get the acquisition monitor module from the pmts dictionary in the config
            # It's a bit bad we've hardcoded 'sum_wv' as detector name here...
            some_ch_from_aqm = self.config['channels_in_detector']['sum_wv'][0]
            self.aqm_module = self.config['pmts'][some_ch_from_aqm]['digitizer']['module']
            self.log.info("Acquisition monitor (module %d) pulses will be saved to %s" % (
                self.aqm_module, aqm_file_path))
            self.aqm_output_handle = open(aqm_file_path, mode='wb')

        self.already_rescued_collections = []

    def transform_event(self, event_proxy):
        if not self.config['delete_data']:
            return event_proxy

        time_since_start = event_proxy.data['stop_time'] - self.time_of_run_start
        if self.split_collections:
            coll_number = self.subcollection_with_time(time_since_start)
            while coll_number > self.last_subcollection_not_yet_deleted:
                self.log.info("Seen event at subcollection %d, clearing subcollection %d" % (
                    coll_number, self.last_subcollection_not_yet_deleted))
                self.drop_collection_named(self.subcollection_name(self.last_subcollection_not_yet_deleted),
                                           self.executor)
                self.last_subcollection_not_yet_deleted += 1
        else:
            if time_since_start > self.last_time_deleted + self.config['batch_window']:
                self.log.info("Seen event at %s, clearing all data until then." % pax_to_human_time(time_since_start))
                for coll in self.input_collections:
                    self.executor.submit(self.delete_pulses,
                                         coll,
                                         start_mongo_time=self._to_mt(self.last_time_deleted),
                                         stop_mongo_time=self._to_mt(time_since_start))
                self.last_time_deleted = time_since_start

        return event_proxy

    def shutdown(self):
        if self.config['delete_data']:

            # Wait for any slow drops to complete
            self.log.info("Waiting for slow collection drops to complete...")
            self.executor.shutdown()
            self.log.info("Collection drops should be complete. Checking for remaining collections.")

            pulses_in_remaining_collections = defaultdict(int)
            for db in self.dbs:
                for coll_name in db.collection_names():
                    if not coll_name.startswith(self.run_doc['name']):
                        continue
                    pulses_in_remaining_collections[coll_name] += db[coll_name].count()

            if len(pulses_in_remaining_collections):
                self.log.info("Remaining collections with pulse counts: %s. Clearing these now." % (
                                  str(pulses_in_remaining_collections)))

                for colname in pulses_in_remaining_collections.keys():
                    self.drop_collection_named(colname)
                self.log.info("Completed.")

            else:
                self.log.info("All collections have already been cleaned, great.")

            # Update the run doc to remove the 'untriggered' entry
            # since we just deleted the last of the untriggered data
            self.refresh_run_doc()
            self.runs_collection.update_one({'_id': self.run_doc['_id']},
                                            {'$set': {'data': [d for d in self.run_doc['data']
                                                               if d['type'] != 'untriggered']}})

        if hasattr(self, 'aqm_output_handle') and self.aqm_output_handle is not None:
            self.aqm_output_handle.close()

    def rescue_acquisition_monitor_pulses(self, collection, query=None):
        """Saves all acquisition monitor pulses from collection the acquisition monitor data file.
         - collection: pymongo object (not collection name!)
         - query: optional query inside the collection (e.g. for a specific time range).
        The condition to select pulses from the acquistion monitor module will be added to this.
        """
        if self.aqm_output_handle is None:
            return

        if query is None:
            query = {}
        query['module'] = self.aqm_module

        # Count first, in case something is badly wrong and we end up saving bazillions of docs we'll at least have
        # a fair warning...
        n_to_rescue = collection.count(query)
        self.log.info("Saving %d acquisition monitor pulses" % n_to_rescue)

        for doc in collection.find(query):
            self.aqm_output_handle.write(pickle.dumps(doc))

        # Flush explicitly: we want to save the data even if the event builder crashes before properly closing the file
        self.aqm_output_handle.flush()

    def delete_pulses(self, collection, start_mongo_time, stop_mongo_time):
        """Deletes all pulses in collection between start_mongo_time (inclusive) and stop_mongo_time (exclusive),
        both in mongo time units (not pax units!).
        """
        query = {'time': {'$gte': start_mongo_time,
                          '$lt': stop_mongo_time}}
        self.rescue_acquisition_monitor_pulses(collection, query)
        collection.delete_many(query)

    def drop_collection_named(self, collection_name, executor=None):
        """Drop the collection named collection_name from db, rescueing acquisition monitor pulses first.
        if executor is passed, will execute the drop command via the pool it represents.

        This function is NOT parallelizable itself, don't pass it to an executor!
        We need to block while rescuing acquisition monitor pulses: otherwise, we would get to the final cleanup in
        shutdown() while there are still collections being rescued.
        """
        if self.aqm_db is not None:
            if collection_name not in self.already_rescued_collections:
                self.already_rescued_collections.append(collection_name)
                self.rescue_acquisition_monitor_pulses(self.aqm_db[collection_name])
            else:
                self.log.warning("Duplicated call to drop collection %s!" % collection_name)
        for db in self.dbs:
            if executor is None:
                db.drop_collection(collection_name)
            else:
                executor.submit(db.drop_collection, collection_name)


def pax_to_human_time(num):
    """Converts a pax time to a human-readable representation"""
    for x in ['ns', 'us', 'ms', 's', 'ks', 'Ms', 'G', 'T']:
        if num < 1000.0:
            return "%3.3f %s" % (num, x)
        num /= 1000.0
    return "%3.1f %s" % (num, 's')


def get_pulses(client_maker_config, input_info, collection_name, query, host, get_area=False):
    """Find pulse times according to query using monary.
    Returns four numpy arrays: times, modules, channels, areas.
    Areas consists of zeros unless get_area = True, in which we also fetch the 'integral' field.

    The monary client is created inside this function, so we could run it with ProcessPoolExecutor.
    """
    fields = ['time', 'module', 'channel'] + (['integral'] if get_area else [])
    types = ['int64', 'int32', 'int32'] + (['area'] if get_area else [])

    try:
        client_maker = ClientMaker(client_maker_config)

        monary_client = client_maker.get_client(database_name=input_info['database'],
                                                uri=input_info['location'],
                                                host=host,
                                                monary=True)

        # Somehow monary's block query fails when we have multiple blocks,
        # we need to take care of copying out the data ourselves, but even if I use .copy it doesn't seem to work
        # Never mind, just make a big block
        results = list(monary_client.block_query(input_info['database'], collection_name, query, fields, types,
                                                 block_size=int(5e8),
                                                 select_fields=True))
        monary_client.close()

    except monary.monary.MonaryError as e:
        if 'Failed to resolve' in str(e):
            raise exceptions.DatabaseConnectivityError("Caught error trying to connect to untriggered database. "
                                                       "Original exception: %s." % str(e))
        raise e

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

    return times, modules, channels, areas

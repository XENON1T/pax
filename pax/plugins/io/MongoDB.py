"""Interfacing to MongoDB

MongoDB is used as a data backend within the DAQ.  For example, 'kodiaq', which
reads out the digitizers, will write data to MongoDB.  This data from kodiaq can
either be triggered or untriggered. In the case of untriggered, an event builder
must be run on the data and will result in triggered data.  Input and output
classes are provided for MongoDB access.  More information is in the docstrings.
"""
import numpy as np
import numba

import time
import pymongo
import snappy

from pax.datastructure import Event, Pulse
from pax import plugin, units


START_KEY = 'time'
STOP_KEY = 'endtime'


@numba.jit((numba.int64)(numba.int64[:],
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
            return "%3.1f %s" % (num, x)
        num /= 1000.0
    return "%3.1f %s" % (num, 's')


class IOMongoDB():

    def startup(self):
        if START_KEY == STOP_KEY:
            raise ValueError("START_KEY and STOP_KEY must be different."
                             "Otherwise, must modify query logic.")

        self.number_of_events = 0

        self.connections = {}  # MongoClient objects
        self.mongo = {}        #

        self.pmt_mappings = self.config['pmt_mappings']

        # Each MongoDB class must acquire the run database document that
        # describes this acquisition.  This is partly because the state can
        # change midrun.  For example, the run can end.
        self.run_doc_id = self.config['run_doc']
        self.log.debug("Run doc %s", self.run_doc_id)
        self.setup_access('run',
                          **self.config['runs_database_location'])

        self.log.info("Fetching run document %s",
                      self.run_doc_id)
        self.query = {'_id': self.run_doc_id}
        update = {'$set': {'trigger.status': 'processing'}}
        self.run_doc = self.mongo['run']['collection'].find_one_and_update(self.query,
                                                                           update)
        self.sort_key = [(START_KEY, 1),
                         (START_KEY, 1)]

        self.mongo_find_options = {'sort': self.sort_key,
                                   'cursor_type': pymongo.cursor.CursorType.EXHAUST}

        # Digitizer bins
        self.sample_duration = self.config.get('sample_duration')
        self.log.debug("Time unit: %f", self.sample_duration)

        # Retrieved from runs database
        self.data_taking_ended = False

    def _to_mt(self, x):
        # Takes time in ns and converts to samples
        # makes number small by 10
        return int(x * units.ns // self.sample_duration)

    def _from_mt(self, x):
        # Takes time in samples and converts to ns
        # Makes number bigger by 10
        return int(x * self.sample_duration // units.ns)

    def setup_access(self, name,
                     address,
                     database,
                     collection,
                     port=27017):
        wc = pymongo.write_concern.WriteConcern(w=0)

        m = {}  # Holds connection, database info, and collection info

        # Used for coordinating which runs to analyze
        self.log.debug("Connecting to: %s" % address)
        if address not in self.connections:
            if '/' in address:  # HACK, fix kodiaq
                replica_set, address = address.split('/')

                c = pymongo.MongoClient(address,
                                        replicaSet=replica_set,
                                        serverSelectionTimeoutMS=500)
            else:

                c = pymongo.MongoClient(address,
                                        port,
                                        serverSelectionTimeoutMS=500)
            self.connections[address] = c

            try:
                self.connections[address].admin.command('ping')
                self.log.debug("Connection succesful")
            except pymongo.errors.ConnectionFailure:
                self.log.fatal("Cannot connect to MongoDB at %s" % address)
                raise

        m['client'] = self.connections[address]

        self.log.debug('Fetching databases: %s', database)
        m['database'] = m['client'].get_database(database,
                                                 write_concern=wc)

        self.log.debug('Getting collection: %s', collection)
        m['collection'] = m['database'].get_collection(collection,
                                                       write_concern=wc)
        self.mongo[name] = m

    def setup_input(self):
        self.log.info("run_doc")
        self.log.info(self.run_doc['reader'])

        buff = self.run_doc['reader']['storage_buffer']

        # Delete after Dan's change in kodiaq issue #48
        buff2 = {}
        buff2['address'] = buff['dbaddr']
        buff2['database'] = buff['dbname']
        buff2['collection'] = buff['dbcollection']
        buff = buff2

        self.setup_access('input',
                          **buff)
        self.mongo['input']['collection'].ensure_index(self.sort_key)

        self.compressed = self.run_doc['reader']['compressed']

    def update_run_doc(self):
        self.run_doc = self.mongo['run']['collection'].find_one(self.query)
        self.data_taking_ended = self.run_doc['reader']['data_taking_ended']

    def number_events(self):
        return self.number_of_events

    @staticmethod
    def chunks(l, n):
        """ Yield successive n-sized chunks from l.
        """
        for i in range(0, len(l), n):
            yield l[i:i + n]


class MongoDBReadUntriggered(plugin.InputPlugin,
                             IOMongoDB):

    def startup(self):
        IOMongoDB.startup(self) # Setup with baseclass
        
        # Load constants from config
        self.window = self.config['window']
        self.multiplicity = self.config['multiplicity']
        self.left = self.config['left_extension']
        self.right = self.config['right_extension']

        self.log.info("Building events with:")
        self.log.info("\tSliding window: %0.2f us", self.window / units.us * units.ns)
        self.log.info("\tMultiplicity: %d hits", self.multiplicity)
        self.log.info("\tLeft extension: %0.2f us", self.left / units.us * units.ns)
        self.log.info("\tRight extension: %0.2f us", self.right / units.us * units.ns)

        self.setup_input()

    def sliding_window(self, x, n, window=1000, multiplicity=3, left=-10, right=7):
        """Sliding window cluster finder (with range extension)

        x is a list of times.  A window will slide over the values in x and
        this function will return all event ranges with more than 'multiplicity'
        of pulses.  We assume that any pulses will have ~1 pe area.
        Also, left and right will be added to ranges, where left can be the
        drift length.
        """
        if left > 0:
            raise ValueError("Left offset must be negative")
        ranges = []

        i = 0  # Start of range to test
        j = 0  # End of range to test

        while j < x.size:  # For every pulse... extend end
            if x[j] - x[i] > window:  # If time diff greater than window, form new cluster
                if j - i > multiplicity:  # If more than 100 pulses, trigger
                    if len(ranges) > 0 and ranges[-1][1] + window > x[i]:
                        ranges[-1][1] = x[j - 1] + right
                    else:
                        ranges.append([x[i] + left, x[j - 1] + right])
                i += 1
            else:
                j += 1

        if j - i > multiplicity:  # If more than 10 pulses, trigger
            ranges.append([x[i] + left, x[j - 1] + right])

        return ranges

    def sliding_window2(self, x, ranges_buffer_size,
                        window=1000, multiplicity=3,
                        left=-10, right=7):
        """Sliding window cluster finder (with range extension)

        x is a list of times.  A window will slide over the values in x and
        this function will return all event ranges with more than 'multiplicity'
        of pulses.  We assume that any pulses will have ~1 pe area.
        Also, left and right will be added to ranges, where left can be the
        drift length.  n is the number of possible ranges.
        """
        if left > 0:
            raise ValueError("Left offset must be negative")

        ranges_buffer = -1 * np.ones((ranges_buffer_size,
                                      2),
                                     dtype=np.int64)

        n_ranges_found = _sliding_window_numba(x, window, multiplicity,
                                               left, right, ranges_buffer,
                                               ranges_buffer_size)
        if n_ranges_found >= ranges_buffer_size:
            print("Too many ranges to fit in buffer... :-(?")
        return ranges_buffer[:n_ranges_found]

    def get_events(self):
        self.last_time = 0  # ns

        while not self.data_taking_ended:
            # Grab new run document in case run ended.  This much happen before
            # processing data to avoid a race condition where the run ends
            # between processing and checking that the run has ended
            #
            self.update_run_doc()

            self.ranges = []

            c = self.mongo['input']['collection']

            delay = 0  # ns

            search_after = self.last_time - delay  # ns
            self.log.info("Searching for pulses after %s",
                          sampletime_fmt(search_after))
            # times is in digitizer samples
            times = list(c.find({'time': {'$gt': self._to_mt(search_after),
                                          '$lt': self._to_mt(search_after + 60.0 * units.s)}},
                                projection=[START_KEY, STOP_KEY],
                                **self.mongo_find_options))

            n = len(times)
            if n == 0:
                self.log.fatal("Nothing found, continue")
                time.sleep(1)  # todo: configure
                continue

            x = [[self._from_mt(doc[START_KEY]),
                  self._from_mt(doc[STOP_KEY])] for doc in times]  # in digitizer units
            x = np.array(x, dtype=np.int64)
            x = x.mean(axis=1, dtype=np.int64)  # in ns

            self.log.info("Processing range [%s, %s]",
                          sampletime_fmt(x[0]),
                          sampletime_fmt(x[-1]))

            self.last_time = x[-1]   # TODO race condition? subtract second?
            self.first_time = x[0]
            rate = 2000
            n = int(((x[-1] - x[0]) // units.s) * rate)

            self.ranges = self.sliding_window(x,
                                              n,
                                              window=self.window,
                                              multiplicity=self.multiplicity,
                                              left=self.left,
                                              right=self.right)

            self.log.info("Found %d events", len(self.ranges))
            self.number_of_events = len(self.ranges)

            yield Event(n_channels=self.config['n_channels'],
                            start_time=x[0],
                            sample_duration=self.sample_duration,
                            stop_time=x[-1],
                            partial=True,
                            event_number = 0)

            break
            for i, this_range in enumerate(self.ranges):
                # Start pax's timer so we can measure how fast this plugin goes
                ts = time.time()
                t0, t1 = [int(t) for t in this_range]
        # docs.append({'test' : 0,
        #                     'docs' : pulses})

                self.total_time_taken += (time.time() - ts) * 1000

                yield Event(n_channels=self.config['n_channels'],
                            start_time=t0,
                            sample_duration=self.sample_duration,
                            stop_time=t1,
                            partial=True,
                            i = i)

            # If run ended, begin cleanup
            #
            # This variable is updated at the start of while loop.
            if self.data_taking_ended:
                self.log.fatal("Data taking ended.")
                status = self.mongo['run']['collection'].update_one({'_id': self.run_doc_id},
                                                                    {'$set': {'trigger.status': 'processed',
                                                                              'trigger.ended': True}})
                self.log.debug(status)


class MongoDBReadUntriggeredFiller(plugin.TransformPlugin, IOMongoDB):

    def startup(self):
        IOMongoDB.startup(self)  # Setup with baseclass

        self.setup_input()

    def process_event(self, event):
        t0, t1 = int(event.start_time), int(event.stop_time)  # ns

        event = Event(start_time=event.start_time,
                      stop_time=event.stop_time,
                      n_channels=self.config['n_channels'],
                      sample_duration=self.sample_duration)

        self.log.info("Building event in range [%s, %s]",
                      sampletime_fmt(t0),
                      sampletime_fmt(t1))

        query = {START_KEY: {"$gte": self._to_mt(t0)},
                 STOP_KEY: {"$lte": self._to_mt(t1)}}

        self.mongo_iterator = self.mongo['input']['collection'].find(query,
                                                                     **self.mongo_find_options)
        pulse_objects = []

        for i, pulse_doc in enumerate(self.mongo_iterator):
            # Fetch raw data from document
            data = pulse_doc['data']

            time_within_event = self._from_mt(pulse_doc[START_KEY]) - t0  # ns

            if self.compressed:
                data = snappy.decompress(data)

            id = str((pulse_doc['module'],
                      pulse_doc['channel']))
            if id in self.pmt_mappings:
                channel = self.pmt_mappings[id]

                pulse_objects.append(Pulse(left=self._to_mt(time_within_event),
                                           raw_data=np.fromstring(data,
                                                                  dtype="<i2"),
                                           channel=channel))
            else:
                self.log.warning("Found data from module %d digitizer channel "
                                 "%d, but not in PMT mapping.  Ignoring.",
                                 pulse_doc['module'],
                                 pulse_doc['channel'])
        self.log.debug("%d pulses added", len(pulse_objects))
        event.pulses = pulse_objects
        return event


class MongoDBWriteTriggered(plugin.OutputPlugin,
                            IOMongoDB):

    def startup(self):
        IOMongoDB.startup(self)  # Setup with baseclass
        self.setup_access('output')
        self.c = self.mongo['output']['collection']

    def write_event(self, event_class):
        # Write the data to database

        # Convert event class to pymongo-able dictionary
        event_dict = event_class.to_dict(convert_numpy_arrays_to='bytes')

        # Insert dictionary into database
        self.c.write(event_dict)

class MongoDBWriteTriggered(plugin.OutputPlugin,
                            IOMongoDB):

    def startup(self):
        IOMongoDB.startup(self)  # Setup with baseclass
        self.setup_access('output')
        self.c = self.mongo['output']['collection']

    def write_event(self, event_class):
        # Write the data to database

        # Convert event class to pymongo-able dictionary
        event_dict = event_class.to_dict(convert_numpy_arrays_to='bytes')

        # Insert dictionary into database
        self.c.write(event_dict)

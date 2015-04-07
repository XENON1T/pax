"""Interfacing to MongoDB

MongoDB is used as a data backend within the DAQ.  For example, 'kodiaq', which
reads out the digitizers, will write data to MongoDB.  This data from kodiaq can
either be triggered or untriggered. In the case of untriggered, an event builder
must be run on the data and will result in triggered data.  Input and output
classes are provided for MongoDB access.  More information is in the docstrings.
"""
from datetime import datetime
import numpy as np

import time
import pymongo
import snappy

from bson.binary import Binary
from pax.datastructure import Event, Occurrence
from pax import plugin, units


START_KEY = 'start_time'
STOP_KEY = 'stop_time'

class IOMongoDB():
    def startup(self):
        self.number_of_events = 0

        self.connections = {}  # MongoClient objects
        self.mongo = {}        #

        #self.setup_access('run')

        self.sort_key = [(START_KEY, 1),
                         (START_KEY, 1)]
        self.mongo_time_unit = int(self.config.get('sample_duration'))

    def setup_access(self, name):
        wc = pymongo.write_concern.WriteConcern(w=0)

        m = {}  # Holds connection, database info, and collection info
        try:
            # Used for coordinating which runs to analyze
            self.log.debug("Connecting to %s" % self.config['%s_address' % name])
            m['client'] = self._get_connection(self.config['%s_address' % name])
            m['database'] = m['client'].get_database(self.config['%s_database' % name],
                                                     write_concern=wc)
            m['collection'] = m['database'].get_collection(self.config['%s_collection' % name],
                                                           write_concern=wc)
        except pymongo.errors.ConnectionFailure as e:
            self.log.fatal("Cannot connect to MongoDB!")
            self.log.exception(e)
            raise

        self.mongo[name] = m


    def _get_connection(self, hostname):
        if hostname not in self.connections:
            try:
                self.connections[hostname] = pymongo.MongoClient(hostname)
            except pymongo.errors.ConnectionFailure:
                self.log.fatal("Cannot connect to MongoDB at %s" % hostname)

        return self.connections[hostname]

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
        self.setup_access('input')

        # Load constants from config
        self.window = self.config['window']
        self.multiplicity =self.config['multiplicity']
        self.left = self.config['left_extension']
        self.right = self.config['right_extension']

        self.log.info("Building events with:")
        self.log.info("\tSliding window: %0.2f us", self.window / units.us)
        self.log.info("\tMultiplicity: %d hits", self.multiplicity)
        self.log.info("\tLeft extension: %0.2f us", self.left / units.us)
        self.log.info("\tRight extension: %0.2f us", self.right / units.us)

        c = self.mongo['input']['collection']

        c.ensure_index(self.sort_key)

        times = list(c.find(projection=[START_KEY, STOP_KEY],
                                 sort=self.sort_key,
                                 cursor_type=pymongo.cursor.EXHAUST))


        x = self.extract_times_from_occurrences(times,
                                                self.mongo_time_unit)
        self.log.debug(x[0:10])  # x in ns

        self.ranges = self.sliding_window(x,
                                         window=self.window,
                                         multiplicity=self.multiplicity,
                                         left=self.left,
                                         right=self.right)

        self.log.info("Found %d events", len(self.ranges))
        self.number_of_events = len(self.ranges)

    @staticmethod
    def extract_times_from_occurrences(times, sample_duration):
        x = [[doc[START_KEY], doc[STOP_KEY]] for doc in times]
        x = np.array(x) * sample_duration
        x = x.mean(axis=1)
        return x

    @staticmethod
    def sliding_window(x, window=1000, multiplicity=3, left=-10, right=7):
        """Sliding window cluster finder (with range extension)

        x is a list of times.  A window will slide over the values in x and
        this function will return all event ranges with more than 'multiplicity'
        of occurrences.  We assume that any occurrence will have ~1 pe area.
        Also, left and right will be added to ranges, where left can be the
        drift length.
        """
        if left > 0:
            raise ValueError("Left offset must be negative")
        ranges = []

        i = 0  # Start of range to test
        j = 0  # End of range to test

        while j < x.size:  # For every occureence... extend end
            if x[j] - x[i] > window:  # If time diff greater than window, form new cluster
                if j - i > multiplicity:  # If more than 100 occurences, trigger
                    if len(ranges) > 0 and ranges[-1][1] + window > x[i]:
                        ranges[-1][1] = x[j-1] + right
                    else:
                        ranges.append([x[i] + left, x[j-1] + right])
                i+= 1
            else:
                j += 1

        if j - i > multiplicity:  # If more than 10 occurences, trigger
            ranges.append([x[i] + left, x[j-1] + right])

        return ranges

    def get_events(self): # {"$lt": d}
        for i, this_range in enumerate(self.ranges):
            # Start pax's timer so we can measure how fast this plugin goes
            ts = time.time()
            t0, t1 = [int(x) for x in this_range]

            self.total_time_taken += (time.time() - ts) * 1000

            yield Event(n_channels=self.config['n_channels'],
                        start_time=t0,
                        sample_duration=self.mongo_time_unit,
                        stop_time=t1,
                        partial=True
                        #occurrences=occurrence_objects
            )

class MongoDBReadUntriggeredFiller(plugin.TransformPlugin, IOMongoDB):
    def startup(self):
        IOMongoDB.startup(self) # Setup with baseclass
        self.setup_access('input')

    def process_event(self, event):
        t0, t1 = event.start_time, event.stop_time

        event = Event(start_time = event.start_time,
                      stop_time = event.stop_time,
                      n_channels=self.config['n_channels'],
                      sample_duration=self.mongo_time_unit)

        self.log.debug("Building event in range [%d,%d]", t0, t1)

        query = {START_KEY : {"$gte" : t0 / self.mongo_time_unit},
                 STOP_KEY : {"$lte" : t1 / self.mongo_time_unit}}

        self.mongo_iterator = self.mongo['input']['collection'].find(query)
                                                                     #exhaust = True)
        occurrence_objects = []

        for i, occurrence_doc in enumerate(self.mongo_iterator):
            # Fetch raw data from document
            data = occurrence_doc["data"]

            time_within_event = int(occurrence_doc[START_KEY]) - (t0 // self.mongo_time_unit)
            self.log.debug(time_within_event)
            self.log.debug(t0)

            occurrence_objects.append(Occurrence(left=(time_within_event),
                                                 raw_data=np.fromstring(data,
                                                                        dtype="<i2"),
                                                 channel=int(occurrence_doc['channel'])))

        event.occurrences = occurrence_objects
        return event


class MongoDBWriteUntriggered(plugin.OutputPlugin,
                              IOMongoDB):
    """Inject PMT pulses into DAQ to test trigger.

    This plugin aims to emulate the DAQReader by creating run control documents
    and feeding raw data into the DAQ's MongoDB format.  Consult here for more
    on formats:

    https://docs.google.com/drawings/d/1dytKBmMARsZtuyUmLbzm9IbXM1hre
    -knkEIU4X3Ot8U/edit

    Note: write run document after output collection created.
    """

    def startup(self):
        """Setup"""
        IOMongoDB.startup(self) # Setup with baseclass

        self.setup_access('run')
        self.setup_access('raw')

        self.mongo['raw']['collection'].ensure_index(self.sort_key)

        self.mongo['raw']['collection'].ensure_index([('_id', pymongo.HASHED)])

        # self.log.info("Sharding %s" % str(c))
        # self.raw_client.admin.command('shardCollection',
        #                              '%s.%s' % (self.config['raw_database'], self.config['raw_collection']),
        #                              key = {'_id': pymongo.HASHED})

        # Send run doc
        self.query = {"name": self.config['raw_collection'],
                      "starttimestamp": str(datetime.now()),
                      "runmode": "calibration",
                      "reader": {
                          "compressed": False,
                          "starttimestamp": str(datetime.now()),
                          "data_taking_ended": False,
                          "options": {},
                          "storage_buffer": {
                              "dbaddr": self.config['raw_address'],
                              "dbname": self.config['raw_database'],
                              "dbcollection": self.config['raw_collection'],
                              },
                          },
                      "trigger": {
                          "mode": "calibration",
                          "status": "waiting_to_be_processed",
                          },
                      "processor": {"mode": "something"},
                      "comments": [],
                      }

        self.log.info("Injecting run control document")
        self.mongo['run']['collection'].insert(self.query)

        self.bob = pymongo.bulk.BulkOperationBuilder(self.mongo['raw']['collection'],
                                                     ordered=False)
        self.bobi = 0

    def shutdown(self):
        """Notify run database that datataking stopped
        """
        self.handle_occurrences()  # write remaining data

        # Update runs DB
        self.query['reader']['stoptimestamp'] = str(datetime.now())
        self.query['reader']['data_taking_ended'] = True
        self.mongo['run']['collection'].save(self.query)

    def write_event(self, event):
        self.log.debug('Writing event')

        # We have to divide by the sample duration because the DAQ expects units
        # of 10 ns.  However, note that the division is done with a // operator.
        # This is an integer divide, thus gives an integer back.  If you do not
        # do this, you will store the time as a float, which will lead to
        # problems with the time precision (and weird errors in the DSP).  See
        # issue #35 for more info.
        time = event.start_time // event.sample_duration
        assert self.mongo_time_unit == event.sample_duration
        assert isinstance(time, int)

        for oc in event.occurrences:
            occurence_doc = {}

            # occurence_doc['_id'] = uuid.uuid4()
            occurence_doc['module'] = oc.channel  # TODO: fix wax
            occurence_doc['channel'] = oc.channel

            occurence_doc[START_KEY] = time + oc.left
            occurence_doc[STOP_KEY] = time + oc.right


            data = np.array(oc.raw_data,
                            dtype=np.int16).tostring()
            if self.query['reader']['compressed']:
                data = snappy.compress(data)

            # Convert raw samples into BSON format
            occurence_doc['data'] = Binary(data, 0)
            self.bob.insert(occurence_doc)
            self.bobi += 1

        self.handle_occurrences()

    def handle_occurrences(self):
        if self.bobi > 10000:
            self.bob.execute(write_concern={'w':0})
            self.bob = pymongo.bulk.BulkOperationBuilder(self.mongo['raw']['collection'],
                                                         ordered=False)
            self.bobi = 0

class MongoDBWriteTriggered(plugin.OutputPlugin,
                            IOMongoDB):
    def startup(self):
        IOMongoDB.startup(self) # Setup with baseclass
        self.setup_access('output')
        self.c = self.mongo['output']['collection']

    def write_event(self, event):
        # Write the data to database
        self.c.write(event.to_dict(json_compatible=True))


class MongoDBFakeDAQOutput(plugin.OutputPlugin):

    """Inject PMT pulses into DAQ to test trigger.

    This plugin aims to emulate the DAQReader by creating run control documents
    and feeding raw data into the DAQ's MongoDB format.  Consult here for more
    on formats:

    https://docs.google.com/drawings/d/1dytKBmMARsZtuyUmLbzm9IbXM1hre
    -knkEIU4X3Ot8U/edit

    Note: write run document after output collection created.
    """

    def startup(self):
        """Setup"""

        # Collect all events in a buffer, then inject them at the end.
        self.collect_then_dump = self.config['collect_then_dump']
        self.repeater = int(self.config['repeater'])  # Hz repeater
        self.runtime = int(self.config['runtime'])  # How long run repeater

        # Schema for input collection
        self.start_time_key = START_TIME_KEY
        self.stop_time_key = 'time_max'
        self.bulk_key = 'bulk'

        self.connections = {}

        try:
            self.client = pymongo.MongoClient(self.config['run_address'])

            # Used for coordinating which runs to analyze
            self.log.debug("Connecting to %s" % self.config['run_address'])
            self.run_client = self.get_connection(self.config['run_address'])

            # Used for storing the binary output from digitizers
            self.log.debug("Connecting to %s" % self.config['raw_address'])
            self.raw_client = self.get_connection(self.config['raw_address'])

        except pymongo.errors.ConnectionFailure as e:
            self.log.fatal("Cannot connect to database")
            self.log.exception(e)
            raise

        self.run_database = self.run_client[self.config['run_database']]
        self.raw_database = self.raw_client[self.config['raw_database']]

        self.run_collection = self.run_database[self.config['run_collection']]
        self.raw_collection = self.raw_database[self.config['raw_collection']]

        self.raw_collection.ensure_index([(self.start_time_key, -1)])
        self.raw_collection.ensure_index([(self.start_time_key, 1)])
        self.raw_collection.ensure_index([(self.stop_time_key, -1)])
        self.raw_collection.ensure_index([(self.stop_time_key, 1)])

        self.raw_collection.ensure_index([('_id', pymongo.HASHED)])

        # self.log.info("Sharding %s" % str(c))
        # self.raw_client.admin.command('shardCollection',
        #                              '%s.%s' % (self.config['raw_database'], self.config['raw_collection']),
        #                              key = {'_id': pymongo.HASHED})

        # Send run doc
        self.query = {"name": self.config['name'],
                      "starttimestamp": str(datetime.now()),
                      "runmode": "calibration",
                      "reader": {
                          "compressed": True,
                          "starttimestamp": 0,
                          "data_taking_ended": False,
                          "options": {},
                          "storage_buffer": {
                              "dbaddr": self.config['raw_address'],
                              "dbname": self.config['raw_database'],
                              "dbcollection": self.config['raw_collection'],
                          },
        },
            "trigger": {
                          "mode": "calibration",
                          "status": "waiting_to_be_processed",
        },
            "processor": {"mode": "something"},
            "comments": [],
        }

        self.log.info("Injecting run control document")
        self.run_collection.insert(self.query)

        # Used for computing offsets so reader starts from zero time
        self.starttime = None

        self.occurences = []

    def get_connection(self, hostname):
        if hostname not in self.connections:
            try:
                self.connections[hostname] = pymongo.Connection(hostname)

            except pymongo.errors.ConnectionFailure:
                self.log.fatal("Cannot connect to mongo at %s" % hostname)

        return self.connections[hostname]

    def shutdown(self):
        """Notify run database that datataking stopped
        """
        self.handle_occurences()  # write remaining data

        # Update runs DB
        self.query['reader']['stoptimestamp'] = str(datetime.now())
        self.query['reader']['data_taking_ended'] = True
        self.run_collection.save(self.query)

    @staticmethod
    def chunks(l, n):
        """ Yield successive n-sized chunks from l.
        """
        for i in range(0, len(l), n):
            yield l[i:i + n]

    def write_event(self, event):
        self.log.debug('Writing event')

        # We have to divide by the sample duration because the DAQ expects units
        # of 10 ns.  However, note that the division is done with a // operator.
        # This is an integer divide, thus gives an integer back.  If you do not
        # do this, you will store the time as a float, which will lead to
        # problems with the time precision (and weird errors in the DSP).  See
        # issue #35 for more info.
        time = event.start_time // event.sample_duration

        assert isinstance(time, int)

        if self.starttime is None:
            self.starttime = time
        elif time < self.starttime:
            error = "Found events before start of run"
            self.log.fatal(error)
            raise RuntimeError(error)

        for oc in event.occurrences:
            pmt_num = oc.channel
            sample_position = oc.left
            samples_occurrence = oc.raw_data

            assert isinstance(sample_position, int)

            occurence_doc = {}

            # occurence_doc['_id'] = uuid.uuid4()
            occurence_doc['module'] = pmt_num  # TODO: fix wax
            occurence_doc['channel'] = pmt_num

            occurence_doc['time'] = time + sample_position - self.starttime

            data = np.array(samples_occurrence, dtype=np.int16).tostring()
            if self.query['reader']['compressed']:
                data = snappy.compress(data)

            # Convert raw samples into BSON format
            occurence_doc['data'] = Binary(data, 0)

            self.occurences.append(occurence_doc)

        if not self.collect_then_dump:
            self.handle_occurences()

    def handle_occurences(self):
        docs = self.occurences  # []
        # for occurences in list(self.chunks(self.occurences,
        # 1000)):

        # docs.append({'test' : 0,
        #                     'docs' : occurences})

        i = 0
        t0 = time.time()  # start time
        t1 = time.time()  # last time

        if self.repeater > 0:
            while (t1 - t0) < self.runtime:
                this_time = time.time()
                n = int((this_time - t1) * self.repeater)
                if n == 0:
                    continue

                t1 = this_time
                self.log.fatal('How many events to inject %d', n)

                modified_docs = []
                min_time = None
                max_time = None

                for _ in range(n):
                    i += 1
                    for doc in self.occurences:
                        # doc['_id'] = uuid.uuid4()
                        doc['time'] += i * (t1 - t0) / self.repeater

                        if min_time is None or doc['time'] < min_time:
                            min_time = doc['time']
                        if max_time is None or doc['time'] > max_time:
                            max_time = doc['time']

                        modified_docs.append(doc.copy())

                        if len(modified_docs) > 1000:
                            self.raw_collection.insert({self.start_time_key: min_time,
                                                        self.stop_time_key: max_time,
                                                        self.bulk_key: modified_docs},
                                                       w=0)
                            modified_docs = []
                            min_time = None
                            max_time = None

        elif len(docs) > 0:
            times = [doc['time'] for doc in docs]
            min_time = min(times)
            max_time = max(times)

            t0 = time.time()

            self.raw_collection.insert({self.start_time_key: min_time,
                                        self.stop_time_key: max_time,
                                        self.bulk_key: docs},
                                       w=0)

            t1 = time.time()

        self.occurences = []


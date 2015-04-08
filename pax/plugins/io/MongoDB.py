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


START_KEY = 'time'
STOP_KEY = 'time'

class IOMongoDB():
    def startup(self):
        self.number_of_events = 0

        self.connections = {}  # MongoClient objects
        self.mongo = {}        #

        self.run_doc_id = None
        self.setup_access('run', **self.config['runs_database_location'])

        self.sort_key = [(START_KEY, 1),
                         (START_KEY, 1)]
        self.mongo_time_unit = int(self.config.get('sample_duration'))

    def setup_access(self, name,
                     address,
                     database,
                     collection):
        wc = pymongo.write_concern.WriteConcern(w=0)

        m = {}  # Holds connection, database info, and collection info

        try:
            # Used for coordinating which runs to analyze
            self.log.debug("Connecting to: %s" % address)
            m['client'] = self._get_connection(address)

            self.log.debug('Fetching databases: %s', database)
            m['database'] = m['client'].get_database(database,
                                                     write_concern=wc)

            self.log.debug('Getting collection: %s', collection)
            m['collection'] = m['database'].get_collection(collection,
                                                           write_concern=wc)
        except pymongo.errors.ConnectionFailure as e:
            self.log.fatal("Cannot connect to MongoDB!")
            self.log.exception(e)
            raise

        self.mongo[name] = m


    def _get_connection(self, hostname):
        if hostname not in self.connections:
            self.connections[hostname] = pymongo.MongoClient(hostname,
                                                             serverSelectionTimeoutMS=500)
                
            try:
                self.connections[hostname].admin.command('ping')
                self.log.debug("Connection succesful")
            except pymongo.errors.ConnectionFailure:
                self.log.fatal("Cannot connect to MongoDB at %s" % hostname)
                raise

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

        self.query = {"trigger.status" : "waiting_to_be_processed"}

        self.wait_time = 1


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

    def get_events(self):
       self.last_time = 0
       while 1:
            if self.run_doc_id is None:
                self.log.fatal("Searching for run")
                self.last_time = 0

                doc = self.mongo['run']['collection'].find_one_and_update(self.query,
                                                                          {'$set': {'trigger.status' : 'in_progress'}})

                if doc is None:
                    self.log.fatal("Nothing found, waiting %d seconds...",
                                   self.wait_time)
                    time.sleep(self.wait_time)
                    continue

                self.run_doc_id = doc['_id']

                buff = doc['reader']['storage_buffer']

                # Delete after Dan's change in kodiaq issue #48
                buff2 = {}
                buff2['address'] = buff['dbaddr']
                buff2['database'] = buff['dbname']
                buff2['collection'] = buff['dbcollection']
                buff = buff2

                self.setup_access('input',
                                  **buff)
                self.mongo['input']['collection'].ensure_index(self.sort_key)

            self.ranges = []

            c = self.mongo['input']['collection']

            delay = 0

            times = list(c.find({'time' : {'$gt' : (self.last_time - delay)}},
                                projection=[START_KEY, STOP_KEY],
                                sort=self.sort_key,
                                #cursor_type=pymongo.cursor.EXHAUST
                            ))

            if len(times) == 0:
                self.log.fatal("Nothing found, continue")
                continue

            x = self.extract_times_from_occurrences(times,
                                                    self.mongo_time_unit)

            self.log.info("Processing range [%d, %d]",
                          x[0], x[-1])

            self.last_time = x[-1]  # TODO race condition? subtract second?

            self.ranges = self.sliding_window(x,
                                             window=self.window,
                                             multiplicity=self.multiplicity,
                                             left=self.left,
                                             right=self.right)

            self.log.info("Found %d events", len(self.ranges))
            self.number_of_events = len(self.ranges)

            for i, this_range in enumerate(self.ranges):
                # Start pax's timer so we can measure how fast this plugin goes
                ts = time.time()
                t0, t1 = [int(x) for x in this_range]

                self.total_time_taken += (time.time() - ts) * 1000

                yield Event(n_channels=self.config['n_channels'],
                            start_time=t0,
                            sample_duration=self.mongo_time_unit,
                            stop_time=t1,
                            partial=True)

            # Check if run ended.  If so, end processing.  Otherwise, retry.
            # Update run document
            doc = self.mongo['run']['collection'].find_one({'_id' : self.run_doc_id})

            # Has data aquisition ended
            data_taking_ended = doc['reader']['data_taking_ended']

            if data_taking_ended:
                self.log.fatal("Data taking ended.")
                status = self.mongo['run']['collection'].update_one({'_id' : self.run_doc_id},
                                      {'$set' : {'trigger.status' : 'processed',
                                                 'trigger.ended' : True}})
                self.log.debug("Updated rundoc")
                self.run_doc_id = None

class MongoDBReadUntriggeredFiller(plugin.TransformPlugin, IOMongoDB):
    def startup(self):
        IOMongoDB.startup(self) # Setup with baseclass
        self.setup_access('run')
        #self.setup_access('input')

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


class MongoDBWriteTriggered(plugin.OutputPlugin,
                            IOMongoDB):
    def startup(self):
        IOMongoDB.startup(self) # Setup with baseclass
        self.setup_access('output')
        self.c = self.mongo['output']['collection']

    def write_event(self, event):
        # Write the data to database
        self.c.write(event.to_dict(json_compatible=True))


"""Interfacing to DAQ via MongoDB

The DAQ uses MongoDB for input and output.  The classes defined hear allow the
user to read data from the DAQ and also inject raw occurences into the DAQ.

"""
import time
import uuid
from datetime import datetime

import pymongo
import snappy
import numpy as np
from bson.binary import Binary

from pax.datastructure import Event
from pax import plugin


class MongoDBInput(plugin.InputPlugin):
    def startup(self):
        self.log.debug("Connecting to %s" % self.config['address'])
        try:
            self.client = pymongo.MongoClient(self.config['address'])
            self.database = self.client[self.config['database']]
            self.collection = self.database[self.config['collection']]
        except pymongo.errors.ConnectionFailure as e:
            self.log.fatal("Cannot connect to database")
            self.log.exception(e)
            raise

        # TODO (tunnell): Sort by event number
        self.cursor = self.collection.find()
        self.number_of_events = self.cursor.count()

        if self.number_of_events == 0:
            raise RuntimeError(
                "No events found... did you run the event builder?")

    def get_events(self):
        """Generator of events from Mongo

        What is returned is all the channel's occurrences
        """
        for i, doc_event in enumerate(self.collection.find()):
            self.log.debug("Fetching document %s" % repr(doc_event['_id']))

            # Store channel waveform-occurrences by iterating over all
            # occurrences.
            # This involves parsing MongoDB documents using WAX output format
            event = Event()
            event.event_number = i  # TODO: should come from Mongo

            assert isinstance(doc_event['range'][0], int)
            assert isinstance(doc_event['range'][1], int)

            # Units is 10 ns from DAQ, but 1 ns within pax
            event.start_time = int(doc_event['range'][0]) * 10
            event.stop_time = int(doc_event['range'][1]) * 10

            assert isinstance(event.start_time, int)
            assert isinstance(event.stop_time, int)

            # Key is channel number, value is list of occurences
            occurrences = {}

            for doc_occurrence in doc_event['docs']:
                # +1 so it works with Xenon100 gain lists
                channel = doc_occurrence['channel']
                if channel not in occurrences:
                    occurrences[channel] = []

                assert isinstance(doc_occurrence['time'], int)
                assert isinstance(doc_event['range'][0], int)

                occurrences[channel].append((
                    # Start sample index
                    doc_occurrence['time'] - doc_event['range'][0],
                    # SumWaveform occurrence data
                    np.fromstring(doc_occurrence['data'], dtype="<i2"),
                ))

            event.occurrences = occurrences

            if event.length() == 0:
                raise RuntimeWarning("Empty event")

            yield event


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
        self.repeater = int(self.config['repeater']) # Hz repeater
        self.runtime = int(self.config['runtime']) # How long run repeater


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

        if self.config['raw_collection'] in self.raw_database.collection_names():
            self.log.error("Data already exists at output location... deleting")
            self.raw_database.drop_collection(self.config['raw_collection'])

        self.raw_collection = self.raw_database[self.config['raw_collection']]

        self.raw_collection.ensure_index([('time', -1),
                                          ('module', -1),
                                          ('_id', -1)])
        self.raw_collection.ensure_index([('time', 1),
                                          ('module', 1),
                                          ('_id', 1)])


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

            except pymongo.errors.ConnectionFailure as e:
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

        for pmt_num, payload in event.occurrences.items():
            for sample_position, samples_occurrence in payload:
                assert isinstance(sample_position, int)

                occurence_doc = {}

                occurence_doc['_id'] = uuid.uuid4()
                occurence_doc['module'] = pmt_num  # TODO: fix wax
                occurence_doc['channel'] = pmt_num

                occurence_doc['time'] = time + sample_position - self.starttime

                data = snappy.compress(np.array(samples_occurrence,
                                                dtype=np.int16).tostring())

                # Convert raw samples into BSON format
                occurence_doc['data'] = Binary(data, 0)

                self.occurences.append(occurence_doc)

        if not self.collect_then_dump:
            self.handle_occurences()

    def handle_occurences(self):
        docs = self.occurences  # []

        # for occurences in list(self.chunks(self.occurences,
        # 1000)):

        #docs.append({'test' : 0,
        #                     'docs' : occurences})

        i = 0
        t0 = time.time() # start time
        t1 = time.time() # last time

        if self.repeater > 0:
            while (t1 - t0) < self.runtime:
                this_time = time.time()
                n = int((this_time - t1) * self.repeater)
                if n == 0:
                    continue

                t1 = this_time
                self.log.fatal('times %d', n)

                modified_docs = []
                doc = self.occurences[0]
                for _ in range(n):
                    i += 1
                    doc['_id'] = uuid.uuid4()
                    doc['time'] += i * (t1 - t0) / self.repeater
                    modified_docs.append(doc.copy())

                if len(modified_docs) > 0:
                    self.log.fatal('size %d', len(modified_docs))
                    self.raw_collection.insert(modified_docs,
                                               w=0)


        elif len(docs) > 0:
            t0 = time.time()
            self.raw_collection.insert(docs,
                                       w=0)
            t1 = time.time()

        self.occurences = []



"""Interfacing to DAQ via MongoDB

The DAQ uses MongoDB for input and output.  The classes defined hear allow the
user to read data from the DAQ and also inject raw occurences into the DAQ.

"""
import time
import uuid
import datetime

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

    https://docs.google.com/drawings/d/1dytKBmMARsZtuyUmLbzm9IbXM1hre-knkEIU4X3Ot8U/edit

    Note: write run document after output collection created.
    """

    def startup(self):
        """Setup"""
        self.log.debug("Connecting to %s" % self.config['address'])
        try:
            self.client = pymongo.MongoClient(self.config['address'])

            # Used for coordinating which runs to analyze
            self.run_database = self.client[self.config['run_database']]

            # Used for storing the binary output from digitizers
            self.raw_database = self.client[self.config['raw_database']]

        except pymongo.errors.ConnectionFailure as e:
            self.log.fatal("Cannot connect to database")
            self.log.exception(e)
            raise


        self.raw_collection = self.database[self.config['collection']]
        if self.config['collection'] in self.database.collection_names():
            self.error("Data already exists at output location... deleting")
            self.database.drop_collection(self.config['collection'])

        self.run_collection = self.database[self.config['collection']]

        # Send run doc
        self.query = {"name": self.config['name'],
                 "starttimestamp": str(datetime.now()),
                 "runmode": "calibration",
                 "reader": {
                     "compressed": True,
                     "starttimestamp": str(datetime.now()),
                     "data_taking_ended": False,
                     "options": self.config,
                     "storage_buffer": {
                         "dbaddr": self.config['address'],
                         "dbname": self.config['raw_database'],
                         "dbcollection": self.config['collection'],
                     },
                 },
                 "trigger": {
                     "mode": "calibration",
                 },
                 "processor": {"mode": "something"},
                 "comments": [],
                }
        self.run_collection.insert(self.query)

        self.occurences = []

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

        if self.control_doc['starttime'] is None:
            self.control_doc['starttime'] = time
        elif time < self.control_doc['starttime']:
            self.control_doc['starttime'] = time

        for pmt_num, payload in event.occurrences.items():
            for sample_position, samples_occurrence in payload:
                assert isinstance(sample_position, int)

                occurence_doc = {}

                occurence_doc['_id'] = uuid.uuid4()
                occurence_doc['module'] = pmt_num  # TODO: fix wax
                occurence_doc['channel'] = pmt_num

                occurence_doc['time'] = time + sample_position

                data = snappy.compress(np.array(samples_occurrence,
                                                dtype=np.int16).tostring())

                # Convert raw samples into BSON format
                occurence_doc['data'] = Binary(data, 0)

                self.occurences.append(occurence_doc)

        self.handle_occurences()

    def handle_occurences(self):
        docs = self.occurences  # []

        # for occurences in list(self.chunks(self.occurences,
        #                                   1000)):

        #docs.append({'test' : 0,
        #                     'docs' : occurences})

        if len(docs) > 0:
            t0 = time.time()
            self.collection.insert(docs,
                                   w=0)
            t1 = time.time()

            self.log.info('dt\t %d %d', t1 - t0, len(docs))

        self.occurences = []



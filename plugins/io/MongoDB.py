"""Interfacing to DAQ via MongoDB

The DAQ uses MongoDB for input and output.  The classes defined hear allow the
user to read data from the DAQ and also inject raw occurences into the DAQ.

"""
import time
import uuid

import pymongo
import numpy as np
from pax.datastructure import Event
from bson.binary import Binary

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
            raise RuntimeError("No events found... did you run the event builder?")

    def get_events(self):
        """Generator of events from Mongo

        What is returned is all the channel's occurrences
        """
        for i, doc_event in enumerate(self.collection.find()):
            self.log.debug("Fetching document %s" % repr(doc_event['_id']))

            # Store channel waveform-occurrences by iterating over all occurrences.
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

    def startup(self):
        self.log.debug("Connecting to %s" % self.config['address'])
        try:
            self.client = pymongo.MongoClient(self.config['address'])
            self.database = self.client[self.config['database']]
        except pymongo.errors.ConnectionFailure as e:
            self.log.fatal("Cannot connect to database")
            self.log.exception(e)
            raise

        if self.config['collection'] in self.database.collection_names():
            self.collection = self.database[self.config['collection']]
            if 'capped' not in self.collection.options() or self.collection.options()['capped']:
                self.log.error("not capped")
                #raise RuntimeError("Collection exists, but not capped")
        else:
            self.database.create_collection(self.config['collection'],
                                            capped = True,
                                            size = self.config['collection_size'])
        self.collection = self.database[self.config['collection']]
        # TODO (tunnell): Sort by event number
        self.cursor = self.collection.find()
        self.number_of_events = self.cursor.count()

        if self.number_of_events != 0:
            self.log.warning("Database collection not empty")
            cursor = self.collection.find({"runtype": {'$exists': True},
                                           "starttime": {'$exists': True},
                                           "compressed": {'$exists': True},
                                           "data_taking_ended": {'$exists': True},
                                           "data": {'$exists': False}})
            control_docs = list(cursor)

            if len(control_docs) > 1:
                raise RuntimeError("More than one control document found in %s." %
                                   self.config['collection'])
            elif len(control_docs) == 0:
                self.log.error("Preexisting data, but no control document")
                self.control_doc = {} # temp
                self.control_doc['compressed'] = False #temp
                self.control_doc['data_taking_ended'] = False # temp
                self.control_doc['runtype'] = 'xenon100' # temp
                self.control_doc['starttime'] = None # temp
                self.collection.insert(self.control_doc) # temp
            else:
                self.control_doc = control_docs[0]
        else:
            self.control_doc = {}
            self.control_doc['compressed'] = False
            self.control_doc['data_taking_ended'] = False
            self.control_doc['runtype'] = 'xenon100'
            self.control_doc['starttime'] = None
            self.collection.insert(self.control_doc)

        self.occurences = []

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

        if self.control_doc['starttime'] is None or time < self.control_doc['starttime']:
            self.control_doc['starttime'] = time

        for pmt_num, payload in event.occurrences.items():
            for sample_position, samples_occurrence in payload:
                assert isinstance(sample_position, int)

                occurence_doc = {}

                occurence_doc['_id'] = uuid.uuid4()
                occurence_doc['module'] = pmt_num  # TODO: fix wax
                occurence_doc['channel'] = pmt_num

                occurence_doc['time'] = time + sample_position

                # Convert raw samples into BSON format
                data = Binary(np.array(samples_occurrence,
                                       dtype=np.int16).tostring(),
                              0)
                occurence_doc['data'] = data
                
                self.occurences.append(occurence_doc)

        #self.handle_occurences()

    @staticmethod
    def chunks(l, n):
        """ Yield successive n-sized chunks from l.
        """
        for i in range(0, len(l), n):
            yield l[i:i+n]

    def handle_occurences(self):
        docs = []

        for occurences in list(self.chunks(self.occurences,
                                           1000)):
            
                docs.append({'test' : 0,
                             'docs' : occurences})

        t0 = time.time()
        while 1:
            self.collection.insert(docs, #self.occurences,
                                   w=0)
        t1 = time.time()

        self.log.fatal('dt\t %d', t1-t0)

        self.occurences = []

    def shutdown(self):
        self.handle_occurences()
        
        self.control_doc['data_taking_ended'] = True
        #self.collection.save(self.control_doc)


import numpy as np
from pymongo import MongoClient

from pax import plugin, units

class MongoDBInput(plugin.InputPlugin):

    def __init__(self, config):
        plugin.InputPlugin.__init__(self, config)

        self.client = MongoClient(config['address'])
        self.database = self.client[config['database']]
        self.collection = self.database[config['collection']]

        self.baseline = config['digitizer_baseline']

        # TODO (tunnell): Sort by event number
        self.cursor = self.collection.find()
        self.number_of_events = self.cursor.count()

        if self.number_of_events == 0:
            raise RuntimeError(
                "No events found... did you run the event builder?")


    def GetEvents(self):
        """Generator of events from Mongo

        What is returned is all the channel's occurences
        """
        for doc_event in self.collection.find():
            # Store channel waveform-occurences by iterating over all occurences.
            # This involves parsing MongoDB documents using WAX output format
            (event_start, event_end) = doc_event['range']
            event = {
                'length'          :   event_end - event_start,
                'channel_occurences'  :   {},
            }
            for doc_occurence in doc_event['docs']:
                channel = doc_occurence['channel']
                if channel not in event['channel_occurences']:
                    event['channel_occurences'][channel] = []
                event['channel_occurences'][channel].append((
                    doc_occurence['time'] - event_start,                    #Start sample index
                    np.fromstring(doc_occurence['data'], dtype=np.int16)    #Waveform occurence data
                ))
            yield event

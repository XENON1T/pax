from pymongo import MongoClient
import datetime

from pax import plugin


class OnlineMonitor(plugin.OutputPlugin):
    def __init__(self, config):
        plugin.OutputPlugin.__init__(self, config)

        self.client = MongoClient(config['address'])
        self.database = self.client[config['database']]
        self.collection = self.database[config['collection']]
        try:
            self.collection.ensure_index("timestamp", 3600, expireAfterSeconds=3600)
        except:
            print("Error connecting to monitoring database")

    def write_event(self,event):
        if len(event['peaks']) >0 :
            data = {"S2_0":event['peaks'][0]['top_and_bottom']['area'],
                    "timestamp": datetime.datetime.utcnow()}        
            self.collection.save(data)

import datetime

import pymongo
import numpy

from pax import plugin


class OnlineMonitor(plugin.OutputPlugin):

    def startup(self):
        self.client = pymongo.MongoClient(self.config['address'])
        self.database = self.client[self.config['database']]
        self.collection = self.database[self.config['collection']]
        self.waveformCollection = self.database[self.config['waveformcollection']]
        self.lastWaveformTime = datetime.datetime.utcnow()
        try:
            self.collection.ensure_index("timestamp", 3600, expireAfterSeconds=3600)
            self.waveformCollection.ensure_index("timestamp", 3600, expireAfterSeconds=3600)
        except pymongo.errors.ConnectionFailure as e:
            self.log.fatal("Cannot connect to database")
            self.log.exception(e)
            raise

    def write_event(self, event):
        nowtime = datetime.datetime.utcnow()
        if (nowtime - self.lastWaveformTime).seconds > 5:
            self.lastWaveformTime = nowtime
            insert = {"waveform": numpy.ndarray.tostring(event['sum_waveforms']['top_and_bottom']),
                      "timestamp": datetime.datetime.utcnow(), }

            self.waveformCollection.save(insert)
        if len(event['peaks']) > 0:
            data = {"S2_0": event['peaks'][0]['top_and_bottom']['area'],
                    "timestamp": datetime.datetime.utcnow(),

                    }
            self.collection.save(data)

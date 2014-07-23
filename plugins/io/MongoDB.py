import pymongo
import numpy as np

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
		for doc_event in self.collection.find():
			# Store channel waveform-occurrences by iterating over all occurrences.
			# This involves parsing MongoDB documents using WAX output format
			(event_start, event_end) = doc_event['range']
			event = {
				'length': event_end - event_start,
				'channel_occurrences': {},
			}
			for doc_occurrence in doc_event['docs']:
				channel = doc_occurrence['channel']
				if channel not in event['channel_occurrences']:
					event['channel_occurrences'][channel] = []
				event['channel_occurrences'][channel].append((
					doc_occurrence['time'] - event_start,  # Start sample index
					np.fromstring(doc_occurrence['data'], dtype=np.int16)  # Waveform occurrence data
				))
			yield event

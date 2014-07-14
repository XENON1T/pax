import numpy as np
from pymongo import MongoClient

from pax import plugin, units


class MongoDBInput(plugin.InputPlugin):

    def __init__(self, config):
        plugin.InputPlugin.__init__(self, config)

        self.client = MongoClient(config['mongodb_address'])
        self.database = self.client[config['database']]
        self.collection = self.database[config['collection']]

        self.baseline = config['digitizer_baseline']

        # TODO (tunnell): Sort by event number
        self.cursor = self.collection.find()
        self.number_of_events = self.cursor.count()

        if self.number_of_events == 0:
            raise RuntimeError(
                "No events found... did you run the event builder?")

        # Few steps of math for conversion factor
        self.conversion_factor = config[
            'digitizer_V_resolution'] * config['digitizer_t_resolution']
        self.conversion_factor /= config['gain']
        self.conversion_factor /= config['digitizer_resistor']
        self.conversion_factor /= config['digitizer_amplification']
        self.conversion_factor /= units.electron_charge

    @staticmethod
    def baseline_mean_stdev(samples):
        """ returns (baseline, baseline_stdev) """
        baseline_sample = samples[:42]
        return (
            np.mean(sorted(baseline_sample)[int(0.4 * len(baseline_sample)):int(0.6 * len(baseline_sample))
                                            ]),  # Ensures peaks in baseline sample don't skew computed baseline
            np.std(baseline_sample)  # ... but do count towards baseline_stdev!
        )
        # Don't want to just take the median as V-resolution is finite

    def GetEvents(self):
        """Generator of events from Mongo

        What is returned is all of the channel waveforms
        """
        for doc_event in self.collection.find():
            current_event_channels = {}

            # Build channel waveforms by iterating over all occurences.  This
            # involves parsing MongoDB documents using WAX output format
            (event_start, event_end) = doc_event['range']
            event_length = event_end - event_start
            for doc_occurence in doc_event['docs']:
                channel = doc_occurence['channel']
                wave_start = doc_occurence['time'] - event_start
                if channel not in current_event_channels:
                    current_event_channels[channel] = self.baseline * np.ones(event_length, dtype=np.int16)
                waveform = np.fromstring(doc_occurence['data'], dtype=np.int16)
                current_event_channels[channel][wave_start:wave_start + len(waveform)] = waveform

            # 'event' is what we will return
            event = {}
            event['channel_waveforms'] = {}

            # Remove baselines
            for channel, data in current_event_channels.items():

                baseline, _ = self.baseline_mean_stdev(data)

                event['channel_waveforms'][channel] = -1 * (data - baseline) * self.conversion_factor

            yield event

from pax import plugin, waveform_readers, units
import numpy as np
from pax import settings

class MongoDBInput(plugin.InputPlugin):
    def __init__(self, config):
        plugin.InputPlugin.__init__(self, config)
        self.mycounter = 0
        self.reader = waveform_readers.MongoDB(settings=settings, server='127.0.0.1', collection='dataset')
        self.reader_generator = self.reader.get_next_event()

        gain = config.get('gain')
        digitizer_resistor = config.get('digitizer_resistor')
        digitizer_amplification = config.get('digitizer_amplification')

        self.conversion_factor = self.reader.dV * self.reader.dt / (gain*digitizer_resistor*digitizer_amplification*units.electron_charge)

    @staticmethod
    def baseline_mean_stdev(samples):
        """ returns (baseline, baseline_stdev) """
        baseline_sample = samples[:42]
        return (
            np.mean(sorted(baseline_sample)[
                int(0.4*len(baseline_sample)):int(0.6*len(baseline_sample))
            ]),  #Ensures peaks in baseline sample don't skew computed baseline
            np.std(baseline_sample)                     #... but do count towards baseline_stdev!
        )
        #Don't want to just take the median as V-resolution is finite

    def GetNextEvent(self):
        event_index = next(self.reader_generator)

        #Read in the channel waveforms, convert to pe/ns
        channel_waveforms = {}
        for channel in self.reader.get_next_channel():
            channel_waveforms[channel] = self.reader.get_channel_data(channel)
            #waveformdumper.dump_channel_waveform(channel, waveforms[channel], {'gain':st.channel_data[channel]['gain']})
            baseline,_ = self.baseline_mean_stdev(channel_waveforms[channel])

            channel_waveforms[channel] = -1 * (channel_waveforms[channel] - baseline) * self.conversion_factor

        event = {}
        event['channel_waveforms'] = channel_waveforms
        return event








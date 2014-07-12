"""Digital signal processing"""
from pax import settings as st
from pax import plugin
import numpy as np

__author__ = 'tunnell'

class ComputeSumWaveform(plugin.TransformPlugin):
    def TransformEvent(self, event):
        channel_waveforms = event['channel_waveforms']
        sum_waveforms = {}
        #Compute summed waveforms
        for group, members in st.channel_groups.items():
            sum_waveforms[group] = sum([wave for name,wave in channel_waveforms.items() if name in members] )
            if type(sum_waveforms[group]) != type(np.array([])):
                sum_waveforms.pop(group)    #None of the group members have a waveform in this event, delete this group's waveform
                continue

        event['sum_waveforms'] = sum_waveforms
        return event

class FilterWaveforms(plugin.TransformPlugin):
    def TransformEvent(self, event):
        for key, value in event['sum_waveforms'].items():
            pass
        return event

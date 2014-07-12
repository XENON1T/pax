from pax import plugin
import matplotlib.pyplot as plt
import numpy as np
__author__ = 'tunnell'

def sizeof_fmt(num):
    """input is bytes"""
    for x in ['B', 'KB', 'MB', 'GB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0
    return "%3.1f %s" % (num, 'TB')


def sampletime_fmt(num):
    """num is in 10s of ns"""
    num *= 10
    for x in ['ns', 'us', 'ms']:
        if num < 1000.0:
            return "%3.1f %s" % (num, x)
        num /= 1000.0
    return "%3.1f %s" % (num, 's')

class PlottingWaveform(plugin.OutputPlugin):
    def WriteEvent(self, event):
        print(event.keys())
        plt.figure()

        for name, wf in event['sum_waveforms'].items():
            plt.plot(wf, label=name)

        plt.xlabel('Time in event [us]')
        plt.ylabel("ADC counts on 14-bit digitizer")
        plt.legend()
        plt.show()

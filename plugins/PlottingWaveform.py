import matplotlib.pyplot as plt

from pax import plugin


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
            plt.plot(
                event['filtered_waveforms'][name], '--', label='filtered %s' % name)

        plt.xlabel('Time in event [us]')
        plt.ylabel("ADC counts on 14-bit digitizer")
        plt.legend()

        plt.figure()

        # Plot all peaks
        for peak in event['peaks']['summed']:
            plt.hlines(1000, peak['left'], peak['right'])

        plt.plot(event['sum_waveforms']['summed'], label=name)
        plt.plot(event['filtered_waveforms']['summed'],
                 '--', label='filtered %s' % name)

        plt.legend()
        plt.xlabel('Time in event [us]')
        plt.ylabel("ADC counts on 14-bit digitizer")
        plt.show()

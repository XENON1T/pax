import matplotlib.pyplot as plt

from pax import plugin
import random

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

    def write_event(self, event):
        """Plot an event

        Will make a fancy plot with lots of arrows etc of a summed waveform
        """
        self.log.debug("Received event %s" % str(event.keys()))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        side = 1
        # Plot all peaks
        for peak in event['peaks']:
            if peak['rejected']:continue
            x = peak['top_and_bottom']['position_of_max_in_waveform']
            y = event['processed_waveforms']['top_and_bottom'][x]

            plt.hlines(y, peak['left'], peak['right'])
            ax.annotate('%s:%s' % (peak['peak_type'], int(peak['top_and_bottom']['area'])),
                        xy=(x, y),
                        xytext=(peak['top_and_bottom']['position_of_max_in_waveform'],
                                event['processed_waveforms']['top_and_bottom'][
                                    peak['top_and_bottom']['position_of_max_in_waveform']] + 100+random.random()*100),
                        arrowprops=dict(arrowstyle="fancy",
                                        fc="0.6", ec="none",
                                        connectionstyle="angle3,angleA=0,angleB=-90"))
            side *= -1

        plt.plot(event['processed_waveforms']['top_and_bottom'], label='top_and_bottom')
        plt.plot(event['processed_waveforms']['filtered_for_large_s2'],
                 '--', label='filtered_for_large_s2')
        plt.plot(event['processed_waveforms']['filtered_for_small_s2'],
                 '--', label='filtered_for_small_s2')

        plt.legend()
        plt.xlabel('Time in event [10 ns]')
        plt.ylabel("pe / bin")

        plt.show(block=False)

        self.log.info("Hit enter to continue...")
        input()

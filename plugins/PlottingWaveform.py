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
        self.log.debug("Received event %s" % str(event.keys()))

        fig = plt.figure()
        ax = fig.add_subplot(111)

        side = 1
        # Plot all peaks
        for peak in event['peaks']:
            x = peak['summed']['position_of_max_in_waveform']
            y = event['sum_waveforms']['summed'][x]

            plt.hlines(y, peak['left'], peak['right'])
            ax.annotate('%0.2f' % peak['summed']['area'],
                        xy=(x, y),
                        xytext=(peak['summed']['position_of_max_in_waveform'] + 20000 * side, event['sum_waveforms']['summed'][peak['summed']['position_of_max_in_waveform']] * 0.7),
                        arrowprops=dict(arrowstyle="fancy",
                                        fc="0.6", ec="none",
                                        connectionstyle="angle3,angleA=0,angleB=-90"))
            side *= -1

        plt.plot(event['sum_waveforms']['summed'], label='summed')
        plt.plot(event['filtered_waveforms']['filtered_for_large_s2'],
                 '--', label='filtered_for_large_s2')

        plt.legend()
        plt.xlabel('Time in event [10 ns]')
        plt.ylabel("pe / bin")

        plt.show(block=False)

        self.log.info("Hit enter to continue...")
        input()

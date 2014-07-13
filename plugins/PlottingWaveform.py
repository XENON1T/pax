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
        plt.figure()

        for name, wf in event['sum_waveforms'].items():
            plt.plot(wf, label=name)
            plt.plot(
                event['filtered_waveforms'][name], '--', label='filtered %s' % name)

        plt.xlabel('Time in event [us]')
        plt.ylabel("ADC counts on 14-bit digitizer")
        plt.legend()

        fig = plt.figure()
        ax = fig.add_subplot(111)

        side = 1
        # Plot all peaks
        for peak in event['peaks']['summed']:
            x = peak['summed']['position_of_max_in_waveform']
            y = event['sum_waveforms']['summed'][x]

            plt.hlines(y, peak['left'], peak['right'])
            ax.annotate('%0.2f' % peak['summed']['area'],
                        xy=(x,y),
                        xytext=(peak['summed']['position_of_max_in_waveform'] + 30000 * side, event['sum_waveforms']['summed'][peak['summed']['position_of_max_in_waveform']] * 1.3),
                        arrowprops=dict(arrowstyle="fancy",
                                fc="0.6", ec="none",
                                connectionstyle="angle3,angleA=0,angleB=-90"))
            side *= -1


        plt.plot(event['sum_waveforms']['summed'], label='summed')
        plt.plot(event['filtered_waveforms']['summed'],
                 '--', label='filtered %s' % 'summed')

        plt.legend()
        plt.xlabel('Time in event [us]')
        plt.ylabel("ADC counts on 14-bit digitizer")
        plt.show()

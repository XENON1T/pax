"""Interactive plotting

Use matplotlib to display various things about the event.
"""

import matplotlib.pyplot as plt
import random
import numpy as np
import os

from pax import plugin, units

class PlotBase(plugin.OutputPlugin):
    def startup(self):
        if self.config['output_dir'] is not None:
            self.output_dir = self.config['output_dir']
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        else:
            self.output_dir = None

        self.skip_counter = 0

        self.substartup()

    def substartup(self):
        pass

    def write_event(self, event):
        # Should we plot or skip?
        if self.skip_counter:
            self.skip_counter -= 1
            self.log.debug(
                "Skipping this event due to plot_every = %s. Skip counter at "
                "%s" % (
                    self.config['plot_every'], self.skip_counter
                ))
            return

        self.plot_event(event)

        self.finalize_plot(event.event_number)

    def plot_event(self, event):
        raise NotImplementedError()

    def finalize_plot(self, num = 0):
        """Finalize plotting and send to screen/file

        Call this instead of plt.show()
        """
        if self.output_dir:
            plt.savefig(self.output_dir + '/' + str(num) + '.png')
        else:
            plt.show(block=False)
            self.log.info("Hit enter to continue...")
            input()
            plt.close()
        self.skip_counter = self.config['plot_every'] - 1


class PlotWaveform(PlotBase):
    def plot_event(self, event):
        """Plot an event

        Will make a fancy plot with lots of arrows etc of a summed waveform
        """

        self.time_conversion_factor = event.sample_duration * units.ns / units.us

        rows = 2
        cols = 3

        plt.figure(figsize=(6 * cols, 4 * rows))

        # Plot the largest S1 and largest S2 in the event
        if event.S1s():
            ax = plt.subplot2grid((rows, cols), (0, 0))
            largest_s1 = sorted(
                event.S1s(), key=lambda x: x.area, reverse=True)[0]
            self.plot_waveform(
                event, left=largest_s1.left, right=largest_s1.right, pad=10)

            time = largest_s1.index_of_maximum * self.time_conversion_factor

            plt.title("S1 at %.1f us" % time)

        if event.S2s():
            ax = plt.subplot2grid((rows, cols), (0, 1))
            ax.yaxis.set_label_position("right")

            largest_s2 = sorted(event.S2s(), key=lambda x: x.area,
                                reverse=True)[0]

            pad = 200 if largest_s2.height > 100 else 50

            self.plot_waveform(event, left=largest_s2.left,
                               right=largest_s2.right, pad=pad)

            time = largest_s2.index_of_maximum * self.time_conversion_factor

            plt.title("S2 at %.1f us" % time)


        # Plot the total waveform
        plt.subplot2grid((rows, cols), (rows - 1, 0), colspan=cols)
        self.plot_waveform(event, show_peaks=True)
        legend = plt.legend(loc='upper left', prop={'size': 10})
        legend.get_frame().set_alpha(0.5)
        plt.tight_layout()

    def plot_waveform(self, event, left=0, right=None, pad=0, show_peaks=False):
        if right is None:
            right = event.length() - 1
        lefti = max(0, left - pad)
        righti = min(right + pad, event.length() - 1)
        nsamples = 1 + righti - lefti
        dt = event.sample_duration * units.ns
        xlabels = np.arange(
            lefti * dt / units.us,
            # 10+ labels will be cut later, prevents off by one errors
            (10 + righti) * dt / units.us,
            dt / units.us
        )
        xlabels = xlabels[:nsamples]
        plt.autoscale(True, axis='both', tight=True)
        for w in self.config['waveforms_to_plot']:
            waveform = event.get_waveform(w['internal_name'])
            plt.plot(xlabels,
                     waveform.samples[lefti:righti + 1],
                     label=w['plot_label'])

        plt.ylabel('Amplitude (pe/bin)')
        plt.xlabel('Time (us)')

        if show_peaks and event.peaks:
            # Plot all peaks
            max_y = max([p.height for p in event.peaks])

            for peak in event.peaks:
                x = peak.index_of_maximum * self.time_conversion_factor
                y = peak.height
                plt.hlines(y, peak.left * self.time_conversion_factor,
                           peak.right * self.time_conversion_factor)
                plt.annotate('%s:%s' % (peak.type, int(peak.area)),
                             xy=(x, y),
                             xytext=(
                                 x, y + (max_y - y) * (
                                 0.05 + 0.2 * random.random())),
                             arrowprops=dict(arrowstyle="fancy",
                                             fc="0.6", ec="none",
                                             connectionstyle="angle3,"
                                                             "angleA=0,"
                                                             "angleB=-90"))


class PlottingHitPattern(PlotBase):
    def substartup(self):
        self.pmts_top = self.config['pmts_top']
        self.pmts_bottom = self.config['pmts_bottom']
        self.pmt_locations = self.config['pmt_locations']

    def _plot(self, peak, ax, pmts):
        area = []
        points = []
        for pmt in pmts:
            area.append(peak.area_per_pmt[pmt] / 10)

            points.append((self.pmt_locations[pmt]['x'],
                           self.pmt_locations[pmt]['y']))


        c = ax.scatter(*zip(*points), c='red',
                        s=area, cmap=plt.cm.hsv)
        c.set_alpha(0.75)

    def plot_event(self, event):
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')

        if event.S2s() and len(event.S2s()) >= 1:
            S2 = event.S2s()[0]
            ax1.set_title('S2 top (from above?)')
            self._plot(S2, ax1, self.pmts_top)
            ax2.set_title('S2 bottom (from above?)')
            self._plot(S2, ax2, self.pmts_bottom)

        if event.S1s() and len(event.S1s()) >= 1:
            S1 = event.S1s()[0]
            ax3.set_title('S1 top (from above?)')
            self._plot(S1, ax3, self.pmts_top)
            ax4.set_title('S1 bottom (from above?)')
            self._plot(S1, ax4, self.pmts_bottom)

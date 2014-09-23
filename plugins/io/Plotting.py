"""Interactive plotting

Use matplotlib to display various things about the event.
"""

import matplotlib.pyplot as plt
import random
import numpy as np
import os

from pax import plugin, units


class PlotWaveform(plugin.OutputPlugin):

    def startup(self):
        self.plt = plt
        if self.config['output_dir'] is not None:
            self.output_dir = self.config['output_dir']
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        else:
            self.output_dir = None
        # If no skipping defined, don't skip any events
        if not 'plot_every' in self.config:
            self.config['plot_every'] = 1
        self.skip_counter = 0

    def write_event(self, event):
        """Plot an event

        Will make a fancy plot with lots of arrows etc of a summed waveform
        """
        # Should we plot or skip?
        if self.skip_counter:
            self.skip_counter -= 1
            self.log.debug("Skipping this event due to plot_every = %s. Skip counter at %s" % (
                self.config['plot_every'], self.skip_counter
            ))
            return
        self.time_conversion_factor = event.sample_duration * \
            units.ns / units.us
        plt = self.plt
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
            plt.title("S1 at %.1f us" %
                      (largest_s1.index_of_maximum * self.time_conversion_factor))
        if event.S2s():
            ax = plt.subplot2grid((rows, cols), (0, 1))
            ax.yaxis.set_label_position("right")
            largest_s2 = sorted(
                event.S2s(), key=lambda x: x.area, reverse=True)[0]
            pad = 200 if largest_s2.height > 100 else 50
            self.plot_waveform(
                event, left=largest_s2.left, right=largest_s2.right, pad=pad)
            #if largest_s2.height: #Eh.. wa?
            #    plt.yscale('log', nonposy='clip')
            #    plt.ylim(10 ** (-1), plt.ylim()[1])
            plt.title("S2 at %.1f us" %
                      (largest_s2.index_of_maximum * self.time_conversion_factor))

        #plt.subplot2grid((rows,cols), (0,2))
        #plt.title('Event %s from %s' % (event.event_number, 'mysterious_dataset'))
        # Todo: plot hitmap of s2

        # Plot the total waveform
        plt.subplot2grid((rows, cols), (rows - 1, 0), colspan=cols)
        self.plot_waveform(event, show_peaks=True)
        legend = plt.legend(loc='upper left', prop={'size': 10})
        legend.get_frame().set_alpha(0.5)
        plt.tight_layout()
        if self.output_dir:
            plt.savefig(self.output_dir + '/' + str(event.event_number) + '.png')
        else:
            plt.show(block=False)
            self.log.info("Hit enter to continue...")
            input()
            plt.close()
        self.skip_counter = self.config['plot_every'] - 1

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
            plt.plot(
                xlabels, waveform.samples[lefti:righti + 1], label=w['plot_label'])
        # try:
        #     plt.plot(xlabels, event.get_waveform('tpc').samples[lefti:righti + 1],  label='TPC')
        #     plt.plot(xlabels, event.get_waveform('filtered_for_s2').samples[lefti:righti + 1], label='TPC - filtered')
        #     plt.plot(xlabels, event.get_waveform('veto').samples[lefti:righti + 1], label='Veto')
        # except:
        #     plt.plot(xlabels, event.get_waveform('uS1').samples[lefti:righti + 1], label='S1 peakfinding')
        #     plt.plot(xlabels, event.get_waveform('uS2').samples[lefti:righti + 1], label='S2 peakfinding')
        #     plt.plot(xlabels, event.get_waveform('filtered_for_large_s2').samples[lefti:righti + 1], label='Large s2 filtered')
        #     plt.plot(xlabels, event.get_waveform('filtered_for_small_s2').samples[lefti:righti + 1], label='Small s2 filtered')
        #     plt.plot(xlabels, [0.6241506363] * nsamples,  '--', label='Large S2 threshold')
        #     plt.plot(xlabels, [0.06241506363] * nsamples,  '--', label='Small S2 threshold')
        #     plt.plot(xlabels, [0.1872451909] * nsamples,  '--', label='S1 threshold')
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
                                 x, y + (max_y - y) * (0.05 + 0.2 * random.random())),
                             arrowprops=dict(arrowstyle="fancy",
                                             fc="0.6", ec="none",
                                             connectionstyle="angle3,angleA=0,angleB=-90"))


# class PlottingHitPattern(plugin.OutputPlugin):
#
#     def startup(self):
#         self.topArrayMap = self.config['topArrayMap']
#
#     def write_event(self, event):
#         """Plot an event
#
#         Will make a fancy plot with lots of arrows etc of a summed waveform
#         """
#         plt.figure()
#
#         hit_position = []
#         areas = []
#
#         S2 = event.S2s()[0]
#         S1 = event.S1s()[0]
#
#         for pmt, pmt_location in self.topArrayMap.items():
#             q = event.pmt_waveforms[pmt].sum()/10
#
#             q_s1 = event.pmt_waveforms[pmt, S1.left:S1.right].sum()/10
#             q_s2 = event.pmt_waveforms[pmt, S2.left:S2.right].sum()/10
#
#             hit_position.append((pmt_location['x'],
#                                  pmt_location['y']))
#
#             areas.append((q, q_s1, q_s2))
#
#         area = np.array(area) / 10
#
#         c = plt.scatter(*zip(*points), c='red',
#                         s=area, cmap=plt.cm.hsv)
#         c.set_alpha(0.75)
#
#         plt.show(block=False)
#
#         self.log.info("Hit enter to continue...")
#         input()

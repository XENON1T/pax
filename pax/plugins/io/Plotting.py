"""Interactive plotting

Use matplotlib to display various things about the event.
"""

import matplotlib.pyplot as plt
import random
import numpy as np
import os
import time
from mpl_toolkits.mplot3d import Axes3D

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
        # Convert times in samples to times in us: need to know how many samples / us
        self.samples_to_us = self.config['digitizer_t_resolution'] / units.us

        self.substartup()

    def substartup(self):
        pass

    def write_event(self, event):
        # Should we plot or skip?
        if self.skip_counter:
            self.skip_counter -= 1
            self.log.debug("Skipping this event due to plot_every = %s. Skip counter at %s" % (self.config['plot_every'],
                                                                                               self.skip_counter))
            return

        self.plot_event(event)
        self.finalize_plot(event.event_number)

    def plot_event(self, event):
        raise NotImplementedError()

    def finalize_plot(self, num = 0):
        """Finalize plotting, send to screen/file, then closes plot properly (avoids runtimewarning / memory leak).
        """
        if self.output_dir:
            plt.savefig(self.output_dir + '/' + str(num) + '.png')
        else:
            plt.show(block=False)
            self.log.info("Hit enter to continue...")
            input()
        plt.close()
        self.skip_counter = self.config['plot_every'] - 1


    def plot_waveform(self, event, left=0, right=None, pad=0, show_peaks=False, show_legend=True, log_y_axis=False):
        """
        Plot part of an event's sum waveform. Defined in base class to ensure a uniform style
        """
        if right is None:
            right = event.length() - 1
        lefti = max(0, left - pad)
        righti = min(right + pad, event.length() - 1)
        nsamples = 1 + righti - lefti
        dt = event.sample_duration * units.ns

        xlabels = np.arange(lefti * dt / units.us, # 10+ labels will be cut later, prevents off by one errors
                            (10 + righti) * dt / units.us,
                            dt / units.us)
        xlabels = xlabels[:nsamples]


        plt.autoscale(True, axis='both', tight=True)

        if log_y_axis:
            plt.yscale('log')
            plt.ylabel('Amplitude + 1 (pe/bin)')
            y_offset = 1
        else:
            plt.ylabel('Amplitude (pe/bin)')
            y_offset = 0
        plt.xlabel('Time (us)')

        for w in self.config['waveforms_to_plot']:
            waveform = event.get_waveform(w['internal_name'])
            plt.plot(xlabels,
                     waveform.samples[lefti:righti + 1] + y_offset,
                     label=w['plot_label'])
        if log_y_axis:
            plt.ylim((0.9,plt.ylim()[1]))


        if show_peaks and event.peaks:
            max_y = max([p.height for p in event.peaks])

            for peak in event.peaks:
                x = peak.index_of_maximum * self.samples_to_us
                y = peak.height
                plt.hlines(y, peak.left * self.samples_to_us,
                           peak.right * self.samples_to_us)
                if log_y_axis:
                    ytext = y + (1-y/max_y)
                    arrowprops = None
                else:
                    ytext =  y + (max_y - y) * (
                                0.05 + 0.2 * random.random()
                             )
                    arrowprops = dict(arrowstyle="fancy",
                                      fc="0.6", ec="none",
                                      connectionstyle="angle3,"
                                                      "angleA=0,"
                                                      "angleB=-90")
                plt.annotate('%s:%s' % (peak.type, int(peak.area)),
                             xy=(x, y),
                             xytext=(x, ytext),
                             arrowprops = arrowprops)
        if show_legend:
            legend = plt.legend(loc='upper left', prop={'size': 10})
            if legend and legend.get_frame():
                legend.get_frame().set_alpha(0.5)



class PlotSumWaveformLargestS2(PlotBase):

    def plot_event(self, event, show_legend=False):
        if not event.S2s():
            self.log.warning("Can't plot the largest S2: there aren't any S2s in this event.")
            plt.title('No S2 in event')
            return

        largest_s2 = sorted(event.S2s(), key=lambda x: x.area,
                            reverse=True)[0]

        pad = 200 if largest_s2.height > 100 else 50

        self.plot_waveform(event, left=largest_s2.left, right=largest_s2.right,
                           pad=pad, show_legend=show_legend, log_y_axis=self.config['log_scale_s2'])

        time = largest_s2.index_of_maximum * self.samples_to_us

        plt.title("S2 at %.1f us" % time)


class PlotSumWaveformLargestS1(PlotBase):

    def plot_event(self, event, show_legend=False):
        if not event.S1s():
            self.log.warning("Can't plot the largest S1: there aren't any S1s in this event.")
            plt.title('No S1 in event')
            return

        largest_s1 = sorted(event.S1s(), key=lambda x: x.area,
                            reverse=True)[0]

        self.plot_waveform(event, left=largest_s1.left, right=largest_s1.right,
                           pad=10, show_legend=show_legend, log_y_axis=self.config['log_scale_s1'])

        time = largest_s1.index_of_maximum * self.samples_to_us

        plt.title("S1 at %.1f us" % time)


class PlotSumWaveformEntireEvent(PlotBase):

    def plot_event(self, event, show_legend=True):
        self.plot_waveform(event, show_peaks=True, show_legend=show_legend,
                           log_y_axis=self.config['log_scale_entire_event'])


class PlottingHitPattern(PlotBase):
    def substartup(self):
        self.pmts_top = self.config['pmts_top']
        self.pmts_bottom = self.config['pmts_bottom']
        self.pmt_locations = self.config['pmt_locations']

    def _plot(self, peak, ax, pmts, size_multiplication_factor=1):
        area = []
        points = []
        for pmt in pmts:
            area.append(peak.area_per_pmt[pmt])
            points.append((self.pmt_locations[pmt]['x'],
                           self.pmt_locations[pmt]['y']))

        area = np.array(area)*size_multiplication_factor
        total_area = np.sum(area)
        c = ax.scatter(*zip(*points), s=5000*area/total_area, c=area, cmap=plt.cm.hot)
        c.set_alpha(0.75)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def plot_event(self, event, show=['S1','S2'], show_dominant_array_only=True, subplots_to_use=None):
        if subplots_to_use is None:
            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
            subplots_to_use = [ax1, ax2, ax3, ax4]

        for peak_type, dominant_array in (
                ('S1', 'bottom'), ('S2', 'top')):
            if peak_type not in show:
                continue
            peak_list = getattr(event, peak_type+'s')()
            if peak_list and len(peak_list) >= 1:
                peak = peak_list[0]
                for array in ('top', 'bottom'):
                    if show_dominant_array_only and array != dominant_array:
                        continue
                    ax = subplots_to_use.pop(0)
                    ax.set_title('%s %s' % (peak_type, array))

                    self._plot(peak, ax, getattr(self, 'pmts_%s'%array))

            
            
class PlotChannelWaveforms(PlotBase): # user sets variables xlim, ylim for 3D plot
    """Plot an event

    Will make a fancy plot with lots of arrows etc of a summed waveform
    """

    def set_y_values(self, value, n_times):
        y = []
        
        for int_i in range(0,n_times):
            y.append(value)
            
        return y

    def plot_event(self, event):
        ylim_channel_start = 1
        ylim_channel_end  = 190
        xlim_time_start = 0
        xlim_time_end = 40000

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        for channel, occurrences in event.occurrences.items(): # is dictionary
            print ('channel number:', channel, ' has this number of occurrences',  len(occurrences))           
            for start_index, waveform in occurrences: # is list
                if channel > ylim_channel_start and channel < ylim_channel_end:
                    x = np.array(np.linspace(start_index,start_index+len(waveform)-1,
                                             len(waveform)))
                    ax.plot(x, self.set_y_values(channel,len(waveform)),
                            zs=-1*waveform, zdir='z', label=str(channel))
                    
        ax.set_xlabel('Time [10ns]')
        ax.set_xlim3d(xlim_time_start, xlim_time_end)
        
        ax.set_ylabel('Channel number')
        ax.set_ylim3d(ylim_channel_start, ylim_channel_end)
        
        ax.set_zlabel('Pulse height')


class PlotEventSummary(PlotBase):
    def plot_event(self, event, plt=plt):
        """
        Combines several plots into a nice summary plot
        """

        rows = 2
        cols = 4

        plt.figure(figsize=(4 * cols, 4 * rows))
        title = 'Event %s from %s -- recorded at %s UTC, %09dns' % (
                        event.event_number, event.dataset_name,
                        time.strftime("%Y/%m/%d, %H:%M:%S", time.gmtime(event.start_time/10**9)),
                         event.start_time%(10**9))
        plt.suptitle(title, fontsize=18)

        plt.subplot2grid((rows, cols), (0, 0))
        q = PlotSumWaveformLargestS1(self.config)
        q.plot_event(event, show_legend=False)

        ax = plt.subplot2grid((rows, cols), (0, 3))
        ax.yaxis.set_label_position("right")
        ax.yaxis.set_ticks_position("right")
        q = PlotSumWaveformLargestS2(self.config)
        q.plot_event(event, show_legend=False)

        q = PlottingHitPattern(self.config)
        q.plot_event(event, show_dominant_array_only=True, subplots_to_use = [
                plt.subplot2grid((rows, cols), (0, 1)),
                plt.subplot2grid((rows, cols), (0, 2))
            ])

        plt.subplot2grid((rows, cols), (rows - 1, 0), colspan=cols)
        q = PlotSumWaveformEntireEvent(self.config)
        q.plot_event(event, show_legend=True)

        plt.tight_layout()
        plt.subplots_adjust(top=0.88)

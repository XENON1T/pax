"""Interactive plotting

Use matplotlib to display various things about the event.
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import random
import numpy as np
import os
import time

from pax import plugin, units


class PlotBase(plugin.OutputPlugin):

    def startup(self):
        if self.config['output_dir'] is not None:
            self.output_dir = self.config['output_dir']
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
        else:
            self.output_dir = None

        self.size_multiplier = self.config.get('size_multiplier', 1)
        self.horizontal_size_multiplier = self.config.get('horizontal_size_multiplier', 1)

        self.skip_counter = 0
        # Convert times in samples to times in us: need to know how many samples / us
        self.samples_to_us = self.config['sample_duration'] / units.us

        self.peak_colors = {
            's1':   'blue',
            's2':   'green',
            'lone_pulse': '0.75'
        }

        self.substartup()

    def substartup(self):
        pass

    def write_event(self, event):
        # Should we plot or skip?
        if self.skip_counter:
            self.skip_counter -= 1
            self.log.debug("Skipping this event due to plot_every = %s. "
                           "Skip counter at %s" % (self.config['plot_every'], self.skip_counter))
            return

        self.plot_event(event)
        self.finalize_plot(event.event_number)

    def plot_event(self, event):
        raise NotImplementedError()

    def finalize_plot(self, event_number=0):
        """Finalize plotting, send to screen/file, then closes plot properly (avoids runtimewarning / memory leak).
        """
        if self.output_dir:
            plt.savefig(self.output_dir + '/%06d.png' % event_number)
        else:
            plt.show(block=False)
            self.log.info("Hit enter to continue...")
            input()
        plt.close()
        self.skip_counter = self.config['plot_every'] - 1

    def plot_waveform(self, event, left=0, right=None, pad=0,
                      show_peaks=False, show_legend=True, log_y_axis=False, scale=1):
        """
        Plot part of an event's sum waveform. Defined in base class to ensure a uniform style
        """
        if right is None:
            right = event.length() - 1
        lefti = max(0, left - pad)
        righti = min(right + pad, event.length() - 1)
        nsamples = 1 + righti - lefti
        dt = event.sample_duration * units.ns

        xlabels = np.arange(lefti * dt / units.us,  # 10+ labels will be cut later, prevents off by one errors
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
            waveform = event.get_sum_waveform(w['internal_name'])
            plt.plot(xlabels,
                     (waveform.samples[lefti:righti + 1] + y_offset) * scale,
                     label=w['plot_label'],
                     drawstyle=w.get('drawstyle'))
        if log_y_axis:
            plt.ylim((0.9, plt.ylim()[1]))

        if show_peaks and event.peaks:
            self.color_peak_ranges(event)
            # Don't rely on max_y to actually be the max height,
            # Possibly the waveform is highest outside a peak.
            max_y = max([p.height for p in event.peaks])

            for peak in event.peaks:
                if peak.type == 'lone_pulse':
                    continue
                textcolor = 'black' if peak.detector == 'tpc' else 'red'

                x = peak.index_of_maximum * self.samples_to_us
                y = peak.height
                if log_y_axis:
                    y += 1
                    ytext = y
                    arrowprops = None
                    # max ensures ytext comes out positive (well, if y is)
                    # and that the text is never below the peak
                    # if max_y != 0:
                    #    ytext = max(y, y * (3-2*y/max_y))
                    # else:
                else:
                    ytext = max(y, y + (max_y - y) * (0.05 + 0.2 * random.random()))
                    arrowprops = dict(arrowstyle="fancy",
                                      fc="0.6", ec="none",
                                      connectionstyle="angle3,"
                                                      "angleA=0,"
                                                      "angleB=-90")
                plt.hlines(y, (peak.left - 1) * self.samples_to_us,
                           peak.right * self.samples_to_us)
                plt.annotate('%s:%s' % (peak.type, int(peak.area)),
                             xy=(x, y),
                             xytext=(x, ytext),
                             arrowprops=arrowprops,
                             color=textcolor)

        if show_legend:
            legend = plt.legend(loc='upper left', prop={'size': 10})
            if legend and legend.get_frame():
                legend.get_frame().set_alpha(0.5)

    def color_peak_ranges(self, event):
        # Separated so PlotChannelWaveforms2D can also call it
        for peak in event.peaks:
            shade_color = self.peak_colors.get(peak.type, 'gray') if peak.detector == 'tpc' else 'red'
            plt.axvspan((peak.left - 1) * self.samples_to_us,
                        peak.right * self.samples_to_us,
                        color=shade_color,
                        alpha=0.2)


class PlotSumWaveformLargestS2(PlotBase):

    def plot_event(self, event, show_legend=False):
        if not event.S2s():
            self.log.debug("Can't plot the largest S2: there aren't any S2s in this event.")
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
            self.log.debug("Can't plot the largest S1: there aren't any S1s in this event.")
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
        self.channels_top = self.config['channels_top']
        self.channels_bottom = self.config['channels_bottom']
        self.pmt_locations = self.config['pmt_locations']

    def _plot(self, peak, ax, pmts):
        area = []
        points = []
        for pmt in pmts:
            area.append(peak.area_per_channel[pmt])
            points.append((self.pmt_locations[pmt]['x'],
                           self.pmt_locations[pmt]['y']))

        area = np.array(area)
        total_area = np.sum(area[area > 0])
        c = ax.scatter(*zip(*points), s=5000 * area / total_area, c=area, cmap=plt.cm.hot)
        c.set_alpha(0.75)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    def plot_event(self, event, show=('S1', 'S2'), show_dominant_array_only=True, subplots_to_use=None):
        if subplots_to_use is None:
            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
            subplots_to_use = [ax1, ax2, ax3, ax4]

        for peak_type, dominant_array in (
                ('S1', 'bottom'), ('S2', 'top')):
            if peak_type not in show:
                continue
            peak_list = getattr(event, peak_type + 's')()
            if peak_list and len(peak_list) >= 1:
                peak = peak_list[0]
                for array in ('top', 'bottom'):
                    if show_dominant_array_only and array != dominant_array:
                        continue
                    ax = subplots_to_use.pop(0)
                    ax.set_title('%s %s' % (peak_type, array))

                    self._plot(peak, ax, getattr(self, 'channels_%s' % array))


class PlotChannelWaveforms3D(PlotBase):  # user sets variables xlim, ylim for 3D plot

    """Plot an event

    Will make a fancy '3D' plot with different y positions for all channels. User sets variables for plot.
    """

    def plot_event(self, event):
        dt = self.config['sample_duration']
        # top : 1-98, bottom : 99-178, Top Shield Array: PMTs 179..210, Bottom Shield Array: PMTs 211..242
        ylim_channel_start = 0
        ylim_channel_end = 242
        xlim_time_start = 0  # s1: 205.5 , s2: 260
        xlim_time_end = 400  # [us] S1: 206.5, s2: 270

        # [pe/(bin*V)] = time_resolution [s/bin]
        #               / ( resistor [V*s/C] * gain [] * amplification [] * electric_charge [C/pe] )
        # For Xenon100 (gain 2e6, amplification 10, resistor 50, time resolution 10ns/bin),
        # this gives 62.4 pe/(bin*V), so 1 pe/bin is 16.0 mV; 1 mV is 0.0624 pe/bin.

        fig = plt.figure(figsize=(self.size_multiplier * 4, self.size_multiplier * 2))
        ax = fig.gca(projection='3d')

        # Plot each individual occurrence
        global_max_amplitude = 0
        for oc in event.occurrences:
            start_index = oc.left
            channel = oc.channel
            end_index = oc.right

            # Takes only channels in the range we want to plot
            if not ylim_channel_start <= channel <= ylim_channel_end:
                continue

            # Take only occurrences that start in the time window
            # -- But don't you also want occurrences which start outside, but end inside the window?
            if not xlim_time_start * 100 <= start_index <= xlim_time_end * 100:
                continue

            waveform = event.channel_waveforms[channel, start_index: end_index + 1]
            if self.config['log_scale']:
                # TODO: this will still give nan's if waveform drops below 1 pe_nominal / bin...
                waveform = np.log10(1 + waveform)

            # We need to keep track of this, apparently gca can't scale itself?
            global_max_amplitude = max(np.max(waveform), global_max_amplitude)

            ax.plot(
                np.linspace(start_index, start_index + len(waveform) - 1, len(waveform)) * dt / units.us,
                channel * np.ones(len(waveform)),
                zs=waveform,
                zdir='z',
                label=str(channel)
            )

        # Plot the sum waveform
        lefti = xlim_time_start * 100
        righti = xlim_time_end * 100
        waveform = event.get_sum_waveform('uS2').samples[lefti:righti]
        scale = global_max_amplitude / np.max(waveform)
        time = np.array(np.linspace(lefti, righti, righti - lefti))
        ax.plot(
            time / 100,
            (ylim_channel_end + 1) * np.ones(len(waveform)),  # time to micro sec
            zs=waveform * scale,
            zdir='z',
            label=str(channel)
        )

        ax.set_xlabel('Time [$\mu$s]')
        ax.set_xlim3d(xlim_time_start, xlim_time_end)

        ax.set_ylabel('Channel number')
        ax.set_ylim3d(ylim_channel_start, ylim_channel_end)

        zlabel = 'Pulse height [pe_nominal / %d ns]' % (self.config['sample_duration'] / units.ns)
        if self.config['log_scale']:
            zlabel = 'Log10 1 + ' + zlabel
        ax.set_zlabel(zlabel)
        ax.set_zlim3d(0, global_max_amplitude)

        plt.tight_layout()


class PlotChannelWaveforms2D(PlotBase):

    """ Plots the occurrences in each channel, like like PlotChannelWaveforms3D, but seen from above

    Circles in the bottom subplot show when individual photo-electrons arrived in each channel .
    Circle color indicates log(peak amplitude / noise amplitude), size indicates peak integral.
    For large peaks you see no dots; single-channel peakfinding is skipped for performance reasons.

    Some channels are grayed out: these are excluded from low-energy peakfinding because they show an
    unusual rate of lone pulses or pure-noise occurrences in this event.
    """

    def plot_event(self, event):
        time_scale = self.config['sample_duration'] / units.us

        # TODO: change from lines to squares
        for oc in event.occurrences:
            if oc.height is None:
                # Maybe gain was 0 or something
                # TODO: plot these too, in a different color
                continue

            # Choose a color for this occurrence based on amplitude
            color_factor = np.clip(np.log10(oc.height) / 2, 0, 1)

            plt.gca().add_patch(Rectangle((oc.left * time_scale, oc.channel), oc.length * time_scale, 1,
                                          facecolor=plt.cm.gnuplot2(color_factor),
                                          edgecolor='none',
                                          alpha=(0.1 if event.is_channel_bad[oc.channel] else 0.7)))

        # Plot the channel peaks as dots
        # All these for loops are slow -- hope we get by-column access some time
        self.log.debug('Plotting channel peaks...')

        for p in event.all_channel_peaks:
            if p.noise_sigma == 0:
                # What perfect world are you living in? WaveformSimulator!
                color_factor = 1
            else:
                color_factor = min(max(p.height / (20 * p.noise_sigma), 0), 1)
            plt.scatter([(0.5 + p.index_of_maximum) * time_scale],
                        [0.5 + p.channel],
                        c=(color_factor, 0, (1 - color_factor)),
                        s=10 * min(10, p.area),
                        edgecolor=(0.5 * color_factor, 0, 0.5 * (1 - color_factor)),
                        alpha=(0.1 if event.is_channel_bad[p.channel] else 1.0))

        # Plot the bottom/top/veto boundaries
        # Assumes the detector names' lexical order is the same as the channel order!
        channel_ranges = [
            ('top',     min(self.config['channels_top'])),
            ('bottom',  min(self.config['channels_bottom'])),
        ]
        for det, chs in self.config['channels_in_detector'].items():
            if det == 'tpc':
                continue
            channel_ranges.append((det, min(chs)))

        # Annotate the channel groups and boundaries
        for i in range(len(channel_ranges)):
            plt.plot(
                [0, event.length() * time_scale],
                [channel_ranges[i][1]] * 2,
                color='black', alpha=0.2)
            plt.text(
                0.03 * event.length() * time_scale,
                (channel_ranges[i][1] +
                    (channel_ranges[i + 1][1] if i < len(channel_ranges) - 1 else self.config['n_channels'])
                 ) / 2,
                channel_ranges[i][0])

        # Tell about the bad channels
        bad_channels = np.where(event.is_channel_bad)[0]
        if len(bad_channels):
            plt.text(0, 0, 'Bad channels: ' + ', '.join(map(str, sorted(bad_channels))), {'size': 8})

        # Color the peak ranges
        self.color_peak_ranges(event)

        # Make sure we always see all channels , even if there are few occurrences
        plt.xlim((0, event.length() * time_scale))
        plt.ylim((0, len(event.channel_waveforms)))

        plt.xlabel('Time (us)')
        plt.ylabel('PMT channel')
        plt.tight_layout()


class PlotEventSummary(PlotBase):

    def plot_event(self, event):
        """
        Combines several plots into a nice summary plot
        """

        rows = 3
        cols = 4
        if not self.config['plot_largest_peaks']:
            rows -= 1

        plt.figure(figsize=(self.horizontal_size_multiplier * self.size_multiplier * cols, self.size_multiplier * rows))
        title = 'Event %s from %s -- recorded at %s UTC, %09dns' % (
            event.event_number, event.dataset_name,
            time.strftime("%Y/%m/%d, %H:%M:%S", time.gmtime(event.start_time / 10 ** 9)),
            event.start_time % (10 ** 9))
        plt.suptitle(title, fontsize=18)

        if self.config['plot_largest_peaks']:

            self.log.debug("Plotting largest S1...")
            plt.subplot2grid((rows, cols), (0, 0))
            q = PlotSumWaveformLargestS1(self.config, self.processor)
            q.plot_event(event, show_legend=False)

            self.log.debug("Plotting largest S2...")
            ax = plt.subplot2grid((rows, cols), (0, 3))
            ax.yaxis.set_label_position("right")
            ax.yaxis.set_ticks_position("right")
            q = PlotSumWaveformLargestS2(self.config, self.processor)
            q.plot_event(event, show_legend=False)

            self.log.debug("Plotting hitpatterns...")
            q = PlottingHitPattern(self.config, self.processor)
            q.plot_event(event, show_dominant_array_only=True, subplots_to_use=[
                plt.subplot2grid((rows, cols), (0, 1)),
                plt.subplot2grid((rows, cols), (0, 2))
            ])

        self.log.debug("Plotting sum waveform...")
        sumw_ax = plt.subplot2grid((rows, cols), (rows - 2, 0), colspan=cols)
        q = PlotSumWaveformEntireEvent(self.config, self.processor)
        q.plot_event(event, show_legend=True)

        self.log.debug("Plotting channel waveforms...")
        plt.subplot2grid((rows, cols), (rows - 1, 0), colspan=cols, sharex=sumw_ax)
        q = PlotChannelWaveforms2D(self.config, self.processor)
        q.plot_event(event)

        plt.tight_layout()

        # Make some room for the title
        plt.subplots_adjust(top=1 - 0.12 * 4 / self.size_multiplier)

"""Interactive plotting

Use matplotlib to display various things about the event.
"""

import random
import os
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

# Init stuff for 3d plotting
# Please do not remove, although it appears to be unused, 3d plotting won't work without it
from mpl_toolkits.mplot3d import Axes3D     # noqa

from pax import plugin, units, utils


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
            'lone_hit': '0.75'
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
            if self.config['plot_format'] == 'pdf':
                plt.savefig(self.output_dir + '/%06d.pdf' % event_number, format='pdf')
            else:
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

        xvalues = np.arange(lefti * dt / units.us,  # 10+ labels will be cut later, prevents off by one errors
                            (10 + righti) * dt / units.us,
                            dt / units.us)
        xvalues = xvalues[:nsamples]

        plt.autoscale(True, axis='y', tight=True)
        plt.xlim((xvalues[0], xvalues[-1]))

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
            plt.plot(xvalues,
                     (waveform.samples[lefti:righti + 1] + y_offset) * scale,
                     label=w['plot_label'],
                     color=w.get('color', 'gray'),
                     drawstyle=w.get('drawstyle'),
                     alpha=w.get('alpha', 1))
        if log_y_axis:
            plt.ylim((0.9, plt.ylim()[1]))

        self.draw_trigger_mark(1 if log_y_axis else 0)

        if show_peaks and event.peaks:
            self.color_peak_ranges(event)
            # Don't rely on max_y to actually be the max height,
            # Possibly the waveform is highest outside a peak.
            max_y = max([p.height for p in event.peaks])

            for peak in event.peaks:
                if peak.type == 'lone_hit':
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
            shade_color = self.peak_colors.get(peak.type, 'black') if peak.detector == 'tpc' else 'red'
            plt.axvspan((peak.left - 1) * self.samples_to_us,
                        peak.right * self.samples_to_us,
                        color=shade_color,
                        alpha=0.1)

    def draw_trigger_mark(self, y=0):
        # Draw a marker (orange star) indicating the event's trigger time
        trigger_time = self.config.get('trigger_time_in_event', None)
        if trigger_time is not None:
            plt.gca().plot([trigger_time / units.us], [y], '*', color='orange', markersize=10)


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
        # Grab PMT numbers and x, y locations
        self.locations = np.array([[self.config['pmt_locations'][ch]['x'],
                                    self.config['pmt_locations'][ch]['y']]
                                   for ch in range(self.config['n_channels'])])
        self.pmts = {array: self.config['channels_%s' % array] for array in ('top', 'bottom')}

    def _plot(self, peak, ax, array, show_colorbar=False):
        # Plot the hitpattern
        pmts_hit = [ch for ch in self.pmts[array] if peak.does_channel_contribute[ch]]
        q = ax.scatter(*self.locations[pmts_hit].T,
                       c=peak.area_per_channel[pmts_hit],
                       norm=matplotlib.colors.LogNorm(),
                       vmin=1e-1,
                       vmax=1e4,
                       alpha=0.4,
                       s=250)

        if show_colorbar:
            plt.gcf().colorbar(mappable=q, label='Area of hits in PMT', ax=ax,
                               orientation='horizontal')

        # Plot the PMT numbers
        for pmt in pmts_hit:
            ax.text(self.locations[pmt, 0], self.locations[pmt, 1], pmt,
                    fontsize=6, va='center', ha='center', color='black')

        # Plot the detector radius
        r = self.config['tpc_radius']
        ax.add_artist(plt.Circle((0, 0), r, edgecolor='black', fill=None))
        ax.set_xlim(-1.2*r, 1.2*r)
        ax.set_ylim(-1.2*r, 1.2*r)

    def plot_event(self, event, show=('S1', 'S2'), show_dominant_array_only=True, subplots_to_use=None):
        if subplots_to_use is None:
            f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
            subplots_to_use = [ax1, ax2, ax3, ax4]

        for peak_type, dominant_array in (('S1', 'bottom'), ('S2', 'top')):
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
                    self._plot(peak, ax, array, show_colorbar=False)


class PlotChannelWaveforms3D(PlotBase):

    """Plot the waveform of several channels in 3D, with different y positions for all channels.
    """

    def plot_event(self, event):
        dt = self.config['sample_duration']

        # Configurable limits
        channels_start = self.config.get('channel_start', min(self.config['channels_in_detector']['tpc']))
        channels_end = self.config.get('channel_start', max(self.config['channels_in_detector']['tpc']))
        t_start = self.config.get('t_start', 0 * units.us)
        t_end = self.config.get('t_end', event.duration())

        fig = plt.figure(figsize=(self.size_multiplier * 4, self.size_multiplier * 2))
        ax = fig.gca(projection='3d')

        global_max_amplitude = 0
        for pulse in event.pulses:
            # Take only channels in the range we want to plot
            if not channels_start <= pulse.channel <= channels_end:
                continue

            # Take only pulses that fall (at least partially) in the time window we want to plot
            if pulse.right * dt < t_start or pulse.left * dt > t_end:
                continue

            w = self.config['digitizer_reference_baseline'] + pulse.baseline - pulse.raw_data.astype(np.float64)
            w *= utils.adc_to_pe(self.config, pulse.channel, use_reference_gain=True)
            if self.config['log_scale']:
                # This will still give nan's if waveform drops below 1 pe_nominal / bin...
                # TODO: So... will it crash? or just fall outside range?
                w = np.log10(1 + w)

            # We need to keep track of this, apparently gca can't scale itself?
            global_max_amplitude = max(np.max(w), global_max_amplitude)

            ax.plot(
                np.linspace(pulse.left, pulse.right, pulse.length) * dt / units.us,
                pulse.channel * np.ones(pulse.length),
                zs=w,
                zdir='z',
                label=str(pulse.channel)
            )

        # Plot the sum waveform
        w = event.get_sum_waveform('tpc').samples[int(t_start / dt):int(t_end / dt) + 1]
        ax.plot(
            np.linspace(t_start, t_end, len(w)) / units.us,
            (channels_end + 1) * np.ones(len(w)),
            zs=w * global_max_amplitude / np.max(w),
            zdir='z',
            label='Tpc'
        )

        ax.set_xlabel('Time [$\mu$s]')
        ax.set_xlim3d(t_start / units.us, t_end / units.us)

        ax.set_ylabel('Channel number')
        ax.set_ylim3d(channels_start, channels_end)

        zlabel = 'Pulse height [pe_nominal / %d ns]' % (self.config['sample_duration'] / units.ns)
        if self.config['log_scale']:
            zlabel = 'Log10 1 + ' + zlabel
        ax.set_zlabel(zlabel)
        ax.set_zlim3d(0, global_max_amplitude)

        plt.tight_layout()


class PlotChannelWaveforms2D(PlotBase):

    """ Plots the pulses in each channel, like like PlotChannelWaveforms3D, but seen from above

    Circles in the bottom subplot show when individual photo-electrons arrived in each channel .
    Circle color indicates log(peak amplitude / noise amplitude), size indicates peak integral.
    """

    def plot_event(self, event):
        dt = self.config['sample_duration']
        time_scale = dt / units.us

        # TODO: change from lines to squares
        for pulse in event.pulses:
            if pulse.maximum is None:
                # Maybe gain was 0 or something
                # TODO: plot these too, in a different color
                continue

            # Choose a color for this pulse based on amplitude
            # color_factor = np.clip(np.log10(oc.height) / 2, 0, 1)
            color_factor = 0

            plt.gca().add_patch(Rectangle((pulse.left * time_scale, pulse.channel - 0.5), pulse.length * time_scale, 1,
                                          facecolor=plt.cm.gnuplot2(color_factor),
                                          edgecolor='none',
                                          alpha=0.5))

        # Plot the channel peaks as dots
        self.log.debug('Plotting channel peaks...')

        result = []
        for hit in event.all_hits:
            color_factor = min(hit.height / hit.noise_sigma, 15)/15
            result.append([
                (0.5 + hit.center / dt) * time_scale,                  # X
                hit.channel,                                           # Y
                color_factor,                                          # Color (in [0,1] -> [Blue, Red])
                10 * min(10, hit.area),                                # Size
                (1 if hit.is_rejected else 0),                         # Is rejected? If not, will make green
            ])
        if len(result) != 0:
            rgba_colors = np.zeros((len(result), 4))
            result = np.array(result).T
            rgba_colors[:, 0] = (1 - result[4]) * result[2]                     # R
            rgba_colors[:, 1] = result[4]                                       # G
            rgba_colors[:, 2] = (1 - result[4]) * (1 - result[2])               # B
            rgba_colors[:, 3] = 1                                               # A
            plt.scatter(result[0], result[1], c=rgba_colors, s=result[3], edgecolor=None, lw=0)

        # Plot the bottom/top/veto boundaries
        # Assumes the detector names' lexical order is the same as the channel order!
        channel_ranges = []
        if self.config['channels_top']:
            channel_ranges.append(('top',
                                   min(self.config['channels_top']),
                                   np.mean(self.config['channels_top'])))
        if self.config['channels_bottom']:
            channel_ranges.append(('bottom',
                                   min(self.config['channels_bottom']),
                                   np.mean(self.config['channels_bottom'])))
        for det, chs in self.config['channels_in_detector'].items():
            if det == 'tpc':
                continue
            channel_ranges.append((det, min(chs), np.mean(chs)))

        # Annotate the channel groups and boundaries
        for i in range(len(channel_ranges)):
            plt.plot(
                [0, event.length() * time_scale],
                [channel_ranges[i][1]] * 2,
                color='black', alpha=0.2)
            plt.text(
                0.03 * event.length() * time_scale,
                channel_ranges[i][2] + 0.5,  # add 0.5 for better alignment for small TPCs. Irrelevant for big ones
                channel_ranges[i][0])

        # Information about suspicious channels
        suspicious_channels = np.where(event.is_channel_suspicious)[0]
        if len(suspicious_channels):
            plt.text(
                0,
                0,
                'Suspicious channels (# of hits rejected): ' + ', '.join([
                    '%s (%s)' % (ch, event.n_hits_rejected[ch]) for ch in suspicious_channels
                ]),
                {'size': 8})

        # Color the peak ranges, place the trigger mark
        self.color_peak_ranges(event)
        self.draw_trigger_mark(0)

        # Make sure we always see all channels
        plt.xlim((0, event.length() * time_scale))
        plt.ylim((0, self.config['n_channels']))

        plt.xlabel('Time (us)')
        plt.ylabel('PMT channel')
        plt.gca().invert_yaxis()    # To ensure top channels (low numbers) appear above bottom channels (high numbers)


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

        # Show the title
        # If there is no trigger time, show the event start time in the title
        trigger_time_ns = (event.start_time + self.config.get('trigger_time_in_event', 0)) / units.ns
        title = 'Event %s from %s -- recorded at %s UTC, %09d ns' % (
            event.event_number, event.dataset_name,
            time.strftime("%Y/%m/%d, %H:%M:%S", time.gmtime(trigger_time_ns / 10 ** 9)),
            trigger_time_ns % (10 ** 9))
        plt.suptitle(title, fontsize=18)

        if self.config['plot_largest_peaks']:

            self.log.debug("Plotting largest S1...")
            plt.subplot2grid((rows, cols), (0, 0))
            q = PlotSumWaveformLargestS1(self.config, self.processor)
            q.plot_event(event, show_legend=False)

            self.log.debug("Plotting largest S2...")
            plt.subplot2grid((rows, cols), (0, 1))
            q = PlotSumWaveformLargestS2(self.config, self.processor)
            q.plot_event(event, show_legend=False)

            self.log.debug("Plotting hitpatterns...")
            q = PlottingHitPattern(self.config, self.processor)
            q.plot_event(event, show_dominant_array_only=True, subplots_to_use=[
                plt.subplot2grid((rows, cols), (0, 2)),
                plt.subplot2grid((rows, cols), (0, 3))
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

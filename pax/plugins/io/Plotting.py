"""Interactive plotting

Use matplotlib to display various things about the event.
"""

import random
import os
import datetime
import textwrap

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from six.moves import input

# Init stuff for 3d plotting
# Please do not remove, although it appears to be unused, 3d plotting won't work without it
from mpl_toolkits.mplot3d import Axes3D     # noqa

from pax import plugin, units, datastructure, dsputils


def epoch_to_human_time(timestamp):
    # Unfortunately the python datetime
    return datetime.datetime.fromtimestamp(timestamp / units.s).strftime("%Y/%m/%d, %H:%M:%S")


class PlotBase(plugin.OutputPlugin):
    block_view = False
    hates_tight_layout = False

    def startup(self):
        if self.config['output_name'] != 'SCREEN':
            self.output_dir = self.config['output_name']
            if not os.path.exists(self.output_dir):
                os.self.makedirs(self.output_dir)
        else:
            self.output_dir = None

        self.size_multiplier = self.config.get('size_multiplier', 1)
        self.horizontal_size_multiplier = self.config.get('horizontal_size_multiplier', 1)
        if 'block_view' in self.config:
            self.block_view = self.config['block_view']

        self.skip_counter = 0
        # Convert times in samples to times in us: need to know how many samples / us
        self.samples_to_us = self.config['sample_duration'] / units.us

        self.peak_colors = {
            's1':   'blue',
            's2':   'green',
            'lone_hit': '0.75'
        }

        # Grab PMT numbers and x, y locations
        self.pmts = {array: self.config['channels_%s' % array] for array in ('top', 'bottom')}
        self.pmt_locations = np.array([[self.config['pmts'][ch]['position']['x'],
                                        self.config['pmts'][ch]['position']['y']]
                                       for ch in range(self.config['n_channels'])])

        self.hitpattern_limits = (1e-1, 1e4)
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

        self.trigger_time_ns = event.start_time / units.ns
        self.plot_event(event)
        self.finalize_plot(event.event_number)

    def plot_event(self, event):
        raise NotImplementedError()

    def finalize_plot(self, event_number=0):
        """Finalize plotting, send to screen/file, then closes plot properly (avoids runtimewarning / memory leak).
        """
        if not self.hates_tight_layout:
            plt.tight_layout()
        if self.output_dir:
            if self.config['plot_format'] == 'pdf':
                plt.savefig(self.output_dir + '/%06d.pdf' % event_number, format='pdf')
            else:
                plt.savefig(self.output_dir + '/%06d.png' % event_number)
        else:
            plt.show(block=self.block_view)
            if not self.block_view:
                self.log.info("Hit enter to continue...")
                input()
        plt.close()
        self.skip_counter = self.config['plot_every'] - 1

    def plot_waveform(self, event,
                      left=0, right=None, pad=0,
                      show_peaks=False, show_legend=True, log_y_axis=False,
                      scale=1,
                      ax=None):
        """
        Plot part of an event's sum waveform. Defined in base class to ensure a uniform style
        """
        if ax is None:
            ax = plt.gca()
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

        ax.set_xlim((xvalues[0], xvalues[-1]))

        if log_y_axis:
            ax.set_yscale('log')
            ax.set_ylabel('Amplitude + 1 (pe/bin)')
            y_offset = 1
        else:
            ax.set_ylabel('Amplitude (pe/bin)')
            y_offset = 0
        ax.set_xlabel('Time (us)')

        y_min = 1
        y_max = 0

        for w in self.config['waveforms_to_plot']:
            waveform = event.get_sum_waveform(w['internal_name'])
            wv = (waveform.samples[lefti:righti + 1] + y_offset) * scale
            y_min = min(y_min, np.min(wv))
            y_max = max(y_max, np.max(wv))
            ax.plot(xvalues,
                    wv,
                    label=w['plot_label'],
                    color=w.get('color', 'gray'),
                    drawstyle=w.get('drawstyle'),
                    alpha=w.get('alpha', 1))
        if log_y_axis:
            ax.set_ylim((0.9, plt.ylim()[1]))

        self.draw_trigger_signals(event, y=1 if log_y_axis else 0, ax=ax)

        if show_peaks and event.peaks:
            self.color_peak_ranges(event, ax=ax)
            # Don't rely on max_y to actually be the max height,
            # Possibly the waveform is highest outside a peak.
            max_y = max([p.height for p in event.peaks])

            for peak in event.peaks:
                if self.config.get('hide_peak_info') or peak.type == 'lone_hit':
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
                    arrowprops = dict(arrowstyle="simple",
                                      fc='black', ec="none", alpha=0.3,
                                      connectionstyle="angle3,"
                                                      "angleA=0,"
                                                      "angleB=-90")
                ax.annotate('%s:%0.1f' % (peak.type, peak.area),
                            xy=(x, y),
                            xytext=(x, ytext),
                            fontsize=8,
                            arrowprops=arrowprops,
                            color=textcolor)

        ax.set_ylim(y_min, y_max * 1.1)

        if show_legend:
            legend = ax.legend(loc='upper left', prop={'size': 10})
            if legend and legend.get_frame():
                legend.get_frame().set_alpha(0.5)

    def color_peak_ranges(self, source, ax=None):
        if ax is None:
            ax = plt.gca()
        if isinstance(source, datastructure.Event):
            peaks = source.peaks
        else:
            peaks = [source]
        # Separated so PlotChannelWaveforms2D can also call it
        for peak in peaks:
            if self.config.get('hide_peak_info'):
                continue
            shade_color = self.peak_colors.get(peak.type, 'black') if peak.detector == 'tpc' else 'red'
            ax.axvspan((peak.left - 1) * self.samples_to_us,
                       peak.right * self.samples_to_us,
                       color=shade_color,
                       alpha=0.1)

    def draw_trigger_signals(self, event, y=0, ax=None):
        """Draw markers (stars) indicating the signals found by the trigger"""
        if self.config.get('hide_trigger_info'):
            return
        if ax is None:
            ax = plt.gca()

        self.log.debug("Drawing %d trigger signals" % len(event.trigger_signals))
        self.log.debug(str(event.trigger_signals['left_time'] / units.s))

        for s in event.trigger_signals:
            x = s['left_time'] / units.us
            level = s['n_pulses']       # TODO: change to contributing channels or area
            signal_type = s['type']
            size_factor = 0.5 + 0.5 * np.log10(level)

            ax.annotate(str(level), (x, y), fontsize=12 * size_factor,
                        # xytext=(0, 0), textcoords='offset points',
                        alpha=0.5)

            ax.plot([x], [y],
                    marker='*' if s['trigger'] else 'o',
                    color={1: 'blue', 2: 'green', 3: 'orange'}.get(signal_type, 'gray'),
                    linewidth=0, alpha=0.5,
                    markersize=10 * size_factor)

    def plot_hitpattern(self, peak, array='top', ax=None):
        if ax is None:
            ax = plt.gca()
        pmts_hit = [ch for ch in self.pmts[array] if peak.does_channel_contribute[ch]]
        q = ax.scatter(*self.pmt_locations[pmts_hit].T,
                       c=peak.area_per_channel[pmts_hit],
                       norm=matplotlib.colors.LogNorm(),
                       vmin=self.hitpattern_limits[0],
                       vmax=self.hitpattern_limits[1],
                       alpha=0.4,
                       s=250)

        # Plot the PMT numbers
        for pmt in pmts_hit:
            ax.text(self.pmt_locations[pmt, 0], self.pmt_locations[pmt, 1], pmt,
                    fontsize=8 if peak.is_channel_saturated[pmt] else 6,
                    va='center', ha='center',
                    color='white' if peak.is_channel_saturated[pmt] else 'black')

        # Plot the detector radius
        r = self.config['tpc_radius']
        ax.add_artist(plt.Circle((0, 0), r, edgecolor='black', fill=None))
        ax.set_xlim(-1.2*r, 1.2*r)
        ax.set_ylim(-1.2*r, 1.2*r)
        # Sets labels on the hitpattern plot
        # 'labelpad' adjusts position
        ax.set_xlabel('x (cm)', labelpad=1)
        ax.set_ylabel('y (cm)', labelpad=1)

        return q


class PlotSumWaveformMainS2(PlotBase):

    def plot_event(self, event, show_legend=False):
        peak = event.main_s2
        if peak is None:
            self.log.debug("Can't plot the largest S2: there aren't any S2s in this event.")
            plt.title('No S2 in event')
            return

        self.plot_waveform(event, left=peak.left, right=peak.right,
                           pad=200 if peak.height > 100 else 50,
                           show_legend=show_legend,
                           log_y_axis=self.config['log_scale_s2'])
        plt.title("S2 at %.1f us" % (peak.index_of_maximum * self.samples_to_us))


class PlotSumWaveformMainS1(PlotBase):

    def plot_event(self, event, show_legend=False):
        peak = event.main_s1
        if peak is None:
            self.log.debug("Can't plot the largest S1: there aren't any S1s in this event.")
            plt.title('No S1 in event')
            return

        self.plot_waveform(event, left=peak.left, right=peak.right,
                           pad=10,
                           show_legend=show_legend,
                           log_y_axis=self.config['log_scale_s1'])
        plt.title("S1 at %.1f us" % (peak.index_of_maximum * self.samples_to_us))


class PlotSumWaveformEntireEvent(PlotBase):

    def plot_event(self, event, show_legend=True, ax=None):
        self.plot_waveform(event,
                           show_peaks=True,
                           show_legend=show_legend,
                           log_y_axis=self.config['log_scale_entire_event'],
                           ax=ax)


class PlottingHitPattern(PlotBase):

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
                    self.plot_hitpattern(peak=peak, array=array, ax=ax)


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
            w *= dsputils.adc_to_pe(self.config, pulse.channel, use_reference_gain=True)
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


class PlotChannelWaveforms2D(PlotBase):

    """ Plots the pulses in each channel, like like PlotChannelWaveforms3D, but seen from above

    Circles in the bottom subplot show when individual photo-electrons arrived in each channel .
    Circle color indicates log(peak amplitude / noise amplitude), size indicates peak integral.
    """

    def plot_event(self, source, ax=None, pad=0, event=None, show_channel_group_labels=True):
        """Source can be Event or Peak, if peak, must also pass event"""

        dt = self.config['sample_duration']
        self.time_scale = time_scale = dt / units.us

        if isinstance(source, datastructure.Event):
            hits = source.all_hits
            pulses = source.pulses
            xlims = (0, source.length() * time_scale)
            event = source
        elif isinstance(source, datastructure.Peak):
            hits = source.hits
            pulses = [event.pulses[i] for i in np.unique([h.found_in_pulse for h in hits])]
            xlims = ((source.left - pad) * time_scale,
                     (source.right + pad) * time_scale)
        else:
            raise ValueError("Unknown source: %s" % source)

        if ax is None:
            ax = plt.gca()

        # TODO: change from lines to squares
        for pulse in pulses:
            if pulse.maximum is None:
                # Maybe gain was 0 or something
                # TODO: plot these too, in a different color
                continue

            # Choose a color for this pulse based on amplitude
            # color_factor = np.clip(np.log10(oc.height) / 2, 0, 1)
            color_factor = 0

            ax.add_patch(Rectangle((pulse.left * time_scale, pulse.channel - 0.5), pulse.length * time_scale, 1,
                                   facecolor=plt.cm.gnuplot2(color_factor),
                                   edgecolor='none',
                                   alpha=0.5))

        # Plot the channel peaks as dots
        self.log.debug('Plotting channel peaks...')

        # TODO: vectorize
        result = []
        for hit in hits:
            color_factor = min(hit['height'] / hit['noise_sigma'], 15)/15
            result.append([
                (0.5 + hit['center'] / dt) * time_scale,                  # X
                hit['channel'],                                           # Y
                color_factor,                                          # Color (in [0,1] -> [Blue, Red])
                10 * min(10, hit['area']),                                # Size
                (1 if hit['is_rejected'] else 0),                         # Is rejected? If so, will make green
            ])
        if len(result) != 0:
            rgba_colors = np.zeros((len(result), 4))
            result = np.array(result).T
            rgba_colors[:, 0] = (1 - result[4]) * result[2]                     # R
            rgba_colors[:, 1] = result[4]                                       # G
            rgba_colors[:, 2] = (1 - result[4]) * (1 - result[2])               # B
            rgba_colors[:, 3] = 1                                               # A
            ax.scatter(result[0], result[1], c=rgba_colors, s=result[3], edgecolor=None, lw=0)

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
        if show_channel_group_labels:
            for i in range(len(channel_ranges)):
                ax.axhline(channel_ranges[i][1], color='black', alpha=0.2)
                ax.text(
                    0.03 * xlims[1],
                    channel_ranges[i][2] + 0.5,  # add 0.5 for better alignment for small TPCs. Irrelevant for big ones
                    channel_ranges[i][0])

        self.color_peak_ranges(source, ax=ax)

        # Make sure we always see all channels
        ax.set_ylim((-0.5, self.config['n_channels'] + 0.5))
        ax.set_xlim(*xlims)

        ax.set_xlabel('Time (us)')
        ax.set_ylabel('PMT channel')
        ax.invert_yaxis()    # To ensure top channels (low numbers) appear above bottom channels (high numbers)


class PlotEventSummary(PlotBase):
    hates_tight_layout = True

    def plot_event(self, event):
        """
        Combines several plots into a nice summary plot
        """

        rows = 3
        cols = 4
        if not self.config['plot_largest_peaks']:
            rows -= 1

        plt.figure(figsize=(self.horizontal_size_multiplier * self.size_multiplier * cols,
                            self.size_multiplier * rows))

        # Show the title
        # If there is no trigger time, show the event start time in the title
        title = 'Event %s from %s\nRecorded at %s UTC, %09d ns' % (
            event.event_number, event.dataset_name,
            epoch_to_human_time(self.trigger_time_ns),
            self.trigger_time_ns % (units.s))
        plt.suptitle(title, fontsize=18)

        if self.config['plot_largest_peaks']:

            self.log.debug("Plotting largest S1...")
            plt.subplot2grid((rows, cols), (0, 0))
            q = PlotSumWaveformMainS1(self.config, self.processor)
            q.plot_event(event, show_legend=False)

            self.log.debug("Plotting largest S2...")
            plt.subplot2grid((rows, cols), (0, 1))
            q = PlotSumWaveformMainS2(self.config, self.processor)
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

        # Make some room for the title: need to call tight_layout first...
        plt.tight_layout()
        plt.subplots_adjust(top=1 - 0.12 * 4 / self.size_multiplier)


class PeakViewer(PlotBase):
    block_view = True
    hates_tight_layout = True
    max_characters = 70
    _buttons = []       # Keeps references to the buttons we create alive. Needed for matplotlib 1.5.

    def substartup(self):
        # Default setting for first peak to show
        self.starting_peak = self.config.get('starting_peak', 'largest')
        # Dictionary mapping event number to left boundary index of desired starting peak
        self.starting_peak_per_event = self.config.get('starting_peak_per_event', {})
        # Convert keys = event numbers to integers -- necessary since JSON only allows string dictionary keys...
        self.starting_peak_per_event = {int(k): v for k, v in self.starting_peak_per_event.items()}

    def plot_event(self, event):
        self.event = event
        self.fig = plt.figure(figsize=(15, 12))

        x_pad = 0.04
        y_pad = 0.05
        extra_y_pad_top = -0.02
        extra_x_pad_left = 0.02
        y_sep_middle = 0.06
        start_x = x_pad + extra_x_pad_left
        start_y = y_pad
        full_x = 1 - x_pad - start_x
        full_y = 1 - y_pad - start_y - y_sep_middle - extra_y_pad_top
        row_y = full_y / 4
        column_x = full_y / 4

        title = 'Event %s from %s' % (event.event_number, event.dataset_name)
        plt.suptitle(title, fontsize=16, horizontalalignment='right', x=0.99)

        ##
        # Channel waveforms plot
        ##
        self.channel_wv_ax = plt.axes([start_x, start_y, full_x, row_y])
        q = PlotChannelWaveforms2D(self.config, self.processor)
        q.plot_event(event, ax=self.channel_wv_ax)
        self.chwvs_2s_time_scale = q.time_scale
        # TODO: Make top, bottom, veto yticks on right y axis?

        ##
        # Event waveform plot
        ##
        event_wv_ax = plt.axes([start_x, start_y + row_y, full_x, row_y],
                               sharex=self.channel_wv_ax)
        q = PlotSumWaveformEntireEvent(self.config, self.processor)
        q.plot_event(event, show_legend=True, ax=event_wv_ax)
        event_wv_ax.get_xaxis().set_visible(False)

        ##
        # Whereami plot
        ##
        whereami_height = 0.01
        self.whereami_ax = plt.axes([start_x, start_y + 2 * row_y, full_x, whereami_height],
                                    sharex=self.channel_wv_ax)
        self.whereami_ax.set_axis_off()

        ##
        # Event and peak text
        ##
        x_sep_text = 0.07
        y_sep_text = 0.05
        x = start_x + 2 * column_x + x_sep_text
        y = start_y + 4 * row_y + y_sep_middle - y_sep_text
        event_text = ''
        event_text += 'Event recorded at %s UTC, %09d ns\n' % (
            epoch_to_human_time(self.trigger_time_ns),
            self.trigger_time_ns % units.s)
        suspicious_channels = np.where(event.is_channel_suspicious)[0]
        event_text += 'Suspicious channels (# hits rejected):\n ' + ', '.join([
            '%s (%s)' % (ch, event.n_hits_rejected[ch]) for ch in suspicious_channels]) + '\n'
        self.fig.text(x, y, self.wrap_multiline(event_text, self.max_characters), verticalalignment='top')
        self.peak_text = self.fig.text(x, start_y + 3 * row_y + y_sep_middle, '', verticalalignment='top')

        ##
        # Peak hitpatterns
        ##
        y = start_y + 2 * row_y + y_sep_middle
        self.bot_hitp_ax = plt.axes([start_x, y, column_x, row_y])
        self.top_hitp_ax = plt.axes([start_x, y + row_y, column_x, row_y],
                                    sharex=self.bot_hitp_ax)
        self.top_hitp_ax.get_xaxis().set_visible(False)

        ##
        # Get the TPC peaks
        ##
        # Did the user specify a peak number? If so, get it
        self.peak_i = self.starting_peak_per_event.get(event.event_number, None)
        self.peaks = event.get_peaks_by_type(detector='tpc', sort_key='left', reverse=False)
        self.peaks = [p for p in self.peaks if p.type != 'lone_hit']
        if len(self.peaks) == 0:
            self.log.debug("No peaks in this event, will be a boring peakviewer plot...")
            return event

        ##
        # Peak Waveforms
        ##
        x_sep = 0.03
        x = start_x + column_x + x_sep
        self.peak_chwvs_ax = plt.axes([x, y, column_x - x_sep, row_y])
        self.peak_chwvs_ax.yaxis.tick_right()
        self.peak_chwvs_ax.yaxis.set_label_position("right")
        self.peak_sumwv_ax = plt.axes([x, y + row_y, column_x - x_sep, row_y],
                                      sharex=self.peak_chwvs_ax)
        self.peak_sumwv_ax.yaxis.tick_right()
        self.peak_sumwv_ax.yaxis.set_label_position("right")
        self.peak_sumwv_ax.get_xaxis().set_visible(False)
        q = PlotChannelWaveforms2D(self.config, self.processor)
        q.plot_event(event, ax=self.peak_chwvs_ax, show_channel_group_labels=False)

        ##
        # Buttons
        ##
        x = start_x + 2 * column_x + x_sep
        button_width = 0.08
        button_height = 0.03
        buttons_x_space = 0.04
        self.make_button([x, y, button_width, button_height],
                         'Prev peak', self.draw_prev_peak)
        self.make_button([x + button_width, y, button_width, button_height],
                         'Next peak', self.draw_next_peak)
        self.make_button([x + 2 * button_width + buttons_x_space, y, button_width, button_height],
                         'Main S1', self.draw_main_s1)
        self.make_button([x + 3 * button_width + buttons_x_space, y, button_width, button_height],
                         'Main S2', self.draw_main_s2)

        ##
        # Select and draw the desired peak
        ##
        if event.event_number in self.starting_peak_per_event:
            # The user specified the left boundary of the desired starting peak
            desired_left = self.starting_peak_per_event[event.event_number]
            self.peak_i = np.argmin([np.abs(p.left - desired_left) for p in self.peaks])
            if self.peaks[self.peak_i].left != desired_left:
                self.log.warning('There is no (tpc, non-lone-hit) peak starting at index %d! '
                                 'Taking closest peak (%d-%d) instead.' % (desired_left,
                                                                           self.peaks[self.peak_i].left,
                                                                           self.peaks[self.peak_i].right))
            self.log.debug("Selected user-defined peak %d" % self.peak_i)
            self.draw_peak()

        elif self.starting_peak == 'first':
            self.peak_i = 0
            self.draw_peak()

        elif self.starting_peak == 'main_s1':
            self.draw_main_s1()

        elif self.starting_peak == 'main_s2':
            self.draw_main_s2()

        if self.peak_i is None:
            # Either self.starting_peak is 'largest' or one of the other peak selections failed
            # (e.g. couldn't draw the main s1 because there is no S1)
            # Just pick the largest peak (regardless of its type)
            self.peak_i = np.argmax([p.area for p in self.peaks])
            self.log.debug("Largest peak is %d (peak list runs from 0-%d)" % (self.peak_i, len(self.peaks)-1))
            self.draw_peak()

        # Draw the color bar for the top or bottom hitpattern plot (they share them)
        # In the rare case the top has no data points, take it from the bottom
        sc_for_hitp = self.top_hitp_sc if self.peaks[self.peak_i].area_fraction_top != 0 else self.bot_hitp_sc
        plt.colorbar(sc_for_hitp, ax=[self.top_hitp_ax, self.bot_hitp_ax])

    def draw_peak(self):
        if self.peak_i is None:
            raise RuntimeError("self.peak_i is None: peak viewer bug!")

        # Select the requested peak
        self.peak_i = self.peak_i % len(self.peaks)
        peak = self.peaks[self.peak_i]

        # Update the whereami
        self.whereami_ax.cla()
        prev_xlim = self.channel_wv_ax.get_xlim()
        self.whereami_ax.scatter(peak.hit_time_mean / units.us, 0, marker='v', s=50, color='black')
        self.whereami_ax.set_axis_off()
        self.whereami_ax.set_xlim(*prev_xlim)

        # Update the hitpatterns
        self.top_hitp_ax.cla()
        self.bot_hitp_ax.cla()
        self.top_hitp_sc = self.plot_hitpattern(peak=peak, ax=self.top_hitp_ax, array='top')
        self.bot_hitp_sc = self.plot_hitpattern(peak=peak, ax=self.bot_hitp_ax, array='bottom')

        # Update peak waveforms
        peak_padding = self.config.get('peak_padding_samples', 30)
        self.plot_waveform(self.event, left=peak.left, right=peak.right,
                           pad=peak_padding, show_legend=False, log_y_axis=False, ax=self.peak_sumwv_ax)
        self.peak_chwvs_ax.set_xlim((peak.left - peak_padding) * self.chwvs_2s_time_scale,
                                    (peak.right + peak_padding) * self.chwvs_2s_time_scale)
        self.peak_sumwv_ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4))

        # Update peak text

        peak_text = 'Selected peak: %s at %d-%d, mean hit time %0.2fus\n' % (
            peak.type,
            peak.left,
            peak.right,
            peak.hit_time_mean / units.us,)
        peak_text += 'Area: %0.2f pe, contained in %d hits in %d channels\n' % (
            peak.area, len(peak.hits), len(peak.contributing_channels))
        peak_text += 'Fraction in top: %0.2f\n' % peak.area_fraction_top
        peak_text += 'Peak widths: hit time std = %dns,\n' \
                     ' 50%% area range = %dns, 90%% area range = %dns\n' % (peak.hit_time_std,
                                                                            peak.range_area_decile[5],
                                                                            peak.range_area_decile[9])
        try:
            pos = peak.get_position_from_preferred_algorithm(['PosRecTopPatternFit', 'PosRecNeuralNet',
                                                              'PosRecRobustWeightedMean', 'PosRecWeightedSum',
                                                              'PosRecMaxPMT'])
        except ValueError:
            peak_text += "Position reconstruction failed!"
        else:
            peak_text += 'Chi2Gamma: %0.1f, /area_top: %0.1f, /channels_top: %0.1f\n' % (
                pos.goodness_of_fit,
                pos.goodness_of_fit / (peak.area_fraction_top * peak.area
                                       if peak.area_fraction_top != 0 else float('nan')),
                pos.goodness_of_fit / (peak.n_contributing_channels_top
                                       if peak.n_contributing_channels_top != 0 else float('nan')))
        peak_text += 'Top spread: %0.1fcm, Bottom spread: %0.1fcm\n' % (peak.top_hitpattern_spread,
                                                                        peak.bottom_hitpattern_spread)
        pos3d = peak.get_reconstructed_position_from_algorithm('PosRecThreeDPatternFit')
        if pos3d is not None and not np.isnan(pos3d.x):
            peak_text += '3d position: x=%0.1f, y=%0.1f, z=%0.1f (all cm)' % (pos3d.x, pos3d.y, pos3d.z)

        self.peak_text.set_text(self.wrap_multiline(peak_text, self.max_characters))

        plt.draw()

    @staticmethod
    def wrap_multiline(text, max_characters):
        return "\n".join(["\n".join(textwrap.wrap(q, max_characters)) for q in text.splitlines()])

    def draw_next_peak(self):
        self.peak_i += 1
        self.draw_peak()

    def draw_prev_peak(self):
        self.peak_i -= 1
        self.draw_peak()

    def draw_main_s1(self):
        peak = self.event.main_s1
        if peak is None:
            self.log.info("This event has no S1s.")
            return
        self.peak_i = self.peaks.index(peak)
        self.draw_peak()

    def draw_main_s2(self):
        peak = self.event.main_s2
        if peak is None:
            self.log.info("This event has no S2s.")
            return
        self.peak_i = self.peaks.index(peak)
        self.draw_peak()

    def make_button(self, rect, label, on_click):
        button_axes = plt.axes(rect)
        button = plt.Button(button_axes, label)
        button.on_clicked(lambda _mpl_event: on_click())
        self._buttons.append(button)

import numpy as np

from pax import plugin, datastructure, utils


class MeanShift(plugin.TransformPlugin):

    """Clusters individual channel peaks into groups, and labels them as noise / lone_pulse / unknown
    'unknown' means an S1 or S2, which will be decided by a later plugin
    """

    def startup(self):
        self.dt = self.config['sample_duration']
        self.cluster_separation_length = self.config['cluster_separation_time']
        self.n_channels = self.config['n_channels']

    def transform_event(self, event):

        # Handle each detector separately
        for detector in self.config['channels_in_detector'].keys():
            self.log.debug("Clustering channel peaks in data from %s" % detector)
            peaks = []      # Superfluous, while loop is always run once... but pycharm complains if we omit

            # Hmzz, python has no do_while, so..
            redo_classification = True
            while redo_classification:
                redo_classification = False
                peaks = []
                dark_count = {}

                # Get all single-pe data in a list of dicts, sorted by index_of_maximum
                spes = sorted([p for p in event.all_channel_peaks if p.channel in self.config['channels_in_detector'][detector] and (not self.config['exclude_bad_channels'] or not event.is_channel_bad[p.channel])],  # noqa, we're replacing this soon anyways
                              key=lambda s: s.index_of_maximum)
                self.log.debug("Found %s channel peaks" % len(spes))

                if not spes:
                    break

                # Cluster the single-pes in groups separated by >= self.cluster_separation_length
                cluster_indices = utils.cluster_by_diff([s.index_of_maximum * self.dt for s in spes],
                                                        self.cluster_separation_length, return_indices=True)
                self.log.debug("Made %s clusters" % len(cluster_indices))

                # Each cluster becomes a peak
                # Compute basic properties, check for too many lone pulses per channel
                for ci in cluster_indices:
                    peak = datastructure.Peak({
                        'channel_peaks':            [spes[cidx] for cidx in ci],
                        'detector':                 detector,
                        'area_per_channel':         np.zeros(self.n_channels),
                        'does_channel_contribute':  np.zeros(self.n_channels, dtype='bool'),
                        'does_channel_have_noise':  np.zeros(self.n_channels, dtype='bool'),
                    })

                    # Compute contributing and noise channels - needed for bad channel rejection

                    # Contributing channels are in the detector and have an spe
                    # (not in a bad channel, but those spes have already been filtered out)
                    for s in peak.channel_peaks:
                        peak.does_channel_contribute[s.channel] = True

                    if peak.number_of_contributing_channels == 0:
                        raise RuntimeError(
                            "Every peak should have at least one contributing channel... what's going on?")

                    # Find how many channels show some data, but no spe
                    coincident_occurrences = event.get_occurrences_between(peak.left, peak.right, strict=False)
                    for oc in coincident_occurrences:
                        ch = oc.channel
                        if not peak.does_channel_contribute[ch]:
                            peak.does_channel_have_noise[ch] = True

                    # Classification for noise and lone_pulse peaks
                    if peak.number_of_noise_channels > 2 * peak.number_of_contributing_channels:
                        peak.type = 'noise'

                    elif peak.number_of_contributing_channels == 1:
                        peak.type = 'lone_pulse'
                        channel = peak.channel_peaks[0].channel
                        dark_count[channel] = dark_count.get(channel, 0) + 1

                    else:
                        # Proper peak, classification done later
                        peak.type = 'unknown'

                    peaks.append(peak)

                # Look for channels with abnormal dark rate
                for ch, dc in dark_count.items():
                    if dc > self.config['maximum_lone_pulses_per_channel']:
                        self.log.debug(
                            "Channel %s shows an abnormally high lone pulse rate (%s): marked as bad." % (ch, dc))
                        event.is_channel_bad[ch] = True
                        if self.config['exclude_bad_channels']:
                            redo_classification = True
                            self.log.debug("Clustering has to be redone!!")

            # Add the peaks to the datastructure
            event.peaks.extend(peaks)

        return event

class HitGap(plugin.TransformPlugin):

    """Clusters individual channel peaks into groups, and labels them as noise / lone_pulse / unknown
    'unknown' means an S1 or S2, which will be decided by a later plugin
    """

    def startup(self):
        self.dt = self.config['sample_duration']
        self.cluster_separation_length = self.config['cluster_separation_time']
        self.n_channels = self.config['n_channels']

    def transform_event(self, event):

        # Handle each detector separately
        for detector in self.config['channels_in_detector'].keys():
            self.log.debug("Clustering channel peaks in data from %s" % detector)
            peaks = []      # Superfluous, while loop is always run once... but pycharm complains if we omit

            # Hmzz, python has no do_while, so..
            redo_classification = True
            while redo_classification:
                redo_classification = False
                peaks = []
                dark_count = {}

                # Get all single-pe data in a list of dicts, sorted by index_of_maximum
                spes = sorted([p for p in event.all_channel_peaks if p.channel in self.config['channels_in_detector'][detector] and (not self.config['exclude_bad_channels'] or not event.is_channel_bad[p.channel])],  # noqa, we're replacing this soon anyways
                              key=lambda s: s.index_of_maximum)
                self.log.debug("Found %s channel peaks" % len(spes))

                if not spes:
                    break

                # Cluster the single-pes in groups separated by >= self.cluster_separation_length
                cluster_indices = utils.cluster_by_diff([s.index_of_maximum * self.dt for s in spes],
                                                        self.cluster_separation_length, return_indices=True)
                self.log.debug("Made %s clusters" % len(cluster_indices))

                # Each cluster becomes a peak
                # Compute basic properties, check for too many lone pulses per channel
                for ci in cluster_indices:
                    peak = datastructure.Peak({
                        'channel_peaks':            [spes[cidx] for cidx in ci],
                        'detector':                 detector,
                        'area_per_channel':         np.zeros(self.n_channels),
                        'does_channel_contribute':  np.zeros(self.n_channels, dtype='bool'),
                        'does_channel_have_noise':  np.zeros(self.n_channels, dtype='bool'),
                    })

                    # Compute contributing and noise channels - needed for bad channel rejection

                    # Contributing channels are in the detector and have an spe
                    # (not in a bad channel, but those spes have already been filtered out)
                    for s in peak.channel_peaks:
                        peak.does_channel_contribute[s.channel] = True

                    if peak.number_of_contributing_channels == 0:
                        raise RuntimeError(
                            "Every peak should have at least one contributing channel... what's going on?")

                    # Find how many channels show some data, but no spe
                    coincident_occurrences = event.get_occurrences_between(peak.left, peak.right, strict=False)
                    for oc in coincident_occurrences:
                        ch = oc.channel
                        if not peak.does_channel_contribute[ch]:
                            peak.does_channel_have_noise[ch] = True

                    # Classification for noise and lone_pulse peaks
                    if peak.number_of_noise_channels > 2 * peak.number_of_contributing_channels:
                        peak.type = 'noise'

                    elif peak.number_of_contributing_channels == 1:
                        peak.type = 'lone_pulse'
                        channel = peak.channel_peaks[0].channel
                        dark_count[channel] = dark_count.get(channel, 0) + 1

                    else:
                        # Proper peak, classification done later
                        peak.type = 'unknown'

                    peaks.append(peak)

                # Look for channels with abnormal dark rate
                for ch, dc in dark_count.items():
                    if dc > self.config['maximum_lone_pulses_per_channel']:
                        self.log.debug(
                            "Channel %s shows an abnormally high lone pulse rate (%s): marked as bad." % (ch, dc))
                        event.is_channel_bad[ch] = True
                        if self.config['exclude_bad_channels']:
                            redo_classification = True
                            self.log.debug("Clustering has to be redone!!")

            # Add the peaks to the datastructure
            event.peaks.extend(peaks)

        return event
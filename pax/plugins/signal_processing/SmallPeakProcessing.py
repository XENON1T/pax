import numpy as np
from pax import plugin, datastructure, dsputils


class ClusterAndClassifySmallPeaks(plugin.TransformPlugin):

    def startup(self):
        self.dt = dt = self.config['digitizer_t_resolution']
        self.cluster_separation_length = self.config['cluster_separation_time']
        self.classification_mode = self.config['classification_mode']
        self.channels_in_detector = {
            'tpc':  self.config['pmts_top'] | self.config['pmts_bottom'],
        }
        for det, chs in self.config['external_detectors'].items():
            self.channels_in_detector[det] = chs

    def transform_event(self, event):

        # Handle each detector separately
        for detector in self.channels_in_detector.keys():
            self.log.debug("Clustering and classifying channel peaks in data from %s" % detector)
            peaks = []      # Superfluous, while loop is always run once... but pycharm complains if we omit

            # Hmzz, python has no do_while, so..
            redo_classification = True
            while redo_classification:
                redo_classification = False
                peaks = []
                dark_count = {}

                # Get all single-pe data in a list of dicts, sorted by index_of_maximum
                spes = sorted([
                    p for p in event.all_channel_peaks
                    if p.channel in self.channels_in_detector[detector]
                    and (not self.config['exclude_bad_channels'] or not event.is_channel_bad[p.channel])
                ], key=lambda s: s.index_of_maximum)
                self.log.debug("Found %s channel peaks" % len(spes))

                if not spes:
                    break

                # Cluster the single-pes in groups separated by >= self.cluster_separation_length
                cluster_indices = dsputils.cluster_by_diff([s.index_of_maximum * self.dt for s in spes],
                                                           self.cluster_separation_length,
                                                           return_indices=True)
                self.log.debug("Made %s clusters" % len(cluster_indices))

                # Each cluster becomes a peak
                for ci in cluster_indices:
                    peak = datastructure.Peak({
                        'channel_peaks': [s for i, s in enumerate(spes) if i in ci],
                        'detector':      detector,
                    })
                    peak.left =  min([s.left  for s in peak.channel_peaks])
                    peak.right = max([s.right for s in peak.channel_peaks])
                    peak.area =  sum([s.area  for s in peak.channel_peaks])

                    # For backwards compatibility with plotting code
                    peak.index_of_maximum = int((peak.left + peak.right)/2)
                    peak.height = 0

                    # Contributing channels are in the detector and have an spe
                    # (not in a bad channel, but those spes have already been filtered out)
                    channels = np.arange(self.config['n_pmts'])
                    peak.does_channel_contribute = (np.in1d(channels,
                                                            # numpy doesn't like sets...
                                                            np.array(list(self.channels_in_detector[detector])))) & \
                                                   (np.in1d(channels, [s.channel for s in peak.channel_peaks]))

                    if peak.number_of_contributing_channels == 0:
                        print(peak.to_dict(), "\n\n")
                        for s in peak.channel_peaks:
                            print(s.to_dict(), "\n\n")
                        print(self.channels_in_detector[detector])
                        print(np.in1d(channels, self.channels_in_detector[detector]), "\n\n")
                        print(np.in1d(channels, [s.channel for s in peak.channel_peaks]), "\n\n")
                        raise RuntimeError(
                            "Every peak should have at least one contributing channel... what's going on?")

                    # Compute the area per pmt -- for the contributing channels only!
                    peak.area_per_pmt = np.zeros(len(channels))
                    for ch in channels:
                        if peak.does_channel_contribute[ch]:
                            peak.area_per_pmt[ch] = sum([s.area for s in peak.channel_peaks])

                    # Find how many channels show some data, but no spe
                    coincident_occurrences = event.occurrences_interval_tree.search(peak.left, peak.right, strict=False)
                    peak.does_channel_have_noise = (np.invert(peak.does_channel_contribute)) & \
                                                   (np.in1d(channels, [oc[2]['channel']
                                                                       for oc in coincident_occurrences]))

                    # Compute some quantities summarizing the timing distributing
                    times = [s.index_of_maximum * self.dt for s in peak.channel_peaks]
                    peak.mean_absolute_deviation = dsputils.mad(times)

                    # Simple ad-hoc classification

                    if peak.number_of_noise_channels > 2 * peak.number_of_contributing_channels:
                        peak.type = 'noise'

                    elif peak.number_of_contributing_channels == 1:
                        peak.type = 'lone_pulse'
                        channel = peak.channel_peaks[0].channel
                        dark_count[channel] = dark_count.get(channel, 0) + 1

                    else:
                        if peak.mean_absolute_deviation < 10:
                            peak.type = 's1'
                        else:
                            if peak.number_of_contributing_channels < 5:
                                peak.type = 'unknown'
                            else:
                                peak.type = 's2'

                    peaks.append(peak)

                # Look for channels with abnormal dark rate
                for ch, dc in dark_count.items():
                    if dc > self.config['maximum_lone_pulses_per_channel']:
                        self.log.debug(
                            "Channel %s shows an abnormally high lone pulse rate (%s): marked as bad." % (ch, dc))
                        event.is_channel_bad[ch] = True
                        if self.config['exclude_bad_channels']:
                            redo_classification = True
                            self.log.debug("Classification has to be redone!!")

            # Classification is now done, so add the peaks to the datastructure
            event.peaks.extend(peaks)

        return event
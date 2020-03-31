import numpy as np

from pax import plugin


class RejectNoiseHits(plugin.ClusteringPlugin):
    """Remove hits in channels with many lone hits from peaks without many other hits.
    Works with a penalty point system (see #126):
    * The more lone hits you see in a channel, the more points a channel gets.
    * Channels with more than a (fairly low) number of penalty points are called suspicious.
      For these, we'll test every peak as follows:
      * A peak scores points for every pe in non-suspicious channels.
      * Hits in channels with more penalty points than the peak's score are removed from the peak.
    """

    def startup(self):
        self.base_penalties = {int(k): v for k, v in self.config.get('base_penalties', {}).items()}

    def transform_event(self, event):
        # Penalty for each noise pulse
        penalty_per_ch = event.noise_pulses_in * self.config['penalty_per_noise_pulse']

        # Penalty for each lone hit
        lone_hits = event.get_peaks_by_type(desired_type='lone_hit', detector='all')
        self.log.debug("This event has %d lone hits" % len(lone_hits))
        for lone_hit_peak in lone_hits:
            if lone_hit_peak.left < 1e5:
                # only count lone-hits before trigger positions
                channel = lone_hit_peak.hits[0]['channel']
                event.lone_hits_per_channel_before[channel] += 1
                penalty_per_ch[channel] += self.config['penalty_per_lone_hit']
        # Add base penalties
        for channel, penalty in self.base_penalties.items():
            penalty_per_ch[channel] += penalty

        # Which channels are suspicious?
        suspicious_channels = np.where(penalty_per_ch >=
                                       self.config['penalty_geq_this_is_suspicious'])[0]
        event.is_channel_suspicious[suspicious_channels] = True

        peaks_to_delete = []
        rejected_hits = []
        for peak_i, peak in enumerate(event.peaks):

            suspicious_channels_in_peak = np.intersect1d(peak.contributing_channels, suspicious_channels)
            if len(suspicious_channels_in_peak) == 0:
                continue

            # Which channels should we reject in this peak?
            # First compute the 'witness area' for this peak: area not in suspicious channels
            # We reject channels whose penalty is larger than the witness area
            witness_area = np.sum(peak.area_per_channel[True ^ event.is_channel_suspicious])
            channels_to_reject = [ch for ch in suspicious_channels_in_peak if penalty_per_ch[ch] > witness_area]
            # self.log.debug('Witness area %s, penalties: %s' % (witness_area,
            #                                                    [penalty_per_ch[ch]
            #                                                     for ch in suspicious_channels_in_peak]))
            if len(channels_to_reject) == 0:
                continue

            # Keep only hits not in the channels_to_reject
            cut = np.in1d(peak.hits['channel'], channels_to_reject)
            for hit_i in np.where(cut)[0]:
                rejected_hits.append(peak.hits[hit_i])
                event.n_hits_rejected[peak.hits[hit_i]['channel']] += 1

            # Has the peak become empty? Then mark it for deletion.
            if np.all(cut):
                self.log.debug('Peak %d consists completely of rejected hits and will be deleted!' % peak_i)
                peaks_to_delete.append(peak_i)
                continue

            else:
                # Else replace the peak with a new peak containing only the remaining hits
                event.peaks[peak_i] = self.build_peak(hits=peak.hits[True ^ cut], detector=peak.detector)

        # Delete any peaks which have gone empty
        event.peaks = [p for i, p in enumerate(event.peaks) if i not in peaks_to_delete]

        # Count the remaining number of lone hits per channel
        for peak in event.peaks:
            if peak.n_contributing_channels == 1:
                event.lone_hits_per_channel[peak.lone_hit_channel] += 1

        # Rebuild the event.all_hits field.
        if len(rejected_hits):
            rejected_hits = np.array(rejected_hits)
            rejected_hits['is_rejected'] = True
            event.all_hits = np.concatenate([rejected_hits] + [p.hits for p in event.peaks])

        return event

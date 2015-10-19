import numpy as np

from pax import plugin
from pax import dsputils


class RejectNoiseHits(plugin.TransformPlugin):
    """Remove hits in channels with many lone hits from peaks without many other hits.
    Works with a penalty point system (see #126):
    * The more lone hits you see in a channel, the more points a channel gets.
    * Channels with more than a (fairly low) number of penalty points are called suspicious.
      For these, we'll test every peak as follows:
      * A peak scores points for every pe in non-suspicious channels.
      * Hits in channels with more penalty points than the peak's score are removed from the peak.
    """

    def startup(self):
        self.detector_by_channel = dsputils.get_detector_by_channel(self.config)

    def transform_event(self, event):
        # Penalty for each noise pulse
        penalty_per_ch = event.noise_pulses_in * self.config['penalty_per_noise_pulse']

        # Penalty for each lone hit
        lone_hits = event.get_peaks_by_type(desired_type='lone_hit', detector='all')
        self.log.debug("This event has %d lone hits" % len(lone_hits))
        for lone_hit_peak in lone_hits:
            penalty_per_ch[lone_hit_peak.hits[0]['channel']] += self.config['penalty_per_lone_hit']

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
            peak.hits = peak.hits[True ^ cut]

            # Has the peak become empty? Then mark it for deletion.
            # We can't delete it now since we're iterating over event.peaks
            if len(peak.hits) == 0:
                self.log.debug('Peak %d consists completely of rejected hits and will be deleted!' % peak_i)
                peaks_to_delete.append(peak_i)
                continue

        # Delete any peaks which have gone empty
        event.peaks = [p for i, p in enumerate(event.peaks) if i not in peaks_to_delete]

        # Set the is_rejected flag for the hits in event.all_hits (needed for sumwaveform and plotting)
        # Assume found_in_pulse + left uniquely identifies each hit
        # First collect indices, then set, as advanced indexing would return a view
        if len(rejected_hits):
            rejected_hits = np.array(rejected_hits)
            rejected_hit_indices = np.where(np.in1d(event.all_hits['found_in_pulse'], rejected_hits['found_in_pulse']) &
                                            np.in1d(event.all_hits['left'], rejected_hits['left']))[0]
            for hit_i in rejected_hit_indices:
                event.all_hits[hit_i]['is_rejected'] = True

        return event

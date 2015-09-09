import numpy as np

from pax import plugin, utils


class RejectNoiseHits(plugin.TransformPlugin):
    """Remove hits in channels with many lone hits from peaks"""

    def startup(self):
        self.detector_by_channel = utils.get_detector_by_channel(self.config)

    def transform_event(self, event):
        # Penalty for each noise pulse
        penalty_per_ch = event.noise_pulses_in * self.config['penalty_per_noise_pulse']

        # Penalty for each lone hit
        lone_hits = event.get_peaks_by_type(desired_type='lone_hit', detector='all')
        self.log.debug("This event has %d lone hits" % len(lone_hits))
        for lone_hit_peak in lone_hits:
            penalty_per_ch[lone_hit_peak.hits[0].channel] += self.config['penalty_per_lone_hit']

        # Which channels are suspicious?
        suspicious_channels = np.where(penalty_per_ch >=
                                       self.config['penalty_geq_this_is_suspicious'])[0]
        event.is_channel_suspicious[suspicious_channels] = True

        peaks_to_delete = []
        for peak_i, peak in enumerate(event.peaks):

            # Area per channel must be computed here... unfortunate code duplication with basicProperti
            peak.area_per_channel = np.zeros(self.config['n_channels'], dtype='float64')
            for hit in peak.hits:
                peak.area_per_channel[hit.channel] += hit.area

            suspicious_channels_in_peak = np.intersect1d(peak.contributing_channels, suspicious_channels)
            if len(suspicious_channels_in_peak) == 0:
                continue
            self.log.debug("Peak %d has suspicious channels %s" % (peak_i, suspicious_channels_in_peak))

            # Which channels should we reject in this peak?
            # First compute the 'witness area' for this peak: area not in suspicious channels
            # We reject channels whose penalty is larger than the witness area
            witness_area = np.sum(peak.area_per_channel[True ^ event.is_channel_suspicious])
            channels_to_reject = [ch for ch in suspicious_channels_in_peak if penalty_per_ch[ch] > witness_area]
            self.log.debug('Witness area %s, penalties: %s' % (witness_area,
                                                               [penalty_per_ch[ch]
                                                                for ch in suspicious_channels_in_peak]))
            self.log.debug("Channels to reject: %s" % channels_to_reject)
            if len(channels_to_reject) == 0:
                continue

            # Keep only hits not in the channels_to_reject
            new_hits = []
            for h in peak.hits:
                if h.channel in channels_to_reject:
                    h.is_rejected = True
                    event.n_hits_rejected[h.channel] += 1
                else:
                    new_hits.append(h)
            peak.hits = new_hits

            # Has the peak become empty? Then mark it for deletion.
            # We can't delete it now since we're iterating over event.peaks
            if len(peak.hits) == 0:
                self.log.debug('Peak %d consists completely of rejected hits and will be deleted!' % peak_i)
                peaks_to_delete.append(peak_i)
                continue

        # Delete any peaks which have gone empty
        event.peaks = [p for i, p in enumerate(event.peaks) if i not in peaks_to_delete]

        return event

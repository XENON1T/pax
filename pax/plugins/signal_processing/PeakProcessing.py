"""
Plugins for computing properties of peaks that have been found
"""
import numpy as np
from pax import dsputils, plugin, datastructure
import math

class DeleteSmallPeaks(plugin.TransformPlugin):
    """Deletes low coincidence peaks, so the low-energy peakfinder can have a crack at them"""
    def transform_event(self, event):
        event.peaks = [p for p in event.peaks
                       if p.coincidence_level >= self.config['prune_if_coincidence_lower_than']
                       and p.area >= self.config['prune_if_area_lower_than']]
        return event


class ComputePeakWidths(plugin.TransformPlugin):
    """Does what it says on the tin"""

    def transform_event(self, event):
        for peak in event.peaks:

            # Check if peak is sane
            if peak.index_of_maximum < peak.left:
                self.log.debug("Insane peak %s-%s-%s, can't compute widths!" % (
                    peak.left, peak.index_of_maximum, peak.right))
                continue

            for width_name, conf in self.config['width_computations'].items():

                peak[width_name] = dsputils.width_at_fraction(
                    peak_wave=event.get_waveform(conf['waveform_to_use']).samples[peak.left : peak.right+1],
                    fraction_of_max=conf['fraction_of_max'],
                    max_idx=peak.index_of_maximum - peak.left,
                    interpolate=conf['interpolate'])

        return event




class ComputePeakAreasAndCoincidence(plugin.TransformPlugin):

    def transform_event(self, event):
        for peak in event.peaks:

            # Compute area in each channel
            peak.area_per_pmt = np.sum(event.pmt_waveforms[:, peak.left:peak.right+1], axis=1)

            # Determine which channels contribute to the peak's total area
            peak.contributing_pmts = np.array(
                np.where(peak.area_per_pmt >= self.config['minimum_area'])[0],
                dtype=np.uint16)

            # Compute the peak's areas
            # TODO: make excluding non-contributing pmts optional
            if peak.type == 'veto':
                peak.area = np.sum(peak.area_per_pmt[list(self.config['pmts_veto'])])
            else:
                if self.config['exclude_non_contributing_channels_from_area']:
                    peak.area = np.sum(peak.area_per_pmt[peak.contributing_pmts])
                else:
                    peak.area = np.sum(peak.area_per_pmt[peak.contributing_pmts])

        return event


class ComputePeakEntropies(plugin.TransformPlugin):
    #TODO: write tests


    def transform_event(self, event):
        for peak in event.peaks:

            peak_waveforms = event.pmt_waveforms[:, peak.left:peak.right+1]

            # Switching from entropy to kurtosis doesn't make it faster...
            # At head put:
            # import scipy
            # from scipy import stats
            # Here put:
            # peak.entropy_per_pmt = scipy.stats.kurtosis(peak_waveforms, axis=1)
            # continue

            if self.config['normalization_mode'] is 'abs':
                normalized = np.abs(peak_waveforms)
            elif self.config['normalization_mode'] is 'square':
                normalized = peak_waveforms**2
            else:
                raise ValueError(
                    'Invalid Configuration for ComputePeakEntropies: normalization_mode must be abs or square')

            # In the case of abs, we could re-use peak.area_per_pmt to normalize
            # This gains only a little bit of performance, and 'square' is what we use in Xenon100 anyway.
            # Note the use of np.newaxis to enable numpy broadcasting of the division
            normalized /= peak.area_per_pmt[:, np.newaxis]

            if self.config['only_for_contributing_pmts']:
                # Could this be vectorized better?
                # There is probably little use in restricting to a set of pmts before here,
                # the logarithm contains most of the work.
                peak.entropy_per_pmt = np.zeros(len(peak_waveforms))
                for pmt in peak.contributing_pmts:
                    peak.entropy_per_pmt[pmt] = -np.sum(normalized[pmt]*np.log(normalized[pmt]))
            else:
                peak.entropy_per_pmt = -np.sum(normalized*np.log(normalized), axis=1)

        return event



class IdentifyPeaks(plugin.TransformPlugin):

    def transform_event(self, event):

        unfiltered = event.get_waveform('tpc').samples
        for p in event.peaks:
            if p.type != 'unknown':
                # Some peakfinder forced the type. Fine, not my problem...
                continue
            # PLACEHOLDER:
            # if area in s1_half_area_in samples around max is > 50% of total area, christen as S1 candidate
            # if peak is smaller than s1_half_area_in, it is certainly an s1
            if p.right - p.left + 1 < self.config['s1_half_area_in']:
                p.type = 's1'
            else:
                left_samples = math.floor(self.config['s1_half_area_in']/2)
                right_samples = math.ceil(self.config['s1_half_area_in']/2)
                if np.sum(unfiltered[p.index_of_maximum - left_samples: p.index_of_maximum + right_samples]) > 0.5 * p.area:
                    p.type = 's1'
                    #self.log.debug("%s-%s-%s: S1" % (p.left, p.index_of_maximum, p.right))
                else:
                    p.type = 's2'
                    #self.log.debug("%s-%s-%s: S2" % (p.left, p.index_of_maximum, p.right))
        return event

#TODO: proper separation of TPC and veto
class IdentifySmallPeaks(plugin.TransformPlugin):

    def startup(self):
        dt = self.config['digitizer_t_resolution']
        self.cluster_separation_length = self.config['cluster_separation_time'] / dt

    def transform_event(self, event):
        bad_channels = []

        # Hmzz, python has no do_while, so..
        redo_classification = True
        while redo_classification:
            redo_classification = False

            # Get all single-pe data in a list of dicts, sorted by index_of_maximum
            spes = sorted([p.to_dict() for p in event.channel_peaks if p.channel not in bad_channels],
                          key=lambda x: x['index_of_maximum'])

            new_bad_channels = False

            times = [s['index_of_maximum'] for s in spes]
            assert(times == sorted(times))
            time_clusters = self.cluster_by_separation(times, self.cluster_separation_length)

            # Make a list of dicts of spe clusters (essentially a dataframe, but I want to do a for loop...)
            clusters = [{
                    'spes':         cluster_spes,
                    'n_spes':       len(cluster_spes),
                    # TODO: add extents of min/max peaks
                    'left':          spes[cluster_spes[0]]['left'],
                    'right':          spes[cluster_spes[-1]]['right'],
                    'type':         'unknown',
                } for cluster_spes in time_clusters]

            dark_count = {}

            for c in clusters:
                # Find how many occurences overlap with the cluster, so we know (roughly) how many channels show a waveform
                c['occurrences'] = len(event.occurrences_interval_tree.search(c['left'], c['right'], strict=False))

                if c['occurrences'] > 2 * c['n_spes']:
                    c['type'] = 'noise'

                elif c['n_spes'] == 1:
                    c['type'] = 'dark_pulse'
                    channel = spes[c['spes'][0]]['channel']
                    dark_count[channel] = dark_count.get(channel, 0) + 1

                else:
                    c['mad'] = dsputils.mad([times[i] for i in c['spes']])

                    if c['mad'] < 10:
                        c['type'] = 's1'
                    else:
                        if c['n_spes'] < 5:
                            c['type'] = 'unknown'
                        else:
                            c['type'] = 's2'

            # Look for channels with abnormal dark rate
            for ch, dc in dark_count.items():
                if dc > 3:
                    self.log.warning(
                        "Channel %s shows an abnormally high dark rate (%s): its spe pulses will be excluded" % (
                            ch, dc))
                    bad_channels.append(ch)
                    redo_classification = True

        # Add the peaks to the datastructure
        for c in clusters:
            # We need an index_of_maximum and height, these we can only get from the sum waveform]
            # TODO: do we really want these in datastructure even for peaks reconstructed from spes?
            sum_wave = event.get_waveform('tpc').samples[c['left'] : c['right'] + 1]
            max_idx = np.argmax(sum_wave)
            height = sum_wave[max_idx]
            event.peaks.append(datastructure.Peak({
                'index_of_maximum':     max_idx + c['left'],
                'height':               height,
                'left':                 c['left'],
                'right':                c['right'],
                'area':                 sum([spes[x]['area'] for x in c['spes']]),
                'contributing_pmts':    np.array(list(set([spes[x]['channel'] for x in c['spes']])), dtype=np.uint16),
                'area_per_pmt':         np.array([
                                            sum([spes[x]['area'] for x in c['spes'] if spes[x]['channel'] == ch])
                                            for ch in range(len(event.pmt_waveforms))]),
                'type':                 c['type'],
            }))

        event.bad_channels = np.array(bad_channels, dtype=np.int)

        return event

    @staticmethod
    def cluster_by_separation(x, separation_length):
        # Returns list of lists of indices of clusters in x
        # TODO: put in dsputils, test exhaustively
        if len(x) == 0:
            return []
        clusters = []
        current_cluster = []
        previous_t = x[0]
        for i, t in enumerate(x):
            if t - previous_t > separation_length:
                clusters.append(current_cluster)
                current_cluster = []
            current_cluster.append(i)
            previous_t = t
        clusters.append(current_cluster)
        return clusters
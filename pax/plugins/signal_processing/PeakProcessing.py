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

    def startup(self):
        self.central_width = round(self.config['central_area_region_width'] / self.config['digitizer_t_resolution'], 1)
        self.channels_in_detector = {
            'tpc':  self.config['pmts_top'] | self.config['pmts_bottom'],
        }
        for det, chs in self.config['external_detectors'].items():
            self.channels_in_detector[det] = chs

    def transform_event(self, event):
        for peak in event.peaks:

            # Compute area in each channel
            # Note this also computes the area for PMTs in other detectors!
            peak.area_per_pmt = np.sum(event.pmt_waveforms[:, peak.left:peak.right+1], axis=1)

            # Determine which channels contribute to the peak's total area
            peak.contributing_pmts = np.array([ch for ch in range(self.config['n_pmts']) if
                ch in self.channels_in_detector[peak.detector] and
                peak.area_per_pmt[ch] >= self.config['minimum_area']
            ], dtype=np.uint16)

            # Compute the peak's area
            peak.area = np.sum(peak.area_per_pmt[peak.contributing_pmts])

            # Compute the peak's central area (used for classification)
            peak.central_area = np.sum(
                event.pmt_waveforms[
                    peak.contributing_pmts,
                    max(peak.left, peak.index_of_maximum - math.floor(self.central_width/2)):
                    min(peak.right+1, peak.index_of_maximum + math.ceil(self.central_width/2))
                ],
                axis=(0, 1)
            )

        return event


class ComputePeakEntropies(plugin.TransformPlugin):
    #TODO: write tests

    def transform_event(self, event):
        for peak in event.peaks:

            peak_waveforms = event.pmt_waveforms[:, peak.left:peak.right+1]
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

            # Could this be vectorized better?
            # Restricting to contributing pmts is not optional: otherwise you'd include other detectors as well.
            peak.entropy_per_pmt = np.zeros(len(peak_waveforms))
            for pmt in peak.contributing_pmts:
                peak.entropy_per_pmt[pmt] = -np.sum(normalized[pmt]*np.log(normalized[pmt]))

        return event


class ClassifyBigPeaks(plugin.TransformPlugin):

    def transform_event(self, event):

        for peak in event.peaks:
            # PLACEHOLDER:
            # if central area is > central_area_ratio * total area, christen as S1 candidate
            if peak.central_area > self.config['central_area_ratio'] * peak.area:
                peak.type = 's1'
                #self.log.debug("%s-%s-%s: S1" % (peak.left, peak.index_of_maximum, peak.right))
            else:
                peak.type = 's2'
                #self.log.debug("%s-%s-%s: S2" % (peak.left, peak.index_of_maximum, peak.right))
        return event


#TODO: proper separation of TPC and veto
class ClusterAndClassifySmallPeaks(plugin.TransformPlugin):

    def startup(self):
        dt = self.config['digitizer_t_resolution']
        self.cluster_separation_length = self.config['cluster_separation_time'] / dt
        self.channels_in_detector = {
            'tpc':  self.config['pmts_top'] | self.config['pmts_bottom'],
        }
        for det, chs in self.config['external_detectors'].items():
            self.channels_in_detector[det] = chs

    def transform_event(self, event):

        # Handle each detector separately
        for detector in self.channels_in_detector.keys():

            # Hmzz, python has no do_while, so..
            redo_classification = True
            while redo_classification:
                redo_classification = False

                # Get all single-pe data in a list of dicts, sorted by index_of_maximum
                spes = sorted([
                    p.to_dict() for p in event.channel_peaks
                                if p.channel not in event.bad_channels
                                   and p.channel in self.channels_in_detector[detector]
                ], key=lambda x: x['index_of_maximum'])

                new_bad_channels = False

                times = [s['index_of_maximum'] for s in spes]
                assert(times == sorted(times))
                time_clusters = self.cluster_by_separation(times, self.cluster_separation_length)

                # Make a list of dicts of spe clusters (essentially a dataframe, but I want to do a for loop...)
                clusters = [{
                        'spes':         cluster_spes,
                        'n_spes':       len(cluster_spes),
                        # TODO: add extents of min/max peaks
                        'left':         spes[cluster_spes[0]]['left'],
                        'right':        spes[cluster_spes[-1]]['right'],
                        'type':         'unknown',
                    } for cluster_spes in time_clusters]

                dark_count = {}

                for c in clusters:
                    # Find how many channels show something (noise, bad)
                    # and how many are good & show photons
                    coincident_occurrences = event.occurrences_interval_tree.search(c['left'], c['right'], strict=False)
                    c['channels_with_something'] = set(self.channels_in_detector[detector]) & \
                                                   set([oc.data['channel'] for oc in coincident_occurrences])
                    c['channels_with_photons'] = set([spes[x]['channel'] for x in c['spes']])
                    c['mad'] = dsputils.mad([times[i] for i in c['spes']])

                    if len(c['channels_with_something']) > 2 * len(c['channels_with_photons']):
                        c['type'] = 'noise'

                    elif len(c['channels_with_photons']) == 1:
                        c['type'] = 'lone_pulse'
                        channel = spes[c['spes'][0]]['channel']
                        dark_count[channel] = dark_count.get(channel, 0) + 1

                    else:

                        if c['mad'] < 10:
                            c['type'] = 's1'
                        else:
                            if c['n_spes'] < 5:
                                c['type'] = 'unknown'
                            else:
                                c['type'] = 's2'

                # Look for channels with abnormal dark rate
                for ch, dc in dark_count.items():
                    if dc > self.config['maximum_lone_pulses_per_channel']:
                        self.log.debug(
                            "Channel %s shows an abnormally high lone pulse rate (%s): its spe pulses will be excluded" % (
                                ch, dc))
                        event.bad_channels.append(ch)
                        redo_classification = True

            # Classification is now done, so add the peaks to the datastructure
            for c in clusters:
                # We need an index_of_maximum and height, these we can only get from the sum waveform]
                # TODO: do we really want these in datastructure even for peaks reconstructed from spes?
                sum_wave = event.get_waveform(detector).samples[c['left'] : c['right'] + 1]
                max_idx = np.argmax(sum_wave)
                height = sum_wave[max_idx]
                event.peaks.append(datastructure.Peak({
                    'index_of_maximum':     max_idx + c['left'],
                    'height':               height,
                    'left':                 c['left'],
                    'right':                c['right'],
                    'area':                 sum([spes[x]['area'] for x in c['spes']]),
                    'contributing_pmts':    np.array(list(c['channels_with_photons']), dtype=np.uint16),
                    'area_per_pmt':         np.array([
                                                sum([spes[x]['area'] for x in c['spes'] if spes[x]['channel'] == ch])
                                                for ch in range(len(event.pmt_waveforms))]),
                    'type':                 c['type'],
                    'detector':             detector,
                    'mean_absolute_deviation': c['mad'],
                }))

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
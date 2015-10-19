import numpy as np

from pax import plugin, datastructure
from pax import dsputils


class GapSizeClustering(plugin.TransformPlugin):
    """Cluster individual hits into rough groups = Peaks separated by at least max_gap_size_in_cluster
    Also labels peak as 'lone_hit' if only one channel contributes
    """

    def startup(self):
        self.dt = self.config['sample_duration']
        self.n_channels = self.config['n_channels']
        self.detector_by_channel = dsputils.get_detector_by_channel(self.config)
        self.gap_threshold = self.config['max_gap_size_in_cluster'] / self.dt

    def transform_event(self, event):
        # Cluster hits in each detector separately
        # Assumes detector channel mappings are non-overlapping
        for detector, channels in self.config['channels_in_detector'].items():
            hits = event.all_hits[(event.all_hits['channel'] >= channels[0]) &
                                  (event.all_hits['channel'] <= channels[-1])]
            if len(hits) == 0:
                continue
            hits.sort(order='left')
            gaps = dsputils.gaps_between_hits(hits)
            cluster_indices = [0] + np.where(gaps > self.gap_threshold)[0].tolist() + [len(hits)]
            for i in range(len(cluster_indices) - 1):
                hits_in_this_peak = hits[cluster_indices[i]:cluster_indices[i + 1]]
                peak = datastructure.Peak(detector=detector,
                                          hits=hits_in_this_peak)

                # Area per channel must be computed here so RejectNoiseHits can use it
                # unfortunate code duplication with basicProperties!
                peak.area_per_channel = dsputils.count_hits_per_channel(peak, self.config,
                                                                        weights=hits_in_this_peak['area'])
                if np.sum(peak.area_per_channel > 0) == 1:
                    peak.type = 'lone_hit'

                event.peaks.append(peak)

        return event

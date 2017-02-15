import numpy as np

from pax import plugin, dsputils


class GapSizeClustering(plugin.ClusteringPlugin):
    """Cluster individual hits into rough groups = Peaks separated by at least max_gap_size_in_cluster
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
            hits.sort(order='left_central')
            gaps = dsputils.gaps_between_hits(hits)
            cluster_indices = [0] + np.where(gaps > self.gap_threshold)[0].tolist() + [len(hits)]
            for i in range(len(cluster_indices) - 1):
                hits_in_this_peak = hits[cluster_indices[i]:cluster_indices[i + 1]]
                event.peaks.append(self.build_peak(hits=hits_in_this_peak,
                                                   detector=detector))

        return event

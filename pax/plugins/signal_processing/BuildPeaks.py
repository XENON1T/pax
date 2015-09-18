from pax import plugin, datastructure, utils
import numpy as np


class GapSizeClustering(plugin.TransformPlugin):
    """Cluster individual hits into rough groups = Peaks separated by at least max_gap_size_in_cluster
    """

    def startup(self):
        self.dt = self.config['sample_duration']
        self.n_channels = self.config['n_channels']
        self.detector_by_channel = utils.get_detector_by_channel(self.config)
        self.gap_threshold = self.config['max_gap_size_in_cluster'] / self.dt

    def transform_event(self, event):
        # Sort hits by detector
        detectors = self.config['channels_in_detector'].keys()
        hits_per_detector = {detector: [] for detector in detectors}

        for hit in event.all_hits:
            hits_per_detector[self.detector_by_channel[hit.channel]].append(hit)

        # Cluster hits in each detector separately
        for detector, hits in hits_per_detector.items():
            if len(hits) == 0:
                continue
            hits.sort(key=lambda x: x.left)
            gaps = utils.gaps_between_hits(hits)
            cluster_indices = [0] + np.where(gaps > self.gap_threshold)[0].tolist() + [len(hits)]
            for i in range(len(cluster_indices) - 1):
                peak = datastructure.Peak(detector=detector,
                                          hits=hits[cluster_indices[i]:cluster_indices[i + 1]])
                event.peaks.append(peak)

        return event

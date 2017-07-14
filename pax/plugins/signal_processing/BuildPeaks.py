import numpy as np

from pax import plugin, dsputils


class GapSizeClustering(plugin.ClusteringPlugin):
    """Cluster individual hits into rough groups = Peaks separated by at least max_gap_size_in_cluster
    """

    def startup(self):
        self.dt = self.config['sample_duration']
        self.n_channels = self.config['n_channels']
        self.detector_by_channel = dsputils.get_detector_by_channel(self.config)

        # Maximum gap inside an S1-like cluster
        self.s1_gap_threshold = self.config.get('max_gap_size_in_s1like_cluster', 200) / self.dt

        # Maximum gap inside other clusters
        self.gap_threshold = self.config['max_gap_size_in_cluster'] / self.dt

        # Rise time threshold to mark S1 candidates
        self.rise_time_threshold = self.config.get('rise_time_threshold', 80)

    @staticmethod
    def iterate_gap_clusters(hits, gap_threshold):
        gaps = dsputils.gaps_between_hits(hits)
        cluster_indices = [0] + np.where(gaps > gap_threshold)[0].tolist() + [len(hits)]
        for i in range(len(cluster_indices) - 1):
            l_i, r_i = cluster_indices[i], cluster_indices[i + 1]
            yield l_i, r_i, hits[l_i:r_i]

    def transform_event(self, event):
        # Cluster hits in each detector separately.
        # Assumes detector channel mappings are non-overlapping
        for detector, channels in self.config['channels_in_detector'].items():
            hits = event.all_hits[(event.all_hits['channel'] >= channels[0]) &
                                  (event.all_hits['channel'] <= channels[-1])]
            if len(hits) == 0:
                continue
            hits.sort(order='left_central')

            # First cluster into small clusters. Try to find S1 candidates among them, and set these apart
            s1_mask = np.zeros(len(hits), dtype=np.bool)    # True if hit is part of S1 candidate
            lone_mask = np.zeros(len(hits), dtype=np.bool)  # Reject Lone hit
            for l_i, r_i, h in self.iterate_gap_clusters(hits, self.s1_gap_threshold):
                l = h['left'].min()
                center = np.sum(h['index_of_maximum'] * h['area']) / h['area'].sum()
                rise_time = (center - l) * self.dt

                if len(h) >= 3 and rise_time < self.rise_time_threshold:
                    # Yes, this is an S1 candidate. Mark it as a peak, hits will be ignored in next stage.
                    # print("S1 candidate %d-%d" % (l_i, r_i))
                    s1_mask[l_i:r_i] = True
                    event.peaks.append(self.build_peak(hits=h, detector=detector))

                elif len(h) <= 2:
                    # We don't want to merge lone hits to an S2 signal as well, this is helpful to split S2s
                    # This won't affect single electrons as they can hardly be < 3 fold coincidence
                    lone_mask[l_i:r_i] = True
                    event.peaks.append(self.build_peak(hits=h, detector=detector))

            # Remove hits that already left us as S1 candidates
            hits = hits[True ^ (s1_mask + lone_mask)]
            if len(hits) == 0:
                continue

            # Cluster remaining hits with the larger gap threshold
            for l_i, r_i, h in self.iterate_gap_clusters(hits, self.gap_threshold):
                event.peaks.append(self.build_peak(hits=h, detector=detector))

        return event

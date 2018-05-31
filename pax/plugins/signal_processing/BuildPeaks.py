import numpy as np
from pax import plugin, dsputils
from pax.plugins.peak_processing.BasicProperties import integrate_until_fraction


class GapSizeClustering(plugin.ClusteringPlugin):
    """Cluster individual hits into rough groups = Peaks separated by at least max_gap_size_in_cluster
    """

    def startup(self):
        self.dt = self.config['sample_duration']
        self.n_channels = self.config['n_channels']
        self.detector_by_channel = dsputils.get_detector_by_channel(self.config)

        # Maximum gap inside an S1-like cluster
        self.s1_gap_threshold = self.config['max_gap_size_in_s1like_cluster'] / self.dt

        # Maximum gap inside other clusters
        self.gap_threshold = self.config['max_gap_size_in_cluster'] / self.dt

        # Rise time bound
        self.s1_rise_time_bound = self.config['s1_risetime_threshold']  # 70ns as rise_time cut
        self.s1_width_bound = self.config['s1_width_threshold']  # 300 ns as width cut

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
            hits.sort(order='index_of_maximum')
            dt = self.dt

            # First cluster into small clusters. Try to find S1 candidates among them, and set these apart
            s1_mask = np.zeros(len(hits), dtype=np.bool)    # True if hit is part of S1 candidate
            for l_i, r_i, h in self.iterate_gap_clusters(hits, self.s1_gap_threshold):
                peak = self.build_peak(hits=h, detector=detector)
                # use summed WF to calculate peak properties
                w = event.get_sum_waveform(peak.detector).samples[peak.left:peak.right + 1]
                max_idx = np.argmax(w)
                peak.index_of_maximum = peak.left + max_idx
                # calculate rise time for peak classification
                area_times = np.ones(21) * float('nan')
                integrate_until_fraction(w, fractions_desired=np.linspace(0, 1, 21), results=area_times)
                area_times *= dt
                area_midpoint = area_times[10]
                # rise time
                area_decile_from_midpoint = area_times[::2] - area_midpoint
                rise_time = -area_decile_from_midpoint[1]
                # width
                range_area_decile = area_times[10:] - area_times[10::-1]

                if (rise_time < self.s1_rise_time_bound and range_area_decile[9] < self.s1_width_bound)\
                        or peak.n_contributing_channels < 4:
                    # Yes, this is an S1 candidate Or, this is a lone hit. Mark it as a peak,
                    # hits will be ignored in next stage.
                    s1_mask[l_i:r_i] = True
                    event.peaks.append(self.build_peak(hits=h, detector=detector))

            # Remove hits that already left us as S1 candidates or lone hits
            hits = hits[True ^ s1_mask]
            if len(hits) == 0:
                continue

            # Cluster remaining hits with the larger gap threshold
            for l_i, r_i, h in self.iterate_gap_clusters(hits, self.gap_threshold):
                event.peaks.append(self.build_peak(hits=h, detector=detector))

        return event

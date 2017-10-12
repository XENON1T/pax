import numpy as np
from scipy import interpolate
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
        self.s1_rise_time_bound = interpolate.interp1d([0, 5, 10, 100],
                                                       [80, 75, 70, 70],
                                                       fill_value='extrapolate', kind='linear')

        # Rise time vs AFT cut
        self.s1_rise_time_aft = interpolate.interp1d([0, 0.4, 0.5, 0.6, 0.70, 0.70, 1.0],
                                                     [70, 70, 68, 65, 60, 0, 0], kind='linear')

        self.first_top_ch = np.min(np.array(self.config['channels_top']))
        self.last_top_ch = np.max(np.array(self.config['channels_top']))

        # tight coincidence
        self.tight_coincidence_window = self.config.get('tight_coincidence_window', 50) // self.dt

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
            dt = self.dt

            # First cluster into small clusters. Try to find S1 candidates among them, and set these apart
            s1_mask = np.zeros(len(hits), dtype=np.bool)    # True if hit is part of S1 candidate
            for l_i, r_i, h in self.iterate_gap_clusters(hits, self.s1_gap_threshold):
                area_sum = np.sum(h['area'])
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
                area_decile_from_midpoint = area_times[::2] - area_midpoint
                rise_time = -area_decile_from_midpoint[1]

                # Area fraction on top
                peak.area_fraction_top =\
                    np.sum(peak.area_per_channel[self.first_top_ch:self.last_top_ch + 1]) / peak.area
                # rounding peak aft
                if peak.area_fraction_top < 0:
                    peak.area_fraction_top = 0
                elif peak.area_fraction_top > 1:
                    peak.area_fraction_top = 1

                rise_time_cut = self.s1_rise_time_aft(peak.area_fraction_top)
                # If area is large enough, change to a looser threshold
                if peak.area > 100 or peak.area < 5:
                    rise_time_cut = self.s1_rise_time_bound(peak.area)
                coincidence = peak.n_contributing_channels
                if rise_time < rise_time_cut or coincidence <= 4 or area_sum < 5:
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

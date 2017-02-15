from pax import plugin
from ckmeans import ckmeans
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class CKMeansClustering(plugin.ClusteringPlugin):

    def startup(self):
        self.min_split_goodness = InterpolatedUnivariateSpline(*self.config['split_goodness_threshold'], k=1)

    def transform_event(self, event):
        self.sum_w = event.get_sum_waveform('tpc').samples
        return super().transform_event(event)

    def cluster_peak(self, peak):
        if not peak.detector == 'tpc':
            return [peak]

        # Get the inter-peak sum waveform
        w = self.sum_w[peak.left:peak.right]

        # Do ckmeans clustering
        result = ckmeans(np.arange(len(w), dtype=np.float64), k=2, weights=w.astype(np.float64))

        x = result.within_ss.sum() / result.total_ss

        if x > 0.5:
            self.log.debug("Peak %s-%s: very bad split, not spliting" % (peak.left, peak.right))
            return [peak]
        else:
            # Convert to a goodness of split value
            x = -np.log(x/(0.5-x))

        threshold = self.min_split_goodness(np.log10(peak.area))
        if x > threshold:
            self.log.debug("Peak %s-%s: Goodness of split %0.2f > threshold %0.2f, splitting!" % (
                peak.left, peak.right, x, threshold))
            split_index = np.where(result.clustering)[0][0]
            return list(self.split_peak(peak, [split_index]))

        self.log.debug("Peak %s-%s: Goodness of split %0.2f <= threshold %0.2f, not splitting" % (
            peak.left, peak.right, x, threshold))
        return [peak]

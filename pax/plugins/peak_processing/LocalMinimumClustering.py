from pax import plugin, datastructure, dsputils
import logging
from pax.dsputils import find_split_points

log = logging.getLogger('LocalMinimumClusteringHelpers')


class LocalMinimumClustering(plugin.ClusteringPlugin):

    def cluster_peak(self, peak):
        if peak.type == 'lone_hit':
            return [peak]
        w = self.event.get_sum_waveform(peak.detector).samples[peak.left:peak.right + 1]
        split_points = list(find_split_points(w,
                                              min_height=self.config['min_height'],
                                              min_ratio=self.config.get('min_ratio', 3)))
        if not len(split_points):
            return [peak]
        else:
            self.log.debug("Splitting %d-%d into %d peaks" % (peak.left, peak.right, len(split_points) + 1))
            return list(self.split_peak(peak, split_points))

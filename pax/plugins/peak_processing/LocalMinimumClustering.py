from pax import plugin, datastructure, dsputils
import numpy as np
import logging
from pax.plugins.signal_processing.HitFinder import build_hits
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

    def finalize_event(self):
        # Update the event.all_hits field (for plotting), since new hits were created
        # Note we must separately get out the rejected hits, they are not in any peak...
        self.event.all_hits = np.concatenate([p.hits for p in self.event.peaks] +
                                             [self.event.all_hits[self.event.all_hits['is_rejected']]])

        # Restores order after shenanigans here
        # Important if someone uses (dict_)group_by from recarray tools later
        # As is done, for example in the hitfinder diagnostic plots... if you don't do this you get really strange
        # things there (missing hits were things were split, which makes you think there is a bug
        # in LocalMinimumClustering, but actually there isn't...)
        self.event.all_hits.sort(order='found_in_pulse')

    def split_peak(self, peak, split_points):
        """Yields new peaks split from peak at split_points = sample indices within peak
        Samples at the split points will fall to the right (so if we split [0, 5] on 2, you get [0, 1] and [2, 5]).
        Hits that straddle a split point are themselves split into two hits: peak.hits is updated.
        """
        # First, split hits that straddle the split points
        # Hits may have to be split several times; for each split point we modify the 'hits' list, splitting only
        # the hits we need.
        hits = peak.hits
        for x in split_points:
            x += peak.left   # Convert to index in event

            # Select hits that must be split: start before x and end after it.
            selection = (hits['left'] <= x) & (hits['right'] > x)
            hits_to_split = hits[selection]

            # new_hits will be a list of hit arrays, which we concatenate later to make the new 'hits' list
            # Start with the hits that don't have to be split: we definitely want to retain those!
            new_hits = [hits[True ^ selection]]

            for h in hits_to_split:
                pulse_i = h['found_in_pulse']
                pulse = self.event.pulses[pulse_i]

                # Get the pulse waveform in ADC counts above baseline (because it's what build_hits expect)
                baseline_to_subtract = self.config['digitizer_reference_baseline'] - pulse.baseline
                w = baseline_to_subtract - pulse.raw_data.astype(np.float64)

                # Use the hitfinder's build_hits to compute the properties of these hits
                # Damn this is ugly... but at least we don't have duplicate property computation code
                hits_buffer = np.zeros(2, dtype=datastructure.Hit.get_dtype())
                adc_to_pe = dsputils.adc_to_pe(self.config, h['channel'])
                hit_bounds = np.array([[h['left'], x], [x+1, h['right']]], dtype=np.int64)
                hit_bounds -= pulse.left   # build_hits expects hit bounds relative to pulse start
                build_hits(w,
                           hit_bounds=hit_bounds,
                           hits_buffer=hits_buffer,
                           adc_to_pe=adc_to_pe,
                           channel=h['channel'],
                           noise_sigma_pe=pulse.noise_sigma * adc_to_pe,
                           dt=self.config['sample_duration'],
                           start=pulse.left,
                           pulse_i=pulse_i,
                           saturation_threshold=self.config['digitizer_reference_baseline'] - pulse.baseline - 0.5)

                new_hits.append(hits_buffer)

            # Now remake the hits list, then go on to the next peak.
            hits = np.concatenate(new_hits)

        # Next, split the peaks, sorting hits to the right peak by their maximum index.
        # Iterate over left, right bounds of the new peaks
        boundaries = list(zip([0] + [y+1 for y in split_points], split_points + [float('inf')]))
        for l, r in boundaries:
            # Convert to index in event
            l += peak.left
            r += peak.left

            # Select hits which have their maximum within this peak bounds
            # The last new peak must also contain hits at the right bound (though this is unlikely to happen)
            hs = hits[(hits['index_of_maximum'] >= l) &
                      (hits['index_of_maximum'] <= r)]

            r = r if r < float('inf') else peak.right

            if not len(hs):
                raise RuntimeError("Attempt to create a peak without hits in LocalMinimumClustering!")

            yield self.build_peak(hits=hs, detector=peak.detector, left=l, right=r)


#  @numba.jit(numba.float64[:])
# TODO: TESTS!
def find_split_points(w, min_height, min_ratio):
    """"Finds local minima in w,
    whose peaks to the left and right both satisfy:
      - larger than minimum + min_height
      - larger than minimum * min_ratio
    """
    last_max = 0
    min_since_max = float('inf')
    min_since_max_i = 0

    for i, x in enumerate(w):
        if x < min_since_max:
            # New minimum since last max
            min_since_max = x
            min_since_max_i = i

        if min(last_max, x) > max(min_since_max + min_height,
                                  min_since_max * min_ratio):
            # Significant local minimum: tell caller, reset both max and min finder
            yield min_since_max_i
            last_max = x
            min_since_max = float('inf')
            min_since_max_i = i

        if x > last_max:
            # New max, reset minimum finder state
            # Notice this is AFTER the split check, to accomodate very fast rising second peaks
            last_max = x
            min_since_max = float('inf')
            min_since_max_i = i

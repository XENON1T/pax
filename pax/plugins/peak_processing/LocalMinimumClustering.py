from pax import plugin, datastructure, dsputils
from scipy.stats import norm
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
                           saturation_threshold=self.config['digitizer_reference_baseline'] - pulse.baseline - 0.5,
                           central_bounds=hit_bounds)       # TODO: Recompute central bounds in an intelligent way...

                # Remove hits with 0 or negative area (very rare, but possible due to rigid integration bound)
                hits_buffer = hits_buffer[hits_buffer['area'] > 0]

                new_hits.append(hits_buffer)

            # Now remake the hits list, then go on to the next peak.
            hits = np.concatenate(new_hits)

        # Next, split the peaks, sorting hits to the right peak by their maximum index.
        # Iterate over left, right bounds of the new peaks
        boundaries = list(zip([0] + [y+1 for y in split_points], split_points + [float('inf')]))
        for left, right in boundaries:
            # Convert to index in event
            left += peak.left
            right += peak.left

            # Select hits which have their maximum within this peak bounds
            # The last new peak must also contain hits at the right bound (though this is unlikely to happen)
            hs = hits[(hits['index_of_maximum'] >= left) &
                      (hits['index_of_maximum'] <= right)]

            if not len(hs):
                # Hits have probably been removed by area > 0 condition
                self.log.info("Localminimumclustering requested creation of peak %s-%s without hits. "
                              "This is a possible outcome if there are large oscillations in one channel, "
                              "but it should be very rare." % (left, right))
                continue

            right = right if right < float('inf') else peak.right

            if not len(hs):
                raise RuntimeError("Attempt to create a peak without hits in LocalMinimumClustering!")

            yield self.build_peak(hits=hs, detector=peak.detector, left=left, right=right)


def find_split_points(w, min_height, min_ratio):
    """"Finds local minima in w,
    whose peaks to the left and right both satisfy:
      - larger than minimum + min_height
    """
    if np.max(w) < min_height:
        return []

    # Smooth sum waveform, then use first and sign of second order dirivitive
    # to locate minima and maxima
    above_min = np.where(w > min_height)[0]
    above_min_diff = np.diff(above_min, n=1)
    above_min_diff = np.where(above_min_diff > 1000)[0]

    above_min_list = np.split(above_min, above_min_diff+1)

    all_good_minima = []
    for above_min in above_min_list:
        left = int(np.clip(above_min[0]-500, 0, np.inf))
        right = int(np.clip(above_min[-1]+500, 0, len(w)))

        partial_xindex = np.arange(right-left)[::2]

        # Some issue with memory usage by lowess, 10000/5000 samples will require 1/.18 Gb memory
        # Skip abnormal long peaks
        if right-left > 5e3:
            continue
        else:
            _w = dsputils.smooth_lowess(w[left:right][::2], partial_xindex, frac=30/(right-left))
        dw = np.diff(_w, n=1)
        minima = np.where((np.hstack((dw, -1)) > 0) & (np.hstack((1, dw)) <= 0))[0]
        maxima = np.where((np.hstack((dw, 1)) <= 0) & (np.hstack((-1, dw)) > 0))[0]

        # We don't care about minima if it don't have maxima on both side
        # This automatically make one less minima then maxima
        if len(maxima) < 2:
            continue
        minima = minima[(minima > maxima[0]) & (minima < maxima[-1])]

        # We look for the last minima between big peaks
        good_minima = [int(partial_xindex[minima[ix]]+left) for ix in range(len(minima))
                       if ((_w[maxima][ix+1] - _w[minima][ix] > min_height)
                           & any(_w[maxima][:ix+1] - _w[minima][ix] > min_height))]
        all_good_minima += good_minima

    return all_good_minima


class TailSeparation(LocalMinimumClustering):
    """Split off the tail from none s1, lone_hit peaks
    """

    def cluster_peak(self, peak):
        if peak.type == 'lone_hit' or peak.type == 's1':
            return [peak]
        w = self.event.get_sum_waveform(peak.detector).samples[peak.left:peak.right + 1]
        split_points = list(find_tail_point(w,
                                            min_height=self.config['min_height'],
                                            ratio=self.config['tail_cutoff_ratio']))
        if not len(split_points):
            return [peak]
        else:
            self.log.debug("Splitting %d-%d into %d peaks" % (peak.left, peak.right, len(split_points) + 1))
            return list(self.split_peak(peak, split_points))


def find_tail_point(w, min_height, ratio):
    """Find best fit area,
    that minimize std of gaussian approximation from waveform.
    find when waveform fall below area * ratio,
    and at least 2 sigma away from mean time.
    """
    if np.max(w) < min_height:
        return []

    # Quick fit to the waveform and locate tail position
    ix = np.arange(len(w))

    # Use center population to estimate mean and sigma
    cnt = w > 0.02*np.max(w)
    mean = np.sum(ix[cnt]*w[cnt])/np.sum(w[cnt])
    sigma = (np.sum((ix[cnt]-mean)**2*w[cnt])/np.sum(w[cnt]))**0.5

    # Use estimated mean and sigma to find amplitude
    amp = np.sum(w[cnt]*norm.pdf(ix[cnt], mean, sigma))/np.sum(norm.pdf(ix[cnt], mean, sigma)**2)

    # Define tail by waveform height drop below certain ratio of amplitude
    tail = np.where(w < ratio*amp)[0]
    tail = tail[(tail > mean+2*sigma) & (tail != len(w))]

    if len(tail) > 0:
        return tail[:1]
    else:
        return []

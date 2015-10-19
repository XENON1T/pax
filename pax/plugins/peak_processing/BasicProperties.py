import numpy as np
# import numba

from pax import plugin, dsputils


class BasicProperties(plugin.TransformPlugin):
    """Computes basic properties of each peak, based on the hits.
    Also sets lone_hit for peaks that have one hit.
    Yes, this is done also in BuildPeaks (and has to be done there, as the noise rejection relies on it)
    but new lone hits may have happened due to the noise rejection & clustering
    """

    def transform_event(self, event):
        last_top_ch = np.max(np.array(self.config['channels_top']))

        for peak in event.peaks:
            hits = peak.hits
            if len(hits) == 0:
                raise ValueError("Can't compute properties of an empty peak!")
            if len(hits) == 1:
                peak.type = 'lone_hit'

            peak.left = hits['left'].min()
            peak.right = hits['right'].max()

            peak.area_per_channel = dsputils.count_hits_per_channel(peak, self.config, weights=hits['area'])
            peak.hits_per_channel = dsputils.count_hits_per_channel(peak, self.config).astype(np.int16)
            n_saturated_tot = hits['n_saturated'].sum()
            if n_saturated_tot:
                peak.n_saturated_per_channel = dsputils.count_hits_per_channel(
                    peak, self.config, weights=hits['n_saturated']).astype(np.int16)
            else:
                peak.n_saturated_per_channel = np.zeros(self.config['n_channels'], dtype=np.int16)

            peak.mean_amplitude_to_noise = np.average(hits['height']/hits['noise_sigma'], weights=hits['area'])

            peak.area = np.sum(peak.area_per_channel)
            peak.n_hits = np.sum(peak.hits_per_channel)
            peak.n_saturated_samples = np.sum(peak.n_saturated_per_channel)
            peak.n_saturated_channels = len(np.where(peak.n_saturated_per_channel)[0])
            peak.n_contributing_channels = len(peak.contributing_channels)
            peak.mean_amplitude_to_noise /= peak.area

            # Compute top fraction
            peak.area_fraction_top = np.sum(peak.area_per_channel[:last_top_ch + 1]) / peak.area
            peak.hits_fraction_top = np.sum(peak.hits_per_channel[:last_top_ch + 1]) / peak.area

            # Compute timing quantities
            peak.hit_time_mean, peak.hit_time_std = weighted_mean_variance(hits['center'], hits['area'])
            peak.hit_time_std **= 0.5  # Convert variance to std
            peak.n_contributing_channels_top = np.sum((peak.area_per_channel[:last_top_ch + 1] > 0))

            if peak.n_contributing_channels == 0:
                raise RuntimeError("Every peak should have at least one contributing channel... what's going on?")

        return event


class SumWaveformProperties(plugin.TransformPlugin):
    """Computes properties based on the hits-only sum waveform"""

    def startup(self):
        self.wv_field_len = int(self.config['peak_waveform_length'] / self.config['sample_duration']) + 1
        if not self.wv_field_len % 2:
            raise ValueError('peak_waveform_length must be an even multiple of the sample size')

    def transform_event(self, event):
        dt = event.sample_duration
        field_length = self.wv_field_len
        for peak in event.peaks:
            peak.sum_waveform = np.zeros(field_length, dtype=peak.sum_waveform.dtype)
            peak.sum_waveform_top = np.zeros(field_length, dtype=peak.sum_waveform.dtype)

            # Get the waveform (in pe/bin) and compute basic sum-waveform derived properties
            w = event.get_sum_waveform(peak.detector).samples[peak.left:peak.right + 1]

            # Center of gravity in the hits-only sum waveform. Identical to peak.hit_time_mean...
            # We may remove one from the data structure, but it's a useful sanity check
            # (particularly since some hits got removed in the noise rejection)
            peak.center_time = (peak.left + np.average(np.arange(len(w)), weights=w)) * dt

            # Index in peak waveform nearest to center of gravity (for sum-waveform alignment)
            cog_idx = int(round(peak.center_time / dt)) - peak.left
            # Index of the peak's maximum
            max_idx = np.argmax(w)
            peak.index_of_maximum = peak.left + max_idx
            # Amplitude at the maximum
            peak.height = w[max_idx]

            # Compute fraction of area in central deciles
            peak.area_midpoint, peak.range_area_decile = compute_area_deciles(w)
            peak.range_area_decile *= dt
            peak.area_midpoint += peak.left * dt

            # Store the waveform; for tpc also store the top waveform
            self.log.debug("Storing sum waveform for peak %d-%d-%d in %s" % (peak.left,
                                                                             int(round(peak.center_time / dt)),
                                                                             peak.right,
                                                                             peak.detector))
            put_w_in_center_of_field(w, peak.sum_waveform, cog_idx)
            if peak.detector == 'tpc':
                put_w_in_center_of_field(event.get_sum_waveform('tpc_top').samples[peak.left:peak.right + 1],
                                         peak.sum_waveform_top, cog_idx)

        return event


class CountCoincidentNoisePulses(plugin.TransformPlugin):

    def transform_event(self, event):
        noise_pulses = [p for p in event.pulses if p.n_hits_found == 0]
        for peak in event.peaks:
            for nop in noise_pulses:
                if nop.left <= peak.right and nop.right >= peak.left:
                    peak.n_noise_pulses += 1
        return event


def compute_area_deciles(w):
    """Return (index of mid area, array of the 0th ... 10 th area decile ranges in samples) of w
    e.g. range_area_decile[5] = range of 50% area = distance (in samples)
    between point of 25% area and 75% area (with boundary samples added fractionally).
    First element (0) of array is always zero, last element (10) is the length of w in samples.
    """
    fractions_desired = np.linspace(0, 1, 21)
    index_of_area_fraction = np.ones(len(fractions_desired)) * float('nan')
    integrate_until_fraction(w, fractions_desired, index_of_area_fraction)
    return index_of_area_fraction[10], (index_of_area_fraction[10:] - index_of_area_fraction[10::-1]),


# @numba.jit(numba.void(numba.float32[:], numba.float64[:], numba.float64[:]),
#            nopython=True, cache=True)
# For some reason numba doesn't clean up its memory properly for this function... leave it in python for now
def integrate_until_fraction(w, fractions_desired, results):
    """For array of fractions_desired, integrate w until fraction of area is reached, place sample index in results
    Will add last sample needed fractionally.
    eg. if you want 25% and a sample takes you from 20% to 30%, 0.5 will be added.
    Assumes fractions_desired is sorted and all in [0, 1]!
    """
    area_tot = w.sum()
    fraction_seen = 0
    current_fraction_index = 0
    needed_fraction = fractions_desired[current_fraction_index]
    for i, x in enumerate(w):
        # How much of the area is in this sample?
        fraction_this_sample = x/area_tot
        # Will this take us over the fraction we seek?
        # Must be while, not if, since we can pass several fractions_desired in one sample
        while fraction_seen + fraction_this_sample >= needed_fraction:
            # Yes, so we need to add the next sample fractionally
            area_needed = area_tot * (needed_fraction - fraction_seen)
            results[current_fraction_index] = i + area_needed/x
            # Advance to the next fraction
            current_fraction_index += 1
            if current_fraction_index > len(fractions_desired) - 1:
                return
            needed_fraction = fractions_desired[current_fraction_index]
        # Add this sample's area to the area seen, advance to the next sample
        fraction_seen += fraction_this_sample
    if needed_fraction == 1:
        results[current_fraction_index] = len(w)
    else:
        # Sorry, can't add the last fraction to the error message: numba doesn't allow it
        raise RuntimeError("Fraction not reached in waveform? What the ...?")


def put_w_in_center_of_field(w, field, center_index):
    """Stores (part of) the array w in a fixed length array field, with center_index in field's center.
    Assumes field has odd length.
    """
    field_length = len(field)
    if not field_length % 2:
        raise ValueError("put_w_in_center_of_field requires an odd field length (so center is clear)")
    field_center = int(field_length/2)      # Index of center of field

    left_overhang = center_index - field_center
    if left_overhang > 0:
        # Chop off the left overhang
        w = w[left_overhang:]
        center_index -= left_overhang
        assert center_index == field_center

    right_overhang = len(w) - field_length + (field_center - center_index)
    if right_overhang > 0:
        # Chop off any remaining right overhang
        w = w[:len(w)-right_overhang]

    start_idx = field_center - center_index
    field[start_idx:start_idx + len(w)] = w


def weighted_mean_variance(values, weights):
    """
    Return the weighted mean, and the weighted sum square deviation from the weighted mean.
    values, weights -- Numpy ndarrays with the same shape.
    Stolen from http://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    """
    weighted_mean = np.average(values, weights=weights)
    weighted_variance = np.average((values-weighted_mean)**2, weights=weights)  # Fast and numerically precise
    return weighted_mean, weighted_variance

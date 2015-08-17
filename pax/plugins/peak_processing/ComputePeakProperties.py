import numpy as np

from pax import plugin, utils


class BasicProperties(plugin.TransformPlugin):
    """Computes basic peak properies such as area and hit time spread ("width")
    """

    def startup(self):
        self.last_top_ch = np.max(np.array(self.config['channels_top']))

    def transform_event(self, event):

        for peak in event.peaks:
            # Compute the area per pmt and saturation count
            peak.n_saturated_per_channel = np.zeros(event.n_channels, dtype=np.int16)
            for s in peak.hits:
                ch = s.channel
                peak.area_per_channel[ch] += s.area
                peak.n_saturated_per_channel[ch] += s.n_saturated

            # Compute the total area and saturation count
            peak.area = np.sum(peak.area_per_channel)
            peak.n_saturated = np.sum(peak.n_saturated_per_channel)

            # Compute top fraction
            peak.area_fraction_top = np.sum(peak.area_per_channel[:self.last_top_ch + 1]) / peak.area

            # Compute timing quantities
            times = [s.center for s in peak.hits]
            peak.hit_time_mean, peak.hit_time_std = utils.weighted_mean_variance(times,
                                                                                 [s.area for s in peak.hits])
            peak.hit_time_std **= 0.5  # Convert variance to std

            # Compute mean amplitude / noise
            try:
                peak.mean_amplitude_to_noise = np.average([hit.height / hit.noise_sigma for hit in peak.hits],
                                                          weights=[hit.area for hit in peak.hits])
            except ZeroDivisionError:
                self.log.warning('One of the hits in %s peak %d-%d has 0 noise sigma... strange!'
                                 'This should only happen in extremely rare cases, '
                                 'simulated data or with very strange settings' % (peak.detector,
                                                                                   peak.left, peak.right))

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

            # Get the waveform and compute some properties
            w = event.get_sum_waveform(peak.detector).samples[peak.left:peak.right + 1]
            peak.center_time = (peak.left + np.average(np.arange(len(w)), weights=w)) * dt
            cog_idx = int(round(peak.center_time / dt)) - peak.left
            max_idx = np.argmax(w)
            peak.index_of_maximum = peak.left + max_idx
            peak.height = w[max_idx]
            peak.range_50p_area = range_of_fraction_of_area(w, center=cog_idx, fraction=0.5) * dt
            peak.range_90p_area = range_of_fraction_of_area(w, center=cog_idx, fraction=0.9) * dt

            # Store the waveform; for tpc also store the top waveform
            put_w_in_center_of_field(w, peak.sum_waveform, cog_idx)
            if peak.detector == 'tpc':
                put_w_in_center_of_field(event.get_sum_waveform('tpc_top').samples[peak.left:peak.right + 1],
                                         peak.sum_waveform_top, cog_idx)

        return event


def range_of_fraction_of_area(w, center, fraction):
    """Compute range of peaks that includes fraction of area, moving outward from center (index in w).
    The move left / move right decision is made by which sample is higher.
    Returns number of samples included. Fractional part is how much of the last sample would have to be included to get
    to the exact fraction, assuming the amplitude is constant over that sample.
    This function is a pretty low-level algorithm, so it could probably be numba'd
    """
    total_area = w.sum()

    left = center       # Last sample index left of maximum already included
    right = center      # Last sample index right of maximum already included
    area_seen = w[center]   # Area already included
    last_sample_included = center   # Last sample included
    while area_seen < total_area * fraction:
        # If we can advance left, and either we can't advance left or advancing left would gain us more, advance left
        if left > 0 and right == len(w) - 1 or w[left] > w[right]:
            left -= 1
            last_sample_included = left
            area_seen += w[left]
        else:
            right += 1
            last_sample_included = right
            area_seen += w[right]

    # Now we have slightly more than the fraction.
    # Estimate how much of the last sample we should exclude to get back to the exact fraction.
    last_amplitude = w[last_sample_included]
    excess_area = area_seen - total_area * fraction
    reduce_fraction = excess_area / last_amplitude

    return right - left + 1 - max(0, min(1, reduce_fraction))


def put_w_in_center_of_field(w, field, center_index):
    """Stores (part of) the array w in a fixed length array field, with center_index in field's center.
    Assumes field has odd length.
    """
    # TODO: Needs tests!
    field_length = len(field)
    if not field_length % 2:
        raise ValueError("put_w_in_center_of_field requires an odd field length (so center is clear)")
    field_center = int(field_length/2) + 1

    left_overhang = center_index - field_center
    if left_overhang > 0:
        # Chop off the left overhang
        w = w[left_overhang:]
        center_index = field_center

    right_overhang = len(w) - field_length + (field_center - center_index)
    if right_overhang > 0:
        # Chop off any remaining right overhang
        w = w[:len(w)-right_overhang]

    start_idx = field_center - center_index
    field[start_idx:start_idx + len(w)] = w


class HitpatternSpread(plugin.TransformPlugin):
    """Computes the weighted root mean square deviation of the top and bottom hitpattern for each peak
    """

    def startup(self):

        # Grab PMT numbers and x, y locations in each array
        self.pmts = {}
        self.locations = {}
        for array in ('top', 'bottom'):
            self.pmts[array] = self.config['channels_%s' % array]
            self.locations[array] = {}
            for dim in ('x', 'y'):
                self.locations[array][dim] = np.array([self.config['pmt_locations'][ch][dim]
                                                       for ch in self.pmts[array]])

    def transform_event(self, event):

        for peak in event.peaks:

            # No point in computing this for veto peaks
            if peak.detector != 'tpc':
                continue

            for array in ('top', 'bottom'):

                hitpattern = peak.area_per_channel[self.pmts[array]]

                if np.all(hitpattern == 0.0):
                    # Empty hitpatterns will give error in np.average
                    continue

                weighted_var = 0
                for dim in ('x', 'y'):
                    _, wv = utils.weighted_mean_variance(self.locations[array][dim],
                                                         weights=hitpattern)
                    weighted_var += wv

                setattr(peak, '%s_hitpattern_spread' % array, np.sqrt(weighted_var))

        return event

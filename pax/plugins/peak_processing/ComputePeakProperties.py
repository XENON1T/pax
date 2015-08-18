import numpy as np
import numba

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
            peak.range_20p_area = range_of_fraction_of_area(w, center=cog_idx, fraction=0.2) * dt
            peak.range_50p_area = range_of_fraction_of_area(w, center=cog_idx, fraction=0.5) * dt
            peak.range_90p_area = range_of_fraction_of_area(w, center=cog_idx, fraction=0.9) * dt

            # Store the waveform; for tpc also store the top waveform
            put_w_in_center_of_field(w, peak.sum_waveform, cog_idx)
            if peak.detector == 'tpc':
                put_w_in_center_of_field(event.get_sum_waveform('tpc_top').samples[peak.left:peak.right + 1],
                                         peak.sum_waveform_top, cog_idx)

        return event


@numba.jit(nopython=True)
def range_of_fraction_of_area(w, center, fraction):
    """Compute range of peaks that includes fraction of area, moving outward from center (index in w)
    towards side that has most area remaining. Returns number of samples included.
    Fractional part is determined as follows:
     - If we can only move in one direction: the fractional part of the sample which takes us over the desired fraction
     - If we can move in both directions, and one of the samples left or right would be enough to take us over
       the desired fraction: the needed fraction of that sample.
     - ... if both samples left and right are needed to take us over the desired fraction: the highest sample is fully
      included, then the other fractionally.
    This function is a pretty low-level algorithm, so it could probably be numba'd
    """
    total_area = w.sum()
    area_todo = total_area * fraction   # Area to still include
    area_left = w[:center].sum()        # Unseen area remaining on the left
    area_right = w[center+1:].sum()     # Unseen area remaining on the right

    # Edge case where center sample would already take us over area_todo
    if w[center] > area_todo:
        return area_todo / w[center]

    left = int(center)       # Last sample index left of maximum already included
    right = int(center)      # Last sample index right of maximum already included
    area_todo -= w[center]
    extra_width = 0.0        # Fractional amount of last sample to add.

    while True:
        if area_todo == 0:
            break
        # If we cannot move left, or there is more remaining area to the right, move right
        if left == 0 or area_right > area_left:
            # Move right
            fraction_of_todo = w[right + 1] / area_todo
            if fraction_of_todo > 1:
                extra_width = 1 / fraction_of_todo
                break
            right += 1
            area_todo -= w[right]
            area_right -= w[right]
        else:
            # Move left
            fraction_of_todo = w[left - 1] / area_todo
            if fraction_of_todo > 1:
                extra_width = 1 / fraction_of_todo
                break
            left -= 1
            area_todo -= w[left]
            area_left -= w[left]

    return right - left + 1 + extra_width


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

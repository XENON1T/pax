import numpy as np

from pax import plugin, utils, exceptions
from pax.InterpolatingMap import InterpolatingMap
from pax.datastructure import Interaction


class BuildInteractions(plugin.TransformPlugin):
    """Construct interactions from combinations of S1 and S2, as long as
      - The S2 occurs after the S1
      - The S2 is larger than s2_pairing_threshold (avoids single electrons)
    Mo more than pair_n_s2s S2s S2s and pair_n_s1s S1s will be paired with each other
    Pairing will start from the largest S1 and S2, then move down S2s in area and eventually down S1s in area

    xy_posrec_preference = ['algo1', 'algo2', ...]
    """
    def startup(self):
        if 'xy_posrec_preference' not in self.config:
            raise ValueError('Configuration for %s must contain xy_posrec_preference' % self.name)

    def transform_event(self, event):
        s1s = event.s1s()
        s1s = s1s[:min(len(s1s), self.config.get('pair_n_s1s'))]

        s2s = event.s2s()
        s2_area_limit = self.config.get('s2_pairing_threshold', 0)
        s2s = [p for p in s2s if p.area >= s2_area_limit]
        s2s = s2s[:min(len(s2s), self.config.get('pair_n_s2s'))]

        for s1 in s1s:
            for s2 in s2s:

                # Compute drift time, add only interactions with s1 before s2
                dt = s2.hit_time_mean - s1.hit_time_mean
                if dt < 0:
                    continue

                ia = Interaction()
                ia.s1 = s1
                ia.s2 = s2
                ia.drift_time = dt
                ia.set_position(ia.s2.get_position_from_preferred_algorithm(self.config['xy_posrec_preference']))

                # Append to event
                event.interactions.append(ia)

        return event


class BasicInteractionProperties(plugin.TransformPlugin):
    """Compute basic properties of each interaction
    S1 and S2 x, y, z corrections, S1 hitpattern fit
    """

    def startup(self):
        self.s1_correction_map = InterpolatingMap(utils.data_file_name(self.config['s1_correction_map']))
        self.s2_correction_map = InterpolatingMap(utils.data_file_name(self.config['s2_correction_map']))
        self.s1_patterns = self.processor.simulator.s1_patterns
        self.s2_patterns = self.processor.simulator.s2_patterns
        self.zombie_pmts_s1 = np.array(self.config.get('zombie_pmts_s1', []))
        self.zombie_pmts_s2 = np.array(self.config.get('zombie_pmts_s2', []))
        self.tpc_channels = self.config['channels_in_detector']['tpc']
        self.do_saturation_correction = self.config.get('active_saturation_and_zombie_correction', False)

    def transform_event(self, event):

        for ia in event.interactions:
            # Electron lifetime correction to S2 area
            ia.s2_area_correction *= np.exp(ia.drift_time / self.config['electron_lifetime_liquid'])

            # Determine z position from drift time
            ia.z = self.config['drift_velocity_liquid'] * ia.drift_time

            # S1(x, y, z) and S2(x, y) corrections for varying light yield
            # TODO: replace correction map by light yield maps in simulator, then divide by their value here
            ia.s1_area_correction *= self.s1_correction_map.get_value_at(ia)
            ia.s2_area_correction *= self.s2_correction_map.get_value_at(ia)

            if self.s2_patterns is not None and self.do_saturation_correction:
                # Correct for S2 saturation
                # As we don't have an (x, y) dependent LCE map for the bottom PMTs for S2s,
                # we can only compute the correction on the top area.
                ia.s2_area_correction *= self.area_correction(
                    peak=ia.s2,
                    channels_in_pattern=self.config['channels_top'],
                    expected_pattern=self.s2_patterns.expected_pattern((ia.x, ia.y)),
                    confused_channels=np.union1d(ia.s2.saturated_channels, self.zombie_pmts_s2))

            if self.s1_patterns is not None:
                confused_s1_channels = np.union1d(ia.s1.saturated_channels, self.zombie_pmts_s1)

                # Correct for S1 saturation
                try:
                    if self.do_saturation_correction:
                        ia.s1_area_correction *= self.area_correction(
                            peak=ia.s1,
                            channels_in_pattern=self.tpc_channels,
                            expected_pattern=self.s1_patterns.expected_pattern((ia.x, ia.y, ia.drift_time)),
                            confused_channels=confused_s1_channels)

                    # Compute the S1 pattern fit statistic
                    ia.s1_pattern_fit = self.s1_patterns.compute_gof(
                        (ia.x, ia.y, ia.drift_time),
                        ia.s1.area_per_channel[self.tpc_channels],
                        pmt_selection=np.setdiff1d(self.tpc_channels, confused_s1_channels),
                        statistic=self.config['s1_pattern_statistic'])

                except exceptions.CoordinateOutOfRangeException:
                    # This happens for interactions reconstructed outside of the TPC
                    # Do not add any saturation correction, leave pattern fit statistic float('nan')
                    pass

        return event

    def area_correction(self, peak, channels_in_pattern, expected_pattern, confused_channels):
        """Return multiplicative area correction obtained by replacing area in confused_channels by
        expected area based on expected_pattern in channels_in_pattern.
        expected_pattern does not have to be normalized: we'll do that for you.
        We'll also ensure any confused_channels not in channels_in_pattern are ignored.
        """
        try:
            confused_channels = np.intersect1d(confused_channels, channels_in_pattern).astype(np.int)
        except exceptions.CoordinateOutOfRangeException:
            self.log.warning("Expected area fractions for peak %d-%d are zero -- "
                             "cannot compute saturation & zombie correction!" % (peak.left, peak.right))
            return 1
        # PatternFitter should have normalized the pattern
        assert abs(np.sum(expected_pattern) - 1) < 0.01

        area_seen_in_pattern = peak.area_per_channel[channels_in_pattern].sum()
        area_in_good_channels = area_seen_in_pattern - peak.area_per_channel[confused_channels].sum()
        fraction_of_pattern_in_good_channels = 1 - expected_pattern[confused_channels].sum()

        # Area in channels not in channels_in_pattern is left alone
        new_area = peak.area - area_seen_in_pattern

        # Estimate the area in channels_in_pattern by excluding the confused channels
        new_area += area_in_good_channels / fraction_of_pattern_in_good_channels

        return new_area / peak.area

import numpy as np

from pax import plugin, exceptions
from pax.datastructure import Interaction
from pax.dsputils import saturation_correction


class BuildInteractions(plugin.TransformPlugin):
    """Construct interactions from combinations of S1 and S2, as long as
      - The S2 occurs after the S1
      - The S2 is larger than s2_pairing_threshold (avoids single electrons)
    Mo more than pair_n_s2s S2s and pair_n_s1s S1s will be paired with each other
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
                ia.s1 = event.peaks.index(s1)
                ia.s2 = event.peaks.index(s2)
                ia.drift_time = dt
                try:
                    # Get x,y position from peak
                    ia.set_position(s2.get_position_from_preferred_algorithm(self.config['xy_posrec_preference']))
                except ValueError:
                    self.log.debug("Could not find any position from the chosen algorithms")
                # Append to event
                event.interactions.append(ia)

        return event


class BasicInteractionProperties(plugin.TransformPlugin):
    """Compute basic properties of each interaction
    S1 and S2 x, y, z corrections, S1 hitpattern fit
    """

    def startup(self):
        self.s1_light_yield_map = self.processor.simulator.s1_light_yield_map
        self.s1_patterns = self.processor.simulator.s1_patterns
        self.zombie_pmts_s1 = np.array(self.config.get('zombie_pmts_s1', []))
        self.tpc_channels = self.config['channels_in_detector']['tpc']
        self.do_saturation_correction = self.config.get('active_saturation_and_zombie_correction', False)

    def transform_event(self, event):

        for ia in event.interactions:
            s1 = event.peaks[ia.s1]
            s2 = event.peaks[ia.s2]

            # Electron lifetime correction on S2 area
            ia.s2_lifetime_correction *= np.exp(ia.drift_time / self.config['electron_lifetime_liquid'])

            # Determine z position from drift time
            ia.z = - self.config['drift_velocity_liquid'] * ia.drift_time

            # S1 and S2 area correction: divide by relative light yield at the position
            ia.s1_spatial_correction /= self.s1_light_yield_map.get_value_at(ia)

            if self.s1_patterns is not None:
                confused_s1_channels = np.union1d(s1.saturated_channels, self.zombie_pmts_s1)

                # Correct for S1 saturation
                try:
                    if self.do_saturation_correction:
                        ia.s1_saturation_correction *= saturation_correction(
                            peak=s1,
                            channels_in_pattern=self.tpc_channels,
                            expected_pattern=self.s1_patterns.expected_pattern((ia.x, ia.y, ia.z)),
                            confused_channels=confused_s1_channels,
                            log=self.log)

                    # Compute the S1 pattern fit statistic
                    ia.s1_pattern_fit = self.s1_patterns.compute_gof(
                        (ia.x, ia.y, ia.z),
                        s1.area_per_channel[self.tpc_channels],
                        pmt_selection=np.setdiff1d(self.tpc_channels, confused_s1_channels),
                        statistic=self.config['s1_pattern_statistic'])

                except exceptions.CoordinateOutOfRangeException:
                    # This happens for interactions reconstructed outside of the TPC
                    # Do not add any saturation correction, leave pattern fit statistic float('nan')
                    pass

                # Get the full area correction
                ia.s1_area_correction *= ia.s1_spatial_correction * ia.s1_saturation_correction
                ia.s2_area_correction *= (ia.s2_lifetime_correction *
                                          s2.s2_spatial_correction *
                                          s2.s2_saturation_correction)
        return event

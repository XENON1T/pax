import numpy as np

from pax import plugin, utils, dsputils
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

                # Compute drift time, continue if s2 before s1
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
    """"""

    def startup(self):
        self.s1_correction_map = dsputils.InterpolatingMap(utils.data_file_name(self.config['s1_correction_map']))
        self.s2_correction_map = dsputils.InterpolatingMap(utils.data_file_name(self.config['s2_correction_map']))

    def transform_event(self, event):

        for ia in event.interactions:
            # Electron lifetime correction
            ia.s2_area_correction *= np.exp(ia.drift_time / self.config['electron_lifetime_liquid'])

            # Determine z position from drift time
            ia.z = self.config['drift_velocity_liquid'] * ia.drift_time

            # S1 and S2 corrections
            ia.s1_area_correction *= self.s1_correction_map.get_value_at(ia)
            ia.s2_area_correction *= self.s2_correction_map.get_value_at(ia)

        return event

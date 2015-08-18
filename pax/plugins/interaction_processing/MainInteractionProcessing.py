
import numpy as np

from pax import plugin, utils


class MainInteractionProcessing(plugin.TransformPlugin):
    """Construct the main interaction from the event.
     1) Select main S1 based on select_main_s1_by (in reverse order, i.e. largest first)
        Do same for S2 based on select_main_s2_by
     2) If no S1s, no S2s, or main S1 is after main S2, return event without modification
     3) Set event.s1 to S1 and event.s2. Compute drift time, z position, S1 and S2 corrections
    xy_posrec_preference = ['algo1', 'algo2', ...]
    """
    def startup(self):
        if 'xy_posrec_preference' not in self.config:
            raise ValueError('Configuration for %s must contain xy_posrec_preference' % self.name)
        self.s1_correction_map = utils.InterpolatingMap(utils.data_file_name(self.config['s1_correction_map']))
        self.s2_correction_map = utils.InterpolatingMap(utils.data_file_name(self.config['s2_correction_map']))

    def transform_event(self, event):
        s1s = event.s1s(sort_key=self.config['select_main_s1_by'])
        s2s = event.s2s(sort_key=self.config['select_main_s2_by'])
        if not s1s or not s2s:
            return event

        drift_time = s2s[0].hit_time_mean - s1s[0].hit_time_mean
        if drift_time < 0:
            return event

        event.s1 = s1s[0]
        event.s2 = s2s[0]
        event.drift_time = drift_time

        # Set the x,y position
        for algo in self.config['xy_posrec_preference']:
            rp = s2s[0].get_reconstructed_position_from_algorithm(algo)
            if rp is not None:
                event.set_position(rp)
                break
        else:
            raise RuntimeError("No reconstructed xy position from an algorithm "
                               "that appears in xy_posrec_preference was found!")

        # Electron lifetime correction
        event.s2_area_correction *= np.exp(event.drift_time / self.config['electron_lifetime_liquid'])

        # Determine z position from drift time
        event.z = event.drift_time / self.config['drift_velocity_liquid']

        # S1 and S2 corrections
        event.s1_area_correction *= self.s1_correction_map.get_value_at(event)
        event.s2_area_correction *= self.s2_correction_map.get_value_at(event)

        return event

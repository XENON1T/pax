import numpy as np
from pax import plugin


class RZCorrection(plugin.TransformPlugin):

    def startup(self):
        self.rzmap = self.processor.simulator.rz_position_distortion_map

    def transform_event(self, event):
        if self.rzmap is None:
            return event

        for ia in event.interactions:
            # Compute the correction
            ia.r_correction = self.rzmap.get_value(ia.r, ia.z, map_name='to_true_r')
            ia.z_correction = self.rzmap.get_value(ia.r, ia.z, map_name='to_true_z')

            # Set the new (x, y, z) position (r and phi are just python properties)
            ia.z = ia.z + ia.z_correction

            r = ia.r + ia.r_correction
            ia.x = r * np.cos(ia.phi)
            ia.y = r * np.sin(ia.phi)

        return event

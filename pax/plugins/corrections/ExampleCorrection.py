from pax import plugin, dsputils

class ExampleCorrection(plugin.TransformPlugin):

    def startup(self):
        self.correction_map = dsputils.InterpolatingDetectorMap('example_3d_correction_map.json')

    def transform_event(self, event):

        # Loop through all peaks found
        for p in event.peaks:
            # Take the first reconstructed position
            # (In real life you may want to select the position from a specific algorithm instead)
            pos = p.reconstructed_positions[0]

            # Get the correction's value
            value = self.correction_map.get_value_at(pos)

            # Print it to the debug log
            self.log.debug("Correction value at (%s, %s, %s): %s" % (pos.x, pos.y, pos.z, value))

        return event



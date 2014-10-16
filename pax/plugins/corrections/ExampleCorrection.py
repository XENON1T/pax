from pax import plugin, dsputils

class ExampleCorrection(plugin.TransformPlugin):

    def startup(self):
        self.correction_map = dsputils.CorrectionMap('example_3d_correction_map.json')

    def transform_event(self, event):

        # You may want to select a specific algorithm's position, I'll just take the first:
        pos = event.peaks[0].reconstructed_positions[0]

        # Get the correction's value
        value = self.correction_map.get_correction(pos)

        # Print it to the debug log (you may want to use it for something more constructive)
        self.log.debug("Correction value at (%s, %s, %s): %s" % (pos.x, pos.y, pos.z, value))

        return event



from pax import core, plugin, dsputils

class ExampleCorrection(plugin.TransformPlugin):

    def startup(self):
        self.correction_map = dsputils.InterpolatingMap(
             core.data_file_name('example_3d_correction_map.json'))
        # self.correction_map = dsputils.InterpolatingMap(
        #     core.data_file_name('s2_xy_lce_map_XENON100_Xerawdp0.4.5.json'))
        # self.correction_map.plot(map_name='60')

    def transform_event(self, event):

        # Loop through all peaks found
        for p in event.peaks:
            # Take the first reconstructed position
            # (In real life you may want to select the position from a specific algorithm instead)
            try:
                pos = p.reconstructed_positions[0]
            except IndexError:
                # S1s don't have a reconstructed position
                # Also, all posrec algorithms may be disabled
                continue

            # Get the correction's value
            value = self.correction_map.get_value_at(pos)
            print(type(value))

            # Print it to the debug log
            self.log.debug("Correction value at (%s, %s, %s): %s" % (pos.x, pos.y, pos.z, value))

        return event



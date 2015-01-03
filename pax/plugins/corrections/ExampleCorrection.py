from pax import core, plugin, utils

class ExampleCorrection(plugin.TransformPlugin):

    def startup(self):
        self.correction_map = utils.InterpolatingMap(
             core.data_file_name('s2_xy_lce_map_XENON100_Xerawdp0.4.5.json.gz'))
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

            # Get the map's value
            value = self.correction_map.get_value_at(pos, map_name='total_LCE')

            # Print it to the debug log
            self.log.debug("Map value at (%s, %s, %s): %s" % (pos.x, pos.y, pos.z, value))

        return event



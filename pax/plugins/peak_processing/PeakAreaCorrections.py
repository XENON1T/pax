import numpy as np
from pax import plugin, exceptions
from pax.dsputils import saturation_correction

# Must be run before 'BuildInteractions.BasicInteractionProperties'


class S2SpatialCorrection(plugin.TransformPlugin):
    """Compute S2 spatial(x,y) area correction
    """

    def startup(self):
        if 'xy_posrec_preference' not in self.config:
            raise ValueError('Configuration for %s must contain xy_posrec_preference' % self.name)
        self.s2_light_yield_map = self.processor.simulator.s2_light_yield_map

    def transform_event(self, event):

        for peak in event.peaks:
            # check that there is a position
            if not len(peak.reconstructed_positions):
                continue
            else:
                try:
                    # Get x,y position from peak
                    xy = peak.get_position_from_preferred_algorithm(self.config['xy_posrec_preference'])

                    # S2 area correction: divide by relative light yield at the position
                    peak.s2_spatial_correction /= self.s2_light_yield_map.get_value_at(xy)
                    peak.s2_top_spatial_correction /= self.s2_light_yield_map.get_value_at(xy, map_name='map_top')
                    peak.s2_bottom_spatial_correction /= self.s2_light_yield_map.get_value_at(xy, map_name='map_bottom')
                except ValueError:
                    self.log.debug("Could not find any position from the chosen algorithms")
        return event


class S2SaturationCorrection(plugin.TransformPlugin):
    """Compute S2 saturation(x,y,pmtpattern) area correction
    """

    def startup(self):
        self.s2_patterns = self.processor.simulator.s2_patterns
        self.zombie_pmts_s2 = np.array(self.config.get('zombie_pmts_s2', []))

    def transform_event(self, event):

        for peak in event.peaks:
            # check that there is a position
            if not len(peak.reconstructed_positions):
                continue
            try:
                # Get x,y position from peak
                xy = peak.get_position_from_preferred_algorithm(self.config['xy_posrec_preference'])
            except ValueError:
                self.log.debug("Could not find any position from the chosen algorithms")
                continue
            try:
                peak.s2_saturation_correction *= saturation_correction(
                    peak=peak,
                    channels_in_pattern=self.config['channels_top'],
                    expected_pattern=self.s2_patterns.expected_pattern((xy.x, xy.y)),
                    confused_channels=np.union1d(peak.saturated_channels, self.zombie_pmts_s2),
                    log=self.log)
            except exceptions.CoordinateOutOfRangeException:
                self.log.debug("Expected light pattern at coordinates "
                               "(%f, %f) consists of only zeros!" % (xy.x, xy.y))

        return event

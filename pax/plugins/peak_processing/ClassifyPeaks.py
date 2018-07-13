from pax import plugin, units


class AdHocClassification1T(plugin.TransformPlugin):

    def startup(self):
        self.s1_rise_time_bound = self.config['s1_risetime_threshold']
        self.s1_width_bound = self.config['s1_width_threshold']
        self.tight_coincidence_threshold = self.config['tight_coincidence_threshold']

    def transform_event(self, event):

        for peak in event.peaks:
            # Don't work on noise and lone hit
            if peak.type in ('noise', 'lone_hit'):
                continue
            # rounding peak aft, for future usage like BDT based classification
            if peak.area_fraction_top < 0:
                peak.area_fraction_top = 0
            elif peak.area_fraction_top > 1:
                peak.area_fraction_top = 1

            # classification based on rise_time and aft
            if -peak.area_decile_from_midpoint[1] < self.s1_rise_time_bound\
                    and peak.range_area_decile[9] < self.s1_width_bound:
                # S1 requirements: Peak rises fast, and width (90p area) small
                if peak.tight_coincidence >= self.tight_coincidence_threshold:
                    peak.type = 's1'
                else:
                    # Too few PMTs contributing, hard to distinguish from junk
                    peak.type = 'unknown'
            else:
                # No fast rise
                if peak.n_contributing_channels >= 4:
                    # Large enough to be considered as S2
                    peak.type = 's2'
                else:
                    # Too few contributing channels, not really S2
                    peak.type = 'unknown'

        return event


class AdHocClassification(plugin.TransformPlugin):

    def transform_event(self, event):

        for peak in event.peaks:
            # Don't work on noise and lone_hit
            if peak.type in ('noise', 'lone_hit'):
                continue

            width = peak.range_area_decile[5]

            if peak.area > 50:
                # We don't have to worry about single electrons anymore
                if width < 100 * units.ns:
                    peak.type = 's1'
                elif width > 250 * units.ns:
                    peak.type = 's2'
            else:
                # Worry about SE-S1 identification.
                if width < 75 * units.ns:
                    peak.type = 's1'
                else:
                    if peak.area < 5:
                        peak.type = 'coincidence'
                    elif width > 100 * units.ns:
                        peak.type = 's2'

        return event

import numpy as np
import pax.plugins.PatternFitter as PatternFitter
from pax import plugin


class PosRecTopPatternFit(plugin.PosRecPlugin):
    """Position reconstruction by optimmizing the top hitpattern goodness of fit to a model of the top hitpattern.
    Takes into account gain and QE errors.
    Additionally calculate goodness of fit for reconstructed positions by other algorithms.
    """

    def startup(self):
        self.skip_reconstruction = self.config['skip_reconstruction']
        self.seed_algorithms = self.config['seed_algorithms']
        self.statistic = self.config['statistic']

        # Set minimum area for a peak to be reconstructed -- performance option, probably no longer needed
        self.area_threshold = self.config['area_threshold']

        # Load gains (gains from config file, gain error 0.5 pe for all pmts for now)
        self.gains = np.array(self.config['gains'])[self.pmts]
        self.gain_errors = np.ones(len(self.pmts)) * 0.5  # TODO: remove placeholder

        # Load QE (for now use arbitrary values)
        self.qes = np.ones(len(self.pmts)) * 0.3  # TODO: remove placeholder
        self.qe_errors = np.ones(len(self.pmts)) * 0.009  # TODO: remove placeholder

        # Number of pmts (minus dead pmts)
        self.is_pmt_alive = self.gains > 0

        # Load the pattern fitter
        self.pf = PatternFitter.PatternFitter('s2_xy_lce_map_XENON100_Xerawdp0.4.5.json.gz')

    def reconstruct_position(self, peak):
        """Reconstruct position by optimizing hitpattern goodness of fit to per-PMT LCE map.
        Secondly, append a chi_square_gamma value and ndf to existing ReconstructedPosition objects.
        """
        # Which PMTs should we include?
        is_pmt_in = self.is_pmt_alive.copy()
        if self.config.get('ignore_saturated_PMTs', False):
            saturated_pmts = np.where(peak.n_saturated_per_channel > 0)[0]
            saturated_pmts = np.intersect1d(saturated_pmts, self.pmts)
            is_pmt_in[saturated_pmts] = False
        is_pmt_in = is_pmt_in[self.pmts]

        # Number of degrees of freedom, n_channels - model degrees of freedom (x,y) - 1
        ndf = np.count_nonzero(is_pmt_in) - 2 - 1

        # Photons observed per pmt (qe-corrected)
        areas_observed = peak.area_per_channel[self.pmts] / self.qes

        # Error term per PMT in chi2 function
        # TODO: Why is QE squared an extra time??
        square_syst_errors = areas_observed**2 * ((self.qe_errors / self.qes) ** 2 + self.gain_errors / self.gains)

        ##
        # Part 1: compute goodness of fit for positions from other algorithms
        ##
        for position in peak.reconstructed_positions:
            try:
                position.goodness_of_fit = self.pf.compute_gof(coordinates=[position.x, position.y],
                                                               areas_observed=areas_observed,
                                                               point_selection=is_pmt_in,
                                                               square_syst_errors=square_syst_errors,
                                                               statistic=self.statistic)
            except PatternFitter.CoordinateOutOfRangeException:
                # Oops, that position is impossible. Leave goodness of fit as nan
                pass
            if np.isnan(position.goodness_of_fit):
                self.log.debug("impossible position x=%s, y=%s: r=%s)" % (position.x, position.y,
                                                                          np.sqrt(position.x**2 + position.y**2)))
            position.ndf = ndf

        ##
        # PART 2 - find an even better position
        ##
        if self.skip_reconstruction:
            return None
        if peak.area < self.area_threshold:
            return None

        # Use the seed position with the most optimal (lowest, confusingly) goodness of fit as a seed
        try:
            if self.seed_algorithms == 'best':
                seed_pos = peak.reconstructed_positions[np.nanargmin([p.goodness_of_fit
                                                                      for p in peak.reconstructed_positions])]
            else:
                seed_pos = peak.get_position_from_preferred_algorithm(self.seed_algorithms)
        except ValueError:
            # All algorithms agree the event came from outside the tpc :-)
            # i.e. this peak is almost certainly very near the edge <-> top hitp dominated by one or two outer ring pmts
            # We'll just give up, nobody cares about these peaks.
            self.log.debug("All positions for peak with area %s, top hitpattern %s are nan!" % (peak.area,
                                                                                                peak.area_per_channel))
            return None

        self.log.debug('Using %s position (%0.1f, %01.f, gof %01.f) as minimizer seed' % (seed_pos.algorithm,
                                                                                          seed_pos.x, seed_pos.y,
                                                                                          seed_pos.goodness_of_fit))

        common_options = dict(areas_observed=areas_observed,
                              point_selection=is_pmt_in,
                              square_syst_errors=square_syst_errors,
                              statistic=self.statistic)
        if self.config['minimimizer'] == 'powell':
            try:
                (x, y), gof = self.pf.minimize_gof_powell(start_coordinates=(seed_pos.x, seed_pos.y),
                                                          **common_options)
            except PatternFitter.CoordinateOutOfRangeException:
                # The central position was out of range of the map! Happens occasionally if you don't use seed=best.
                return None
        else:
            (x, y), gof = self.pf.minimize_gof_grid(center_coordinates=(seed_pos.x, seed_pos.y),
                                                    grid_size=self.config['grid_size'], **common_options)

        if np.isnan(gof):
            return None

        return {'x': x,
                'y': y,
                'goodness_of_fit': gof,
                'ndf': ndf}

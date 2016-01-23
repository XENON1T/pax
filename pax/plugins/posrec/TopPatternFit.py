import numpy as np
from pax import plugin, exceptions


class PosRecTopPatternFit(plugin.PosRecPlugin):
    """Position reconstruction by optimmizing the top hitpattern goodness of fit to a model of the top hitpattern.
    Takes into account gain and QE errors.
    Additionally calculate goodness of fit for reconstructed positions by other algorithms.
    """

    def startup(self):
        self.skip_reconstruction = self.config['skip_reconstruction']
        self.seed_algorithms = self.config['seed_algorithms']
        self.statistic = self.config['statistic']
        self.is_pmt_alive = np.array(self.config['gains']) > 0

        # Load the S2 hitpattern fitter
        self.pf = self.processor.simulator.s2_patterns

    def reconstruct_position(self, peak):
        """Reconstruct position by optimizing hitpattern goodness of fit to per-PMT LCE map.
        Secondly, append a goodness_of_fit value and ndf to existing ReconstructedPosition objects.
        """
        # Which PMTs should we include?
        is_pmt_in = self.is_pmt_alive.copy()
        if self.config.get('ignore_saturated_PMTs', False):
            saturated_pmts = np.where(peak.n_saturated_per_channel > 0)[0]
            saturated_pmts = np.intersect1d(saturated_pmts, self.pmts)
            is_pmt_in[saturated_pmts] = False
        is_pmt_in = is_pmt_in[self.pmts]

        # Check if any pmts are in -- if all living PMTs saturated we can't reconstruct a position
        if not np.sum(is_pmt_in):
            self.log.warning("All living PMTs for peak %d-%d are saturated... Can't reconstruct a position." % (
                peak.left, peak.right))
            return None

        # Number of degrees of freedom, n_channels - model degrees of freedom (x,y) - 1
        ndf = np.count_nonzero(is_pmt_in) - 2 - 1

        # Pe observed per pmt. Don't QE correct: pattern map has been adjusted for QE already
        areas_observed = peak.area_per_channel[self.pmts]

        ##
        # Part 1: compute goodness of fit for positions from other algorithms
        ##
        for position in peak.reconstructed_positions:
            try:
                position.goodness_of_fit = self.pf.compute_gof(coordinates=[position.x, position.y],
                                                               areas_observed=areas_observed,
                                                               pmt_selection=is_pmt_in,
                                                               statistic=self.statistic)
                position.ndf = ndf
            except exceptions.CoordinateOutOfRangeException:
                # Oops, that position is impossible. Leave goodness of fit as nan
                self.log.debug("impossible position x=%s, y=%s: r=%s)" % (position.x, position.y,
                                                                          np.sqrt(position.x**2 + position.y**2)))

        ##
        # Part 2 - find an even better position...
        ##
        if self.skip_reconstruction:
            return None

        # Use the seed position with the most optimal (lowest, confusingly) goodness of fit as a seed
        if not peak.reconstructed_positions:
            raise ValueError("TopPatternFit needs at least one seed position: please run at least MaxPMT...")
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
                              pmt_selection=is_pmt_in,
                              statistic=self.statistic)
        if self.config['minimizer'] == 'powell':
            try:
                (x, y), gof = self.pf.minimize_gof_powell(start_coordinates=(seed_pos.x, seed_pos.y),
                                                          **common_options)
            except exceptions.CoordinateOutOfRangeException:
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

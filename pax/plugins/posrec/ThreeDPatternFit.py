import numpy as np
from pax import plugin, exceptions


class PosRecThreeDPatternFit(plugin.PosRecPlugin):
    uses_only_top = False

    def startup(self):
        self.is_pmt_alive = np.array(self.config['gains']) > 0
        self.pf = self.processor.simulator.s1_patterns
        self.config.setdefault('minimizer', 'grid')
        self.config.setdefault('statistic', 'chi2gamma')
        self.config.setdefault('only_s1s', True)

    def reconstruct_position(self, peak):
        """Reconstruct position by optimizing hitpattern goodness of fit to per-PMT LCE map."""
        if self.config['only_s1s'] and peak.type != 's1':
            return None

        # Which PMTs should we include?
        is_pmt_in = self.is_pmt_alive.copy()
        if self.config.get('ignore_saturated_PMTs', False):
            saturated_pmts = np.where(peak.n_saturated_per_channel > 0)[0]
            saturated_pmts = np.intersect1d(saturated_pmts, self.pmts)
            is_pmt_in[saturated_pmts] = False
        is_pmt_in = is_pmt_in[self.pmts]

        # Number of degrees of freedom, n_channels - model degrees of freedom (x,y) - 1
        ndf = np.count_nonzero(is_pmt_in) - 2 - 1

        # Pe observed per pmt. Don't QE correct: pattern map has been adjusted for QE already
        areas_observed = peak.area_per_channel[self.pmts]

        # For now just take a TPC-wide grid... not very good for performance!!
        z_mid = self.config['tpc_length'] / 2
        grid_size = 4 * max(z_mid, self.config['tpc_radius'])

        common_options = dict(areas_observed=areas_observed,
                              pmt_selection=is_pmt_in,
                              statistic=self.config['statistic'])
        if self.config['minimizer'] == 'powell':
            try:
                (x, y, z), gof = self.pf.minimize_gof_powell(start_coordinates=(0, 0, z_mid),
                                                             **common_options)
            except exceptions.CoordinateOutOfRangeException:
                # The central position was out of range of the map! Happens occasionally if you don't use seed=best.
                return None
        else:
            (x, y, z), gof, err = self.pf.minimize_gof_grid(center_coordinates=(0, 0, z_mid),
                                                            grid_size=grid_size, **common_options)
        if np.isnan(gof):
            return None

        return {'x': x, 'y': y, 'z': z,
                'goodness_of_fit': gof,
                'ndf': ndf,
                'confidence_tuples': err}
                

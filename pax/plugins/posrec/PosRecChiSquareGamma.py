"""Position reconstruction algorithm using chi square gamma distribution minimization"""

import numpy as np
from scipy.optimize import fmin_powell

from pax import plugin
from pax.datastructure import ReconstructedPosition


class PosRecChiSquareGamma(plugin.TransformPlugin):

    """Position reconstruction by minimization of chi square gamma function.

    Class to reconstruct S2's x, y and chi square gamma
    (as 'goodness-of-fit' parameter) using the S2 signal in top PMTs, simulated
    LCE map, gains and QE's.

    Additionally calculate chi square gamma for reconstructed positions by other
    algorithms.
    """

    def startup(self):
        """Initialize reconstruction algorithm

        Determine which PMTs to use for reconstruction.
        Load LCE maps, gains and QE's
        """

        # Set mode for algorithm
        self.mode = self.config['mode']
        self.log.debug("Startup, mode: %s" % self.mode)
        if not self.mode == 'no_reconstruct' and not self.mode == 'only_reconstruct' and not self.mode == 'full':
            raise RuntimeError("Bad choice 'mode'")

        # List of integers of which PMTs to use, this algorithm uses the top pmt array to reconstruct
        self.pmts = self.config['channels_top']

        # (x,y) Locations of these PMTs, stored as np.array([(x,y), (x,y), ...])
        self.pmt_locations = np.zeros((len(self.pmts), 2))
        for ch in self.pmts:
            for dim in ('x', 'y'):
                self.pmt_locations[ch][{'x': 0, 'y': 1}[dim]] = self.config['pmt_locations'][ch][dim]

        # Set area threshold, minimum area for a peak to be reconstructed
        self.area_threshold = self.config['area_threshold']

        # Set the TPC radius squared, add 1% to avoid edge effects
        self.tpc_radius_squared = (self.config['tpc_radius'] * 1.01)**2

        # Load LCE maps
        self.s2_lce_map = self.processor.simulator.s2_lce_map

        # Load gains (gains from config file, gain error 0.5 pe for all pmts for now)
        self.gains = np.array(self.config['gains'])[self.pmts]
        self.gain_errors = np.ones(len(self.pmts)) * 0.5  # TODO: remove placeholder

        # Load QE (for now use arbitrary values)
        self.qes = np.ones(len(self.pmts)) * 0.3  # TODO: remove placeholder
        self.qe_errors = np.ones(len(self.pmts)) * 0.009  # TODO: remove placeholder

        # Number of pmts (minus dead pmts)
        self.is_pmt_alive = self.gains > 0

        # Log the total number of calls and success rate to debug, see shutdown
        self.total_rec_calls = 0
        self.total_rec_success = 0

    def function_chi_square_gamma(self, position):
        """Return Chi Square Gamma value for position x,y

        This function is passed to the minimizer, the lowest
        function value will yield the reconstructed position"""

        x = position[0]
        y = position[1]

        # Cutoff value at TPC radius, chi_square_gamma is +infinity here
        if x ** 2 + y ** 2 > self.tpc_radius_squared:
            return float('inf')

        # Get all LCE map values for the live PMTs at position x,y
        map_values = self.s2_lce_map.get_value(x, y)[self.is_pmt_in]

        # Convert to relative LCEs among included PMTs
        map_values = np.clip(map_values, 0, 1)
        map_values /= map_values.sum()

        # Compute the chi2gamma for each PMT, add up contributions from living PMTs
        term_numerator = (self.photons + np.clip(self.photons, 1, float('inf')) - self.area_photons * map_values) ** 2
        term_denominator = self.photons ** 2 * self.pmt_errors + self.area_photons * map_values + 1.0
        function_values = term_numerator / term_denominator

        # assert len(function_values) == np.count_nonzero(self.is_pmt_in)

        return np.sum(function_values)

    def transform_event(self, event):
        """Reconstruct the position of S2s in an event.

        For each S2 peak, append a chi_square_gamma value and ndf to existing
        ReconstructedPosition objects.
        Secondly minimize and find the minimum chi_square_gamma and reconstructed
        positions and append a ReconstructedPosition object for those.
        """

        # For every S2 peak found in the event
        for peak in event.S2s():

            # Which PMTs should we include?
            self.is_pmt_in = self.is_pmt_alive.copy()
            if self.config.get('ignore_saturated_PMTs', False):
                saturated_pmts = np.where(peak.n_saturated_per_channel > 0)[0]
                self.is_pmt_in[saturated_pmts] = False
            self.is_pmt_in = self.is_pmt_in[self.pmts]

            # Number of degrees of freedom, n_channels - model degrees of freedom (x,y) - 1
            self.ndf = np.count_nonzero(self.is_pmt_in) - 2 - 1

            # This is an array where every i-th element is how many pe
            # were seen by the i-th PMT
            self.hits = peak.area_per_channel[self.is_pmt_in]
            self.photons = self.hits / self.qes[self.is_pmt_in]

            # Total number of detected photons in the top array (pe/qe)
            self.area_photons = self.photons.sum()

            # Error term per PMT in chi2 function
            self.pmt_errors = (self.qe_errors[self.is_pmt_in] / self.qes[self.is_pmt_in]) ** 4
            self.pmt_errors += (self.gain_errors[self.is_pmt_in] / self.gains[self.is_pmt_in]) ** 2

            # Calculate which pmt has maximum signal
            pmts_in = np.array(self.pmts)[self.is_pmt_in]
            max_pmt_index = pmts_in[np.argmax(self.photons)]

            # Start position for minimizer, if no weighted sum position is present
            # use max pmt location as minimizer start position
            start_x = self.pmt_locations[max_pmt_index][0]
            start_y = self.pmt_locations[max_pmt_index][1]

            if not self.mode == 'only_reconstruct':
                # Check for reconstructed positions by other algorithms, if found, calculate chi_square_gamma for those
                for position in peak.reconstructed_positions:
                    if position.algorithm == self.name:
                        continue

                    position.goodness_of_fit = self.function_chi_square_gamma([position.x, position.y])
                    position.ndf = self.ndf
                    self.log.debug("Found reconstructed position by other algorithm"
                                   " %s x: %f y: %f, appending chi_square_gamma: %f ndf: %d"
                                   % (position.algorithm, position.x, position.y, position.goodness_of_fit, self.ndf))

                    # If a neural net position is already calculated for this peak, use it as start position
                    if position.algorithm == 'NeuralNet' \
                            and not self.mode == 'no_reconstruct' \
                            and self.config.get('seed_from_neural_net'):
                        self.log.debug('Using NeuralNet position as minimizer start position')

                        start_x = position.x
                        start_y = position.y

            if self.mode == 'no_reconstruct':
                return event

            # Only reconstruct peak if it has an area of more then area_threshold pe
            # Setting a very low threshold is bad for speed
            if peak.area < self.area_threshold:
                self.log.debug("Peak area below threshold, skipping to the next peak")
                continue

            # Set initial search direction of the minimizer to center to avoid edge effects
            s = lambda d: 1 if d < 0 else -1

            direc = np.array([[s(start_x), 0],
                              [0, s(start_y)]])

            # Start minimization
            self.log.debug("Starting minimizer for position reconstruction")

            # Minimize chi_square_gamma function, fmin_powell is the call to the SciPy minimizer
            # It takes the function to minimize, starting position and several options
            # It returns the optimal values for the position (xopt) and function value (fopt)
            # A warnflag tells if the maximum number of iterations was exceeded
            #    warnflag 0, OK
            #    warnflag 1, maximum functions evaluations exceeded
            #    warnflag 2, maximum iterations exceeded
            xopt, fopt, direc, iter, funcalls, warnflag = fmin_powell(self.function_chi_square_gamma,
                                                                      [start_x, start_y],
                                                                      args=(),
                                                                      xtol=0.0001,
                                                                      ftol=0.0001,
                                                                      maxiter=10,
                                                                      maxfun=None,
                                                                      full_output=1,
                                                                      disp=0,
                                                                      direc=direc,
                                                                      retall=0)

            self.log.debug("Minimizer warnflag: %d" % warnflag)

            self.total_rec_calls += 1
            if not warnflag:
                self.total_rec_success += 1

            x = float(xopt[0])
            y = float(xopt[1])
            chi_square_gamma = float(fopt)

            # Correct strange error where numpy arrays are passed as fopt
            if isinstance(chi_square_gamma, np.ndarray):
                self.log.warning("Help! Optimizer gave me an ndarray instead of float... "
                                 "To be precise, I got this: %s" % str(chi_square_gamma))
                chi_square_gamma = float('nan')

            # If the minimizer failed do not report a position
            if chi_square_gamma == float('nan'):
                x = y = float('nan')

            self.log.debug("Reconstructed event at x: %f y: %f chi_square_gamma:"
                           " %f ndf: %d" % (x, y, chi_square_gamma, self.ndf))

            # Create a reconstructed position object
            rp = ReconstructedPosition({'x': x,
                                        'y': y,
                                        'goodness_of_fit': chi_square_gamma,
                                        'ndf': self.ndf,
                                        'algorithm': self.name})

            # Append our reconstructed position object to the peak
            peak.reconstructed_positions.append(rp)

        # Return the event such that the next plugin can work on it
        return event

    def shutdown(self):
        if not self.total_rec_calls:
            return

        self.log.debug("Total number of reconstruct calls: %d" % self.total_rec_calls)
        self.log.debug("Success rate: %f" % (self.total_rec_success / self.total_rec_calls))

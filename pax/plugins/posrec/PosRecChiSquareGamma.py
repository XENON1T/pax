"""Position reconstruction algorithm using chi square gamma distribution minimization"""
from scipy.optimize import fmin_powell

from pax import plugin

from pax.datastructure import ReconstructedPosition
from pax.utils import InterpolatingMap
from pax.core import data_file_name


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

        # (x,y) Locations of these PMTs.  This is stored as a dictionary such
        # that self.pmt_locations[int] = {'x' : int, 'y' : int, 'z' : None}
        self.pmt_locations = self.config['pmt_locations']

        # Load LCE maps
        self.lce_map_file_name = self.config['lce_map_file_name']
        self.maps = InterpolatingMap(data_file_name(self.lce_map_file_name))

        # Load gains (gains from config file, gain error 0.5 pe for all pmts for now)
        self.gains = self.config['gains']
        self.gain_errors = [0.5 for i in range(99)]  # test for now, make config variable

        # Load QE (for now use arbitrary values)
        self.qes = [0.3 for i in range(99)]  # test for now, make config variable
        self.qe_errors = [0.009 for i in range(99)]  # test for now, make config variable

        # Number of pmts (minus dead pmts)
        self.n_channels = 0
        for pmt in self.pmts:
            if not self.gains[pmt] == 0:
                self.n_channels += 1

        # Number of degrees of freedom, n_channels - model degrees of freedom (x,y) - 1
        self.ndf = self.n_channels - 2 - 1

        # Log the total number of calls and success rate to debug, see shutdown
        self.total_rec_calls = 0
        self.total_rec_success = 0

    def function_chi_square_gamma(self, position):
        """Return Chi Square Gamma value for position x,y

        This function is passed to the minimizer, the lowest
        function value will yield the reconstructed position"""

        x = position[0]
        y = position[1]

        # Cutoff value at r=15 cm, chi_square_gamma is +infinity here
        if x ** 2 + y ** 2 > 225:
            return float('inf')

        function_value = 0

        # Iterate over all pmts, adding each pmts contribution to function_value
        for pmt in self.pmts:
            # Exclude dead pmts
            if self.gains[pmt] == 0:
                continue

            photons_in_pmt = self.hits[pmt] / self.qes[pmt]
            pmt_error = (self.qe_errors[pmt] / self.qes[pmt]) ** 4 + (self.gain_errors[pmt] / self.gains[pmt]) ** 2

            # Lookup value from LCE map for pmt at position x,y
            map_value = self.maps.get_value(x, y, map_name=str(pmt))

            term_numerator = (photons_in_pmt + min(photons_in_pmt, 1) - self.area_photons * map_value) ** 2
            term_denominator = photons_in_pmt ** 2 * pmt_error + self.area_photons * map_value + 1.0

            function_value += term_numerator / term_denominator

        return function_value

    def transform_event(self, event):
        """Reconstruct the position of S2s in an event.

        For each S2 peak, append a chi_square_gamma value and ndf to existing
        ReconstructedPosition objects.
        Secondly minimize and find the minimum chi_square_gamma and reconstructed
        positions and append a ReconstructedPosition object for those.
        """

        # For every S2 peak found in the event
        for peak in event.S2s():
            # This is an array where every i-th element is how many pe
            # were seen by the i-th PMT
            self.hits = peak.area_per_channel

            # Total number of detected photons in the top array (pe/qe)
            self.area_photons = 0.0

            # Highest number of pe seen in one PMT and the id of this PMT
            max_pmt = 0
            max_pmt_id = 0

            # Calculate which pmt has maximum signal
            for pmt in self.pmts:
                # Exclude dead pmts
                if self.gains[pmt] == 0 or str(self.hits[pmt]) == 'nan' or self.qes[pmt] == 0:
                    continue

                if self.hits[pmt] > max_pmt:
                    max_pmt = self.hits[pmt]
                    max_pmt_id = pmt

                self.area_photons += self.hits[pmt] / self.qes[pmt]

            # Start position for minimizer, if no weighted sum position is present
            # use max pmt location as minimizer start position
            start_x = self.pmt_locations[max_pmt_id]['x']
            start_y = self.pmt_locations[max_pmt_id]['y']

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

                    # If a weighted sum is already calculated for this peak, use it as start position
                    if position.algorithm == 'PosRecWeightedSum' and not self.mode == 'no_reconstruct':
                        self.log.debug('Using weighted sum by PosRecWeightedSum as minimizer start position')

                        start_x = position.x
                        start_y = position.y

            if self.mode == 'no_reconstruct':
                return event

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
                                                                      maxiter=5,
                                                                      maxfun=None,
                                                                      full_output=1,
                                                                      disp=0,
                                                                      direc=None,
                                                                      retall=0)

            self.log.debug("Minimizer warnflag: %d" % warnflag)

            self.total_rec_calls += 1
            if not warnflag:
                self.total_rec_success += 1

            x = xopt[0]
            y = xopt[1]
            chi_square_gamma = fopt

            self.log.debug("Reconstructed event at x: %f y: %f chi_square_gamma:"
                           " %f ndf: %d" % (x, y, chi_square_gamma, self.ndf))

            # Create a reconstructed position object
            rp = ReconstructedPosition({'x': x,
                                        'y': y,
                                        'z': float('nan'),
                                        'goodness_of_fit': chi_square_gamma,
                                        'ndf': self.ndf,
                                        'index_of_maximum': peak.index_of_maximum,
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

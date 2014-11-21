"""Position reconstruction algorithm using chi square gamma distribution"""
import numpy as np
from scipy.optimize import minimize
from scipy.misc import derivative

from pax import plugin

from pax.datastructure import ReconstructedPosition
from pax.dsputils import InterpolatingMap
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

        #Set mode for algorithm, this can be:
        #    -'no_reconstruct', to only append a chi square gamma to existing reconstructed positions
        #    -'only_reconstruct', to add the reconstructed position object for this algorithm but leave others alone
        #    -'full', to do both
        self.mode = self.config['mode']
        print(self.mode)
        if not self.mode == 'no_reconstruct' and not self.mode == 'only_reconstruct' and not self.mode == 'full':
            raise RuntimeError("Bad choice 'mode', using mode = 'full'")

        # This can either be 'top' or 'bottom'.
        self.which_pmts = 'pmts_%s' % self.config['pmts_to_use_for_reconstruction']
        if self.which_pmts not in self.config.keys():
            raise RuntimeError("Bad choice 'pmts_to_use_for_reconstruction'")

        # List of integers of which PMTs to use (WHY does this include 0, there is no such pmt!)
        self.pmts = self.config[self.which_pmts]

        # (x,y) locations of these PMTs.  This is stored as a dictionary such
        # that self.pmt_locations[int] = {'x' : int, 'y' : int, 'z' : None}
        self.pmt_locations = self.config['pmt_locations']

		#Load LCE maps (check if vals are response or probability, check rotation of map)
        self.lce_map_file_name = self.config['lce_map_file_name']
        self.maps = InterpolatingMap(data_file_name(self.lce_map_file_name))

        #Load gains (gains from config file, gain error 0.5 pe for all pmts for now)
        self.gains = self.config['gains']
        self.gain_errors = [0.5 for i in range(99)]

        #Load QE (for now use arbitrary values)
        self.qes = [0.3 for i in range(99)]
        self.qe_errors = [0.0 for i in range(99)] #test with 0 uncertainty (as was also done in libchi)

    def function_chi_square_gamma(self, position):
        """Return Chi Square Gamma value for position x,y"""

        #apparently coordinates are in cm (why), the lce map uses mm
        #convert to mm, also lce map uses a different pmt numbering effectively rotating 90deg anti-clockwise, rotate back 90deg  yes/no?
        x = position[1] * 10  #position[0] * 10 * cos(-90) - position[1] * 10 * sin(-90)
        y = position[0] * -10  #position[0] * 10 * sin(-90) + position[1] * 10 * cos(-90)

        function_value = 0

        #Does self.pmts contain dead pmts? If so this code works, if not the range should be range(1,99)
        for pmt in self.pmts:
            #why do we even have a non existing pmt in the list of pmts to use!
            if pmt == 0:
                continue

            if self.qes[pmt] == 0 or self.gains[pmt] == 0:
                photons_in_pmt = 0
                pmt_error = 0
            else:
                photons_in_pmt = self.hits[pmt]/self.qes[pmt]
                pmt_error = (self.qe_errors[pmt] / self.qes[pmt])**4 + (self.gain_errors[pmt]/self.gains[pmt])**2

            term_numerator = (photons_in_pmt + min(photons_in_pmt,1) - self.area_photons * self.maps.get_value(x,y,map_name=str(pmt)))**2
            term_denominator = photons_in_pmt**2 * pmt_error + self.area_photons * self.maps.get_value(x,y,map_name=str(pmt)) + 1

            function_value += term_numerator / term_denominator

        return function_value

    def jac_chi_square_gamma(self, position):
        """Gradient of chi_square_gamma function at position (x,y)
        returns numpy array with derivatives"""

        #first, very crude implementation, just to be able to use newton-CG as minimizer

        x = position[0]
        y = position[1]
        delta = 0.01 #0.1mm

        dfdx = (self.function_chi_square_gamma([x - delta/2, y])-self.function_chi_square_gamma([x + delta/2, y]))/delta
        dfdy = (self.function_chi_square_gamma([x, y - delta/2])-self.function_chi_square_gamma([x, y + delta/2]))/delta

        return np.array([dfdx, dfdy])

    def transform_event(self, event):
        """Reconstruct the position of S2s in an event.

        Information on how to use the 'event' object is at:

          http://xenon1t.github.io/pax/format.html
        """

        # For every S2 peak found in the event
        for peak in event.S2s():
            # This is an array where every i-th element is how many pe
            # were seen by the i-th PMT
            # store with class scope for now
            self.hits = peak.area_per_pmt
            self.area_photons = 0.0

            max_pmt = 0
            max_pmt_id = 0

            for pmt in self.pmts:
                if pmt == 0:
                    continue

                if self.hits[pmt] > max_pmt:
                    max_pmt = self.hits[pmt]
                    max_pmt_id = pmt

                if self.qes[pmt] == 0 or str(self.hits[pmt]) == 'nan': #this check should not be nessesary in a good language, why o why python
                    continue

                self.area_photons += self.hits[pmt] / self.qes[pmt]

            if not self.mode == 'only_reconstruct':
                #check for reconstructed positions by other algorithms, if found, calculate chi_square_gamma for those
                for position in peak.reconstructed_positions:
                    if position.algorithm == self.name:
                        continue
                    print(position.x, position.y) # what units are these? cm probably
                    position.chi_square_gamma = self.function_chi_square_gamma([position.x, position.y])
                    print("found reconstructed position by other algorithm, appending chi square gamma", position.chi_square_gamma)


            if self.mode == 'no_reconstruct':
                return event

            #Start reconstruction
            print("start own reconstruction")

            #assume pmt with highest energy as start position
            print("max pmt id", max_pmt_id)
            #print("area photons", self.area_photons)
            #print("area", peak.area)

            #units are in cm here
            start_x = self.pmt_locations[max_pmt_id]['x']
            start_y = self.pmt_locations[max_pmt_id]['y']

            #Minimize chi_square_gamma function
            minimize_result = minimize(self.function_chi_square_gamma, [start_x, start_y], method='Newton-CG', jac=self.jac_chi_square_gamma, options={'xtol' : 0.5, 'eps': [0.1,0.1], 'disp' : True})
            
            print("Minimization success: ", minimize_result.success)
            if not minimize_result.success:
                print(minimize_result.message)

            x = minimize_result.x[0]
            y = minimize_result.x[1]
            chi_square_gamma = self.function_chi_square_gamma([x,y])

            print("start point func val", self.function_chi_square_gamma([start_x, start_y]))
            print("reconstructed x", x)
            print("reconstructed_y", y)
            print("chi square gamma", chi_square_gamma)

            # Create a reconstructed position object
            rp = ReconstructedPosition({'x': x,
                                        'y': y,
                                        'z': 42,
                                        'chi_square_gamma': chi_square_gamma,
                                        'index_of_maximum': peak.index_of_maximum,
                                        'algorithm': self.name})

            # Append our reconstructed position object to the peak
            peak.reconstructed_positions.append(rp)

        # Return the event such that the next plugin can work on it
        return event

    def shutdown(self):
        pass

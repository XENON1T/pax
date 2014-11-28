"""Position reconstruction algorithm using chi square gamma distribution"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from scipy.optimize import minimize
from scipy.optimize import fmin_powell

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
        self.log.debug("Startup, mode: %s" % self.mode)
        if not self.mode == 'no_reconstruct' and not self.mode == 'only_reconstruct' and not self.mode == 'full':
            raise RuntimeError("Bad choice 'mode', using mode = 'full'")

        # This can either be 'top' or 'bottom'.
        self.which_pmts = 'pmts_%s' % self.config['pmts_to_use_for_reconstruction']
        if self.which_pmts not in self.config.keys():
            raise RuntimeError("Bad choice 'pmts_to_use_for_reconstruction'")

        # List of integers of which PMTs to use (WHY does this include 0, there is no such pmt!)
        self.pmts = self.config[self.which_pmts]
        #number of pmts, assumes that pmt 0 is included hence the -1
        self.n_pmts = len(self.pmts) - 1

        # (x,y) locations of these PMTs.  This is stored as a dictionary such
        # that self.pmt_locations[int] = {'x' : int, 'y' : int, 'z' : None}
        self.pmt_locations = self.config['pmt_locations']

		#Load LCE maps (check if vals are response or probability, check rotation of map)
        self.lce_map_file_name = self.config['lce_map_file_name']
        self.maps = InterpolatingMap(data_file_name(self.lce_map_file_name))

        #Load gains (gains from config file, gain error 0.5 pe for all pmts for now)
        self.gains = self.config['gains']
        self.gain_errors = [0.5 for i in range(self.n_pmts+1)] #test, get from config

        #Load QE (for now use arbitrary values)
        self.qes = [0.3 for i in range(self.n_pmts+1)] #test, get from config
        self.qe_errors = [0.0 for i in range(self.n_pmts+1)] #test with 0 uncertainty (as was also done in libchi)

        #Some debug values
        self.total_rec_calls = 0
        self.total_rec_success = 0

    def plot_s2(self, x, y, max_pmt_x, max_pmt_y, x_plot, y_plot):
        """Plot the signal pmts and reconstructed positions on top of a heatmap of chi_square_gamma"""

        x_bin = np.linspace(-15, 15, 30)
        y_bin = np.linspace(-15, 15, 30)
        z_bin = np.empty([30,30])

        for i in range(0,30):
            for j in range(0,30):
                z_bin[i][j] = self.function_chi_square_gamma([x_bin[i]+0.5,y_bin[j]+0.5])

        plt.pcolor(x_bin, y_bin, z_bin)
        plt.axis([x_bin.min(), x_bin.max(), y_bin.min(), y_bin.max()])
        plt.title('S2 hit pattern and reconstructed positions')
        plt.xlabel('x [cm]')
        plt.ylabel('y [cm]')
        plt.colorbar().set_label(r'$\chi^{2}_{\gamma}$',rotation=0)
        plt.gcf().gca().add_artist(plt.Circle((0,0),15,color='black',fill=False))

        for i in self.pmts:
            if i == 0:
                continue
            if self.hits[i] <  0.5:
                continue

            plt.plot([self.pmt_locations[i]['x']],[self.pmt_locations[i]['y']],'bo')
            plt.annotate("%.0f" % self.hits[i],(self.pmt_locations[i]['x'],self.pmt_locations[i]['y']))

        plt.plot([max_pmt_x],[max_pmt_y],'ro')

        plt.plot([x_plot],[y_plot],'go')
        plt.annotate('PosSimple',(x_plot,y_plot))
        plt.plot([x],[y],'go')
        plt.annotate('PosCSG',(x,y))

        plt.show()

    def function_chi_square_gamma(self, position):
        """Return Chi Square Gamma value for position x,y"""

        #lce map uses a different pmt numbering effectively rotating 90deg anti-clockwise, rotate back 90deg  yes? Turns out... NO
        #x = position[1] * 1  #position[0] * cos(-90) - position[1] * sin(-90)
        #y = position[0] * -1  #position[0] * sin(-90) + position[1] * cos(-90)

        #Since this is depends on lce_map pmt numbering convention, move to config maybe? At least dont use it in this function were
        #it doesnt belong

        #this is the correct rotation, distribution matches hitpattern
        x = position[0] * 1
        y = position[1] * -1

        #cutoff value at r=15
        if x**2 + y**2 > 225:
            return 0 #maybe make this +inf so the minimizer will never find a value for r>15, but plots dont look nice so not now

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

            #Lookup value from LCE map for pmt at position x,y
            map_value_scaled = self.maps.get_value(x,y,map_name=str(pmt))
            #Normalize back (ultimately for speed do this offline, basically the original map)
            scale = self.maps.get_value(x,y,map_name='total_LCE')
            if not scale < 0.0001:
                map_value = map_value_scaled / scale
            else:
                map_value = 0.0

            #map_value = map_value_scaled

            #libchi leaves out saturated pmts and normalizes again, why put a saturated pmt to 0 signal?

            term_numerator = (photons_in_pmt + min(photons_in_pmt,1) - self.area_photons * map_value)**2
            term_denominator = photons_in_pmt**2 * pmt_error + self.area_photons * map_value + 1.0

            function_value += term_numerator / term_denominator

        return function_value

    def transform_event(self, event):
        """Reconstruct the position of S2s in an event.

        Information on how to use the 'event' object is at:

          http://xenon1t.github.io/pax/format.html
        """

        # For every S2 peak found in the event
        for peak in event.S2s():
            # This is an array where every i-th element is how many pe
            # were seen by the i-th PMT
            self.hits = peak.area_per_pmt
            # total number of detected photons in the top array (pe/qe)
            self.area_photons = 0.0
            #highest number of pe seen in one pmt
            max_pmt = 0
            #id of the max_pmt pmt
            max_pmt_id = 0
            
            #number of degrees of freedom, number of pmts - number of dead pmts - model degrees of freedom (x,y) - 1
            ndf = self.n_pmts - 2 - 1

            for pmt in self.pmts:
                if pmt == 0:
                    continue

                if self.hits[pmt] > max_pmt:
                    max_pmt = self.hits[pmt]
                    max_pmt_id = pmt

                if self.qes[pmt] == 0 or str(self.hits[pmt]) == 'nan': #this check should not be nessesary, why can this value be 'nan'
                    ndf -= 1 #non contributing pmt (dead), so 1 dof less
                    continue

                self.area_photons += self.hits[pmt] / self.qes[pmt]

            #if no weighted sum position is present, use max pmt location as minimizer start position 
            start_x = max_pmt_x = self.pmt_locations[max_pmt_id]['x']
            start_y = max_pmt_y = self.pmt_locations[max_pmt_id]['y']

            if not self.mode == 'only_reconstruct':
                #check for reconstructed positions by other algorithms, if found, calculate chi_square_gamma for those
                for position in peak.reconstructed_positions:
                    if position.algorithm == self.name:
                        continue
                    if position.algorithm == 'PosRecWeightedSum':
                        self.log.debug('Using weighted sum by PosRecWeightedSum as minimizer start position')
                        start_x = position.x
                        start_y = position.y

                    x_plot = position.x #plot test
                    y_plot = position.y #plot test

                    position.chi_square_gamma = self.function_chi_square_gamma([position.x, position.y])
                    position.ndf = ndf
                    self.log.debug("Found reconstructed position by other algorithm %s x: %f y: %f, appending chi_square_gamma: %f ndf: %d" % (position.algorithm, position.x, position.y, position.chi_square_gamma, ndf))

            if self.mode == 'no_reconstruct':
                return event

            #Start minimization
            self.log.debug("Starting minimizer for position reconstruction")

            #Minimize chi_square_gamma function
            #default vals, xtol=0.0001, ftol=0.0001
            xopt, fopt, direc, iter, funcalls, warnflag = fmin_powell(self.function_chi_square_gamma,
                                                                      [start_x, start_y],
                                                                      args=(),
                                                                      xtol=1,
                                                                      ftol=1,
                                                                      maxiter=5,
                                                                      maxfun=50,
                                                                      full_output=1,
                                                                      disp=0,
                                                                      direc=None)
            
            #checks on output, success of not?
            self.log.debug("Minimizer warnflag: %d" % warnflag)

            self.total_rec_calls += 1
            if not warnflag:
                self.total_rec_success += 1

            x = xopt[0]
            y = xopt[1]
            chi_square_gamma = fopt

            #PLOT option (warning, very slow)
            #self.plot_s2(x, y, max_pmt_x, max_pmt_y, x_plot, y_plot)

            self.log.debug("Reconstructed event at x: %f y: %f chi_square_gamma: %f ndf: %d" % (x, y, chi_square_gamma, ndf))

            # Create a reconstructed position object
            rp = ReconstructedPosition({'x': x,
                                        'y': y,
                                        'z': float('nan'),
                                        'chi_square_gamma': chi_square_gamma,
                                        'ndf': ndf,
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
        self.log.debug("Success rate: %f" % (self.total_rec_success/self.total_rec_calls))

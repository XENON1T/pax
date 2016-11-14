from __future__ import division
from collections import namedtuple
import json
import gzip
import re
import logging

import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
from matplotlib import _cntr
from scipy.optimize import fmin_powell
from scipy.ndimage.interpolation import zoom as image_zoom

from pax import utils
from pax.exceptions import CoordinateOutOfRangeException
from pax.datastructure import ConfidenceTuple

# Named tuple for coordinate data storage
# Maybe works faster than dictionary... can always remove later
CoordinateData = namedtuple('CoordinateData', ('minimum', 'maximum', 'n_points', 'point_spacing'))


class PatternFitter(object):

    def __init__(self, filename, zoom_factor=1, adjust_to_qe=None, default_errors=None):
        """Initialize a pattern map file from filename.
        Format of the file is very similar to InterpolatingMap; a (gzip compressed) json containing:
            'coordinate_system' :   [['x', (x_min, x_max, n_x)], ['y',...
            'map' :                 [[[valuex1y1pmt1, valuex1y1pmt2, ...], ...], ...]
            'name':                 'Nice file with maps',
            'description':          'Say what the maps are, who you are, your favorite food, etc',
            'timestamp':            unix epoch seconds timestamp
        where x_min is the lowest x coordinate of a point, x_max the highest, n_x the number of points
        zoom_factor is factor by which the spatial dimensions of the map will be upsampled.

        adjust_to_qe: array of same length as the number of pmts in the map;
            we'll adjust the patterns to account for these QEs, upweighing PMTs with higher QEs
            Obviously this should be None if map already includes QE effects (e.g. if it is data-derived)!

        default_errors: array of the same length as the number of pmts in the map;
            This is the default factor which will be applied to obtain the squared systematic errors in the goodness
            of fit statistic, as follows:
                squared_systematic_errors = (areas_observed * default_errors)**2
        """
        self.log = logging.getLogger('PatternFitter')
        with gzip.open(utils.data_file_name(filename)) as infile:
            json_data = json.loads(infile.read().decode())

        self.data = np.array(json_data['map'])
        self.log.debug('Loaded pattern file named: %s' % json_data['name'])
        self.log.debug('Description:\n    ' + re.sub(r'\n', r'\n    ', json_data['description']))
        self.log.debug('Data shape: %s' % str(self.data.shape))
        self.log.debug('Will zoom in by factor %s' % zoom_factor)
        self.dimensions = len(json_data['coordinate_system'])    # Spatial dimensions (other one is sampling points)

        # Zoom the spatial map using linear interpolation, if desired
        if zoom_factor != 1:
            self.data = image_zoom(self.data, zoom=[zoom_factor] * self.dimensions + [1], order=1)

        # Adjust the expected patterns to the PMT's quantum efficiencies, if desired
        # No need to re-normalize: will be done in each gof computation anyway
        if adjust_to_qe is not None:
            self.data *= adjust_to_qe[[np.newaxis] * self.dimensions]

        # Store index starts and distances for quick access, assuming uniform grid spacing
        self.coordinate_data = []
        for dim_i, (name, (start, stop, n_points)) in enumerate(json_data['coordinate_system']):
            n_points *= zoom_factor
            if not n_points == self.data.shape[dim_i]:
                raise ValueError("Map interpretation error: %d points expected along %s, but map is %d points long" % (
                    n_points, name, self.data.shape[dim_i]))
            self.coordinate_data.append(CoordinateData(minimum=start,
                                                       maximum=stop,
                                                       n_points=n_points,
                                                       point_spacing=(stop - start)/(n_points - 1)))
        self.log.debug('Coordinate ranges: %s' % ', '.join(['%s-%s (%d points)' % (cd.minimum, cd.maximum, cd.n_points)
                                                            for cd in self.coordinate_data]))

        # TODO: Technically we should zero the points outside the tpc bounds again:
        # some LCE may have leaked into this region due to upsampling... but doesn't matter:
        # if it causes a bias, it will push some events who are already far outside the fiducial volume
        # even further out.
        self.n_points = self.data.shape[-1]
        self.default_pmt_selection = np.ones(self.n_points, dtype=np.bool)
        if default_errors is None:
            default_errors = 0
        self.default_errors = default_errors

    def expected_pattern(self, coordinates):
        """Returns expected, normalized pattern at coordinates
        'Pattern' means: expected fraction of light seen in each PMT, among PMTs included in the map.
        Keep in mind you'll have to re-normalize if there are any dead / saturated PMTs...
        """
        # Copy is to ensure the map is not modified accidentally... happened once, never again.
        pattern = self.data[self.coordinates_to_indices(coordinates) + [slice(None)]].copy()
        sum_pattern = pattern.sum()
        if sum_pattern == 0:
            raise CoordinateOutOfRangeException("Expected light pattern at coordinates %s "
                                                "consists of only zeros!" % str(coordinates))
        return pattern / sum_pattern

    def compute_gof(self, coordinates, areas_observed,
                    pmt_selection=None, square_syst_errors=None, statistic='chi2gamma'):
        """Compute goodness of fit at a single coordinate point
        :param areas_observed: arraylike of length n_points containing observed area at each point
        :param coordinates: arraylike of n_dimensions, coordinates to test
        :param pmt_selection: boolean array of length n_points, if False point will be excluded from statistic
        :param square_syst_errors: float array of length n_points, systematic error to use for each point
        :param statistic: 'chi2' or 'chi2gamma': goodness of fit statistic to use
        :return: value of goodness of fit statistic, or float('inf') if coordinates outside of range
        """
        return self._compute_gof_base(self.coordinates_to_indices(coordinates), areas_observed,
                                      pmt_selection, square_syst_errors, statistic)

    def compute_gof_grid(self, center_coordinates, grid_size, areas_observed,
                         pmt_selection=None, square_syst_errors=None, statistic='chi2gamma', plot=False):
        """Compute goodness of fit on a grid of points of length grid_size in each coordinate,
        centered at center_coordinates. All other parameters like compute_gof.
        Returns gof_grid, (index of lowest grid point in dimension 1, ...)
        :return:
        """
        index_selection = []
        lowest_indices = []
        for dimension_i, x in enumerate(center_coordinates):
            cd = self.coordinate_data[dimension_i]
            start = self._coordinate_to_index(max(x - grid_size / 2, cd.minimum),
                                        dimension_i)
            lowest_indices.append(start)
            stop = self._coordinate_to_index(min(x + grid_size / 2, cd.maximum),
                                       dimension_i)
            index_selection.append(slice(start, stop + 1))        # Don't forget python's silly indexing here...

        gofs = self._compute_gof_base(index_selection, areas_observed, pmt_selection, square_syst_errors, statistic)

        # The below code is for diagnostic plots only
        if plot:
            plt.figure()
            plt.set_cmap('viridis')
            # Make the linspaces of coordinates along each dimension
            # Remember the grid indices are
            q = []
            for dimension_i, cd in enumerate(self.coordinate_data):
                dimstart = self._index_to_coordinate(index_selection[dimension_i].start, dimension_i)
                dimstart -= 0.5 * cd.point_spacing
                # stop -1 for python silly indexing again...
                dimstop = self._index_to_coordinate(index_selection[dimension_i].stop - 1, dimension_i)
                dimstop += 0.5 * cd.point_spacing
                q.append(np.linspace(dimstart, dimstop, gofs.shape[dimension_i] + 1))

                if dimension_i == 0:
                    plt.xlim((dimstart, dimstop))
                else:
                    plt.ylim((dimstart, dimstop))

            if statistic == 'likelihood_poisson':
                # because ln(a/b) = ln(a) - ln(b), also different ranges
                q.append(gofs.T - np.nanmin(gofs))
                plt.pcolormesh(*q, vmin=1, vmax=100, alpha=0.9)
                plt.colorbar(label=r'$L - L_0$')
            else:
                q.append(gofs.T / np.nanmin(gofs))
                plt.pcolormesh(*q, vmin=1, vmax=4, alpha=0.9)
                plt.colorbar(label='Goodness-of-fit / minimum')
            plt.xlabel('x [cm]')
            plt.ylabel('y [cm]')

        return gofs, lowest_indices

    def coordinates_to_indices(self, coordinates):
        return [self._coordinate_to_index(x, dimension_i) for dimension_i, x in enumerate(coordinates)]

    def _coordinate_to_index(self, value, dimension_i):
        """Return array index along dimension_i which contains value.
        Raises CoordinateOutOfRangeException if value out of range.
        TODO: check if this is faster than just using np.digitize on the index list
        """
        cd = self.coordinate_data[dimension_i]
        if not cd.minimum - cd.point_spacing / 2 <= value <= cd.maximum + cd.point_spacing / 2:
            raise CoordinateOutOfRangeException("%s is not in allowed range %s-%s" % (value, cd.minimum, cd.maximum))
        value = max(cd.minimum, min(value, cd.maximum - 0.01 * cd.point_spacing))
        return int((value - cd.minimum) / cd.point_spacing + 0.5)

    def _index_to_coordinate(self, index_i, dimension_i):
        cd = self.coordinate_data[dimension_i]
        return cd.minimum + cd.point_spacing * index_i

    def _compute_gof_base(self, index_selection, areas_observed, pmt_selection, square_syst_errors, statistic):
        """Compute goodness of fit statistic: see compute_gof
        index_selection will be used to slice the spatial histogram.
        :return: gof with shape determined by index_selection.
        """
        if pmt_selection is None:
            pmt_selection = self.default_pmt_selection
        if square_syst_errors is None:
            square_syst_errors = (self.default_errors * areas_observed) ** 2

        # The following aliases are used in the numexprs below
        areas_observed = areas_observed.copy()[pmt_selection]
        q = self.data[index_selection + [pmt_selection]]
        qsum = q.sum(axis=-1)[..., np.newaxis]          # noqa
        fractions_expected = ne.evaluate("q / qsum")    # noqa
        total_observed = areas_observed.sum()           # noqa
        ao = areas_observed                             # noqa
        square_syst_errors = square_syst_errors[pmt_selection]    # noqa

        # The actual goodness of fit computation is here...
        # Areas expected = fractions_expected * sum(areas_observed)
        if statistic == 'chi2gamma':
            result = ne.evaluate("(ao + where(ao > 1, 1, ao) - {ae})**2 /"
                                 "({ae} + square_syst_errors + 1)".format(ae='fractions_expected * total_observed'))
        elif statistic == 'chi2':
            result = ne.evaluate("(ao - {ae})**2 /"
                                 "({ae} + square_syst_errors)".format(ae='fractions_expected * total_observed'))
        elif statistic == 'likelihood_poisson':
            # Simple Poisson likelihood
            # Clip areas to range [0.0001, +inf), because of log(0)
            areas_expected_clip = np.clip(fractions_expected * total_observed, 0.0001, float('inf'))
            # Actually compute -2ln(L) so the same interval computation can be used later
            result = ne.evaluate("-2*(ao * log({ae}) - {ae})".format(ae='areas_expected_clip'))
        else:
            raise ValueError('Pattern goodness of fit statistic %s not implemented!' % statistic)

        return np.sum(result, axis=-1)

    def minimize_gof_grid(self, center_coordinates, grid_size, areas_observed,
                          pmt_selection=None, square_syst_errors=None, statistic='chi2gamma', plot=False, cls=None):
        """Return (spatial position which minimizes goodness of fit parameter, gof at that position,
        errors on that position) minimum is found by minimizing over a grid centered at
        center_coordinates and extending by grid_size in all dimensions.
        Errors are optionally calculated by tracing contours at given confidence levels, from the
        resulting set of points the distances to the minimum are calculated for each dimension and
        the mean of these distances is reported as (dx, dy).
        All other parameters like compute_gof
        """
        gofs, lowest_indices = self.compute_gof_grid(center_coordinates, grid_size, areas_observed,
                                                  pmt_selection, square_syst_errors, statistic, plot)
        min_index = np.unravel_index(np.nanargmin(gofs), gofs.shape)
        # Convert index back to position
        result = []
        for dimension_i, i_of_minimum in enumerate(min_index):
            x = self._index_to_coordinate(lowest_indices[dimension_i] + i_of_minimum, dimension_i)
            result.append(x)

        # Compute confidence level contours (but only in 2D)
        n_dim = len(min_index)
        # Store contours for plotting only
        cl_segments = []
        # Store (dx, dy) for each CL for output
        confidence_tuples = []

        if cls is not None and n_dim == 2:
            x, y = np.mgrid[:gofs.shape[0], :gofs.shape[1]]
            # Use matplotlib _Cntr module to trace contours (without plotting)
            c = _cntr.Cntr(x, y, gofs)

            for cl in cls:
                ct = ConfidenceTuple()
                ct.level = cl
                # Trace at the required value
                cl_trace = c.trace(gofs[min_index] + cl)
                # Check for failure
                if len(cl_trace) == 0:
                    confidence_tuples.append(ct)
                    continue

                # Get the actual contour, the first half of cl_trace is an array of (x, y) pairs
                half_length = int(len(cl_trace)//2)
                cl_segment = np.array(cl_trace[:half_length][0])

                # Extract the x values and y values seperately, also convert to the TPC coordinate system
                x_values = np.array([self._index_to_coordinate(lowest_indices[0] + x, 0) for x in cl_segment[:,0]])
                y_values = np.array([self._index_to_coordinate(lowest_indices[1] + y, 1) for y in cl_segment[:,1]])
                if np.all(np.isnan(x_values)) or np.all(np.isnan(y_values)):
                    self.log.debug("Cannot compute confidence contour: all x or y values are Nan!")
                    # If we'd now call nanmin, we get an annoying numpy runtime warning.
                else:
                    # Calculate the confidence tuple for this CL
                    ct.x0 = np.nanmin(x_values)
                    ct.y0 = np.nanmin(y_values)
                    ct.dx = abs(np.nanmax(x_values) - np.nanmin(x_values))
                    ct.dy = abs(np.nanmax(y_values) - np.nanmin(y_values))

                # Does the contour touch the edge of the TPC
                if np.isnan(x_values).any() or np.isnan(y_values).any():
                    ct.at_edge = True

                confidence_tuples.append(ct)

                # The contour points, only for plotting
                if plot:
                    contour_points = np.array([x_values, y_values]).T
                    # Take out point if x or y is nan
                    contour_points = [p for p in contour_points if not np.isnan(p).any()]
                    cl_segments.append(contour_points)

        if plot and n_dim == 2:
            plt.scatter(*[[r] for r in result], marker='*', s=20, color='orange', label='Grid minimum')
            for i, contour in enumerate(cl_segments):
                if len(contour) == 0:
                    continue
                color = lambda x: 'w' if x % 2 == 0 else 'r'
                p = plt.Polygon(contour, fill=False, color=color(i), label=str(cls[i]))
                plt.gca().add_artist(p)
            # plt.savefig("plot_%.2f_%.2f.pdf" % (result[0], result[1]), dpi=150)

        return result, gofs[min_index], confidence_tuples

    def minimize_gof_powell(self, start_coordinates, areas_observed,
                            pmt_selection=None, square_syst_errors=None, statistic='chi2gamma'):
        direc = None
        if self.dimensions == 2:
            # Hack to match old chi2gamma results
            s = lambda d: 1 if d < 0 else -1  # flake8: noqa
            direc = np.array([[s(start_coordinates[0]), 0],
                              [0, s(start_coordinates[1])]])

        def safe_compute_gof(*args, **kwargs):
            try:
                return self.compute_gof(*args, **kwargs)
            except CoordinateOutOfRangeException:
                return float('inf')

        # Minimize chi_square_gamma function, fmin_powell is the call to the SciPy minimizer
        # It takes the function to minimize, starting position and several options
        # It returns the optimal values for the position (xopt) and function value (fopt)
        # A warnflag tells if the maximum number of iterations was exceeded
        #    warnflag 0, OK
        #    warnflag 1, maximum functions evaluations exceeded
        #    warnflag 2, maximum iterations exceeded
        rv = fmin_powell(safe_compute_gof,
                         start_coordinates, direc=direc,
                         args=(areas_observed, pmt_selection, square_syst_errors, statistic),
                         xtol=0.0001, ftol=0.0001,
                         maxiter=10, maxfun=None,
                         full_output=1, disp=0, retall=0)
        xopt, fopt, direc, iter, funcalls, warnflag = rv
        # On failure the minimizer seems to give np.array([float('inf')])
        if isinstance(fopt, np.ndarray):
            fopt = float('nan')
        return xopt, fopt

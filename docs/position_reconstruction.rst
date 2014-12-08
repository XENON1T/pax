=======================
Position Reconstruction
=======================

Intro
#####

Position reconstruction (in the x,y plane) in pax is done via transformation plugins. Such a plugin will take an event and for each S2 peak reconstruct the original position of the interaction. Each peak will then be appended with a `ReconstructedPosition` object so other plugins in the processing chain can use this information.

Pax has the following position reconstruction plugins:

* Charge-weighted sum reconstruction
* Chi-square-gamma minimization reconstruction 

Charge-weighted Sum
###################

For each S2 peak this algorithm calculates the charge-weighted sum and appends this information to the peak in the form of a `ReconstructedPosition` object.

Usage and Settings
------------------

In the configuration file, in the list of transformation plugins, include: ::

  'PosSimple.PosRecWeightedSum'

Options:

* pmts_to_use_for_reconstruction

pmts_to_use_for_reconstruction can either be `top` or `bottom`


Chi-square-gamma Minimization
#############################

This position reconstruction is based on the assumption that the number of counts in the PMTs of the top array are generally low and we are thus dealing with poisson distributed data. Therefore this data will not in general be described by a chi-square distribution since it underestimates the true mean. Using a modified chi-square statistic for low counts, called the chi-square-gamma distribution it is possible to reconstruct the true mean.

This plugin uses a simulated LCE map and an S2 hit pattern to construct a chi-square-gamma distribution function in the x,y plane. Then finding the global minimum of this function yields the reconstructed position. The function value at this position is a goodness-of-fit parameter "chi-square-gamma". The plugin also calculates the number of degrees of freedom (ndf).

To minimize the function the plugin uses a SciPy minimizer using the Powell method, the options of this minimizer are currently optimized for accuracy, not for speed. The starting position for the minimizer is the position of the top PMT which sees the most signal, if a weighted sum is already calculated this position is taken as start value.

Since the chi-square-gamma function value at the reconstructed position is a goodness-of-fit parameter it can also be used to check how well other positions are reconstructed by calculating their chi-square-gamma value. Therefore the plugin checks if a peak already contains a `ReconstructedPosition` object created with another position reconstruction algorithm and appends the goodness-of-fit and ndf to these.

Usage and Settings
------------------

In the configuration file, in the list of transformation plugins, include: ::

  'PosRecChiSquareGamma.PosRecChiSquareGamma'

There are currenly two options that need to be set in order to use the plugin. `posrecChi.ini` provides a "minimal working example".

Options are:

* mode
* lce_map_file_name

mode can be either `full`, `no_reconstruct` or `only_reconstruct`:

* `no_reconstruct` will only append a new goodness-of-fit and ndf parameter to existing `ReconstructedPosition` objects created with other algorithms.
* `only_reconstruct` will leave existing `ReconstructedPosition` objects alone, do position reconstruction and append a new `ReconstructedPosition` object.
* `full` will do both

lce_map_file is the name of the LCE-map in the data directory.

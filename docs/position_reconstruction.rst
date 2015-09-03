=======================
Position Reconstruction
=======================

Intro
#####

The position reconstruction of S2's in the (x,y) plane is implemented in transformation plugins. Such a plugin will take an event and for each S2 peak reconstruct the original position of the interaction. Each peak will then be appended with a `ReconstructedPosition` object so other plugins in the processing chain can use this information.

PAX has the following position reconstruction methods:

* Maximum PMT: The position of the PMT that sees the most photons of the S2 yields the reconstructed position.
* Charge-weighted mean: The reconstructed position is computed as the charge-weighted mean of PMTs hit.
* Robust weighted mean: Iterative weighted mean. Improves charge-weighted mean by removing outliers in the hitpattern.
* Neural Network (XENON100 only): The same neural network as used in xerawdp.
* Chi-square-gamma reconstruction (XENON100 tested, XENON1T ready): The position is determined by comparing the hitpattern to expected values from a simulated LCE map. A test statistic is computed and yields the reconstructed position when minimized. This method also provides a per-event error on the reconstructed position (confidence contours) and is physically motivated.

Charge-weighted Sum
###################

For each S2 peak this algorithm calculates the charge-weighted sum and appends this information to the peak in the form of a `ReconstructedPosition` object.

Usage and Settings
------------------

In the configuration file, in the list of transformation plugins, include: ::

  'PosSimple.PosRecWeightedSum'

Options:

* channels_to_use_for_reconstruction

channels_to_use_for_reconstruction can either be `top` or `bottom`


Chi-square-gamma Minimization
#############################

This plugin uses a simulated LCE map and an S2 hit pattern to construct a chi-square-gamma distribution function in the (x,y) plane. Then finding the global minimum of this function yields the reconstructed position. The function value at this position is a goodness-of-fit parameter "chi-square-gamma". The plugin also calculates the number of degrees of freedom (ndf).

Since the chi-square-gamma function value at the reconstructed position is a goodness-of-fit parameter it can also be used to check how well other positions are reconstructed by calculating their chi-square-gamma value. Therefore the plugin checks if a peak already contains a `ReconstructedPosition` object created with another position reconstruction algorithm and appends the goodness-of-fit and ndf to these.

This method also provides a way to calculate confidence contours around each reconstructed position. This feature is still to be implemented.

The plugin currently works only in a XENON100 configuration since it depends on a simulated LCE map. When a per-PMT S2 LCE map is available for XENON1T, this method can also be used for XENON1T.

Usage and Settings
------------------

The chi-square-gamma reconstruction is run by default in the XENON100.ini

To include the plugin in a custom .ini file, include: ::

  'PosRecChiSquareGamma.PosRecChiSquareGamma'

There are currently several options that can be set or changed, these are:

* mode (default: `full`)
* seed_from_neural_net (default: `True`)
* ignore_saturated_PMTs (default: `True`)

`mode` can be either `full`, `no_reconstruct` or `only_reconstruct`:

* `no_reconstruct` will only append a new goodness-of-fit and ndf parameter to existing `ReconstructedPosition` objects created with other algorithms.
* `only_reconstruct` will only do position reconstruction and append a new `ReconstructedPosition` object.
* `full` will do both.

`seed_from_neural_net` will use the position as reconstructed by the neural network as seed for the minimizer, if it is available.

`ignore_saturated_PMTs` will remove saturated PMTs from the reconstruction completely. This is different from the case that the PMT sees nothing!


Working principle
-----------------

The chi-square-gamma method is based on the calculation of a test statistic given a hit pattern and simulated LCE map. The minimum function value will yield the reconstructed position and the function value itself is a goodness-of-fit parameter called chi-square-gamma.

In short a chi-square statistic will allow to compare data to model values, a lower chi-square meaning a better fit. In our case we compare an S2 hit pattern (number of photons seen by each top PMT) to the expected pattern if the event originated from position (x,y), finding the (x,y) which gives the best agreement yields the reconstructed position. However, an S2 hit pattern has to be modelled with Poisson statistics since there are generally a low number of counts in a PMT. In this case a chi-square statistic will not accurately model the data. Fortunately a modified chi-square statistic exists for low counts, this is the modified chi-square distribution called chi-square-gamma as described by Mighell(1999).

As noted before, the method uses an LCE map, this simulated map gives the probability for a PMT in the top array to receive a photon out of the total photons detected in the top array from position (x,y). One complication is that the LCE map is not continuous but simulated on a grid. To provide the reconstruction algorithm with a continuous LCE, interpolation is used.

To minimize the function the plugin uses a SciPy minimizer using the Powell method. The starting position (or seed) of the minimizer is the position of the top PMT which sees the most signal, if a neural net position is already calculated this position is taken as start value.

Performance
-----------

To examine the performance of the position reconstruction plugins the PosrecTest IPython notebook in pax/examples can be used. Based on monte carlo hitpatterns sampled from the LCE map the chi-square-gamma method performs best overall.

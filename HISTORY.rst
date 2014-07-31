.. :changelog:

History
-------

0.2.0 (2014-07-13)
---------------------

* Define static event class data structure
* Transforms now specified in ini file
* Can launch small web server for viewing plots
* Matching to XeRawDP (Jelle elaborate?)
* Transforms have start and stop methods

0.1.0 (2014-07-18)
---------------------

* First release of software framework
* Functional but not complete digital signal processing
  * Sum waveform for top, bottom, veto
  * Filtering with raised cosine filter
  * Peak finding of S1 and S2
* Basic inputs
  * MongoDB (used online for DAQ)
  * XED (XENON100 format)
* Basic outputs
  * ROOT
  * Pickle
  * Plots
* Demo reconstruction algorithm of charge-weighted sum

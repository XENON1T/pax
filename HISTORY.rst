.. :changelog:

History
-------

------------------
0.2.1 (2014-08-14)
------------------

* paxit binaries installed by default to allow working out of source (Commit 03470f7, Issues #25) 

------------------
0.2.0 (2014-08-04)
------------------

* Define static event class data structure
* Transforms now specified in ini file
* Can launch small web server for viewing plots
* Major changes to essentially every part of the peak finding code to better match Xerawdp. Agreement is currently at the 95% level.
 * Two important bugfixes for determining included channels : XED channel mask parsing, 0->1 start
 * Filter impulse response now identical now to Xerawdp
 * Different summed waveforms for s1 and s2 peakfinding
* Transforms have start and stop methods

------------------
0.1.0 (2014-07-18)
------------------

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

===============================
Processor for Analyzing XENON1T
===============================

PAX is used for doing digital signal processing and other data processing on the XENON1T raw data

* Free software: BSD license
* Documentation: http://pax.readthedocs.org.

Features
--------

* Digital signal processing
 * Sum waveform for top, bottom, veto
 * Filtering with raised cosine filter
 * Peak finding of S1 and S2
* I/O
 * MongoDB (used online for DAQ)
 * XED (XENON100 format)
 * ROOT
 * Pickle
 * Plots
* Position reconstruction of events
 * Demo reconstruction algorithm of charge-weighted sum


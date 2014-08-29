.. :changelog:

History
-------

------------------
1.1.0 (2014-08-29)
------------------

* HDF5 output - will be, for now, default output format.

  * We now have a binary output format for peaks and event.
  * Should also be easily extendible to ROOT output, which is blocked until ROOT solves some Py3.4 bugs.
  * Allows bulk comparison with high statistics for things like trigger efficiency

* Bug fixes relating to difference between XENON100 and XENON1T formats (occurences extending past event windows).
* Starting work on a new SimpleDSP processor
* Waveform generator
* General bug fixes and cleanup

------------------
1.0.0 (2014-08-16)
------------------

* Completely refactored event datastructure

 * Moved from Python dictionaries to an event class, seen in pax.datastructure
 * Ported all modules with pax to the new structure
 * Should open up I/O and C++ binding opportunities
 * Now there are Event, Peak, Waveform, and ReconstructedPosition classes
 * All of this is based on an extensively modified fork of 'micromodels'.

* Input control (See Issue #26)

 * Can now run pax with single events
 * Run paxit --help to see how one can process events

* Binaries of paxit installed when pax is installed
* Improved testing

 * Started testing plugins (this will start including other plugins later in the release)
 * Extensively testing the event class

* Peak finder now nearly identical to Xerawdp: better than 99.9% agreement on >20000 peaks tested

 * Simulation of the Xerawdp convolution bug (filtered waveform mutilation around pulse edges)
 * Small bugfixes (empty isolation test regions, strange behaviour when max of filtered waveform is negative)
 * Xerawdp XML file interpretation is off-by one (min_width=10 means: width must be 11 or higher)

* Integration of a waveform simulator (FaX) which can simulate S1s, S2s, and white noise

 * Script to convert from MC/NEST root files to FaX instructions
 * Simplified but much faster simulation mode used for peaks >1000 pe

* Plotting improvement: largest S1 & S2 in separate subplot
* Numerous bug fixes:

 * Pickler I/O
 * Remove dead code (clustering)




------------------
0.2.1 (2014-08-14)
------------------

* paxit binaries installed by default to allow working out of source

------------------
0.2.0 (2014-08-04)
------------------

* Define static event class data structure
* Transforms now specified in ini file
* Can launch small web server for viewing plots
* Major changes to the peak finding to better match Xerawdp. Agreement is currently at the 95% level.

 * Two important bugfixes for determining included channels : XED channel mask parsing, 0->1 start
 * Filter impulse response now identical to Xerawdp
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

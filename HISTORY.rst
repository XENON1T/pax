.. :changelog:

History
-------

------------------
3.1.0 (2015-06-07)
------------------

* Simplified hit finder (#167)
* ZLE in waveform simulator
* BSON output
* Cleanup of Travis building
* Various bug fixes

------------------
3.0.0 (2015-04-20)
------------------

* Reprocessing capability, switch HDF5 backend (#116)
* Better clustering algorithms: MeanShift, GapSize (#124)
* Hitfinder: faster, new noise definition, work on raw ADC data (#126)
* Bad channel rejection -> suspicious channel testing (#126)
* ROOT output, including tests for Travis (#127)
* Speed and feature improvements to folder-based IO (XED, Avro, ...) (#131)
* Datastructure update (#139)
* Bugfixes, plotting and comment improvements

------------------
2.1.0 (2015-02-15)
------------------

 * Avro raw data output
 * Neural net reconstruction
 * And lots of meaningless commits to get Travis continuous integration and Coveralls code coverage to work!  (And ROOT, which will be in 2.2)

------------------
2.0.1 (2015-01-26)
------------------

 * Travis CI continuous integration is enabled
 * Minor bug fixes

   * Values missing from output if they were always default
   * Memory leak after many events due to logger


------------------
2.0.0 (2015-01-15)
------------------

 * Changes to core
 
   * Cleanup of datastructure (see #80 and #81)
   * Most of core wrapped in Processor class
   * Plugins shut down at end of Processor.run(), not just on destruction

 * New signal processing chain

   * BaselineExcursionMethod, finds single-photon peaks in each channel (Default)
   * FindBigPeaks, a traditional sum-waveform peakfinder
   * Supporting peak classification and property computation plugins

 * Chi-square gamma x,y position reconstruction
 * Waveform simulator enhancements

   * Wrapped in Simulator class, loaded along with processor
   * Performance improvements
   * Basic zero-length encoding emulation 
   
 * WritePandas: write our data as DataFrames to containers supported by pandas
 * 2D channel waveforms plot
 * Support for arbitrary external detectors / extra channel groups
 * More tests

 
------------------
1.4.0 (2014-11-21)
------------------

 * DSP
   
   * Peak width fields added to datastructure
   * newDSP: Interpolated peak width computations
   * DSP plugins cleaned up and reorganized (except old peak finder) 
   * Frequency bandpass filtering support
 
 * Updated docs, comments, logging
 * Plots
   
   * 3D channel waveforms plot
   * Event summary plot
 
 * Music output (fun side project)
 * Separate directory for example data files
 * Configurations for XAMS, Bern test setup
 * --input and --output override settings for most plugins
 * WaveformSimulator: improved defaults
 * Stable DAQ injector
 * Various bug fixes and cleanups that polish


------------------
1.3.0 (2014-10-17)
------------------

* Plugin directory moved (fix bug in previous release)
* Bulk processing enhancements

  * Scripts for parallelization
  * XED: read in entire datasets, not just single files

* More command line arguments: input, plotting
* Configuration enhancements

  * Module-level settings
  * Multiple inheritance

* DAQInjector

  * New run-database format
  * Repeat single events
  * Create shard index
  * Further debugging and maturing
  
* Implement run database interface
* WaveformSimulator (Fax) cleanup:

  * Several truth file & instruction file formats
  * Better-motivated settings
  * ER/NR S1s
  
* Cut overhanging pulses
* Several PosSimple improvements 
* Interpolating detector maps (for position-dependent signal corrections)
* Plot 2D hit patterns


------------------
1.2.0 (2014-10-02)
------------------

* DAQ injector - can inject data into DAQs.
* Nested configurations - better handling of configurations and allows for nesting


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

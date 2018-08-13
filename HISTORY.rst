.. :changelog:

History
-------

------------------
6.10.1 (2018-08-13)
------------------
* Add per PMT contribution to tight coincidence level (#717)
* Push git-tag before actual commit (#714)
* clean map name and RunDB example (#715, #716)

------------------
6.10.0 (2018-07-19)
------------------
* Clustering and desaturation upgrade for high-E (#712)
* Enabling lone-hit simulation in FAX (#708)
* notebooks to update RunDB and fix NN file version issue (#713)
* Some minor improvement: Lower tight coincidence to 2-fold at pax level (#705), speed improvement for map interpolation (#676, #672), update S1 pulse shape parameter (#673, #665), show reconstructed position in WF plot (#677)
* FIX related to Maps/inputs: 4th FDC map (#670), PMT after pulse (#671),maps renaming (#654), add missing tensorflow files (#679), new NN files after SR1 (#712)
* Code style and dependencies: Issue template (#707), solfware dependencies (#701, #683), increase deletion timeout (#667)


------------------
6.9.0 (2018-01-18)
------------------
* Clustering change: Tuning gap size parameters, using index_of_maximum instead of left_central (#642), configuration of tight coincidence (#644)
* Maps changes: Alpha map using time depending 3d FDC for SR1 (#643), per-PMT S2 LCE map update (#646), True LCE map (#658), Update time-dependent 3D FDC maps for SR1 (#662), Add SR0 and update SR1 corrected LCE maps (#661), Update S2 xy map (#664)
* Tensor Flow NN: 3D FDC for TFNN in SR0 (#656), Time-dependent 3D FDC for Tensor Flow Neural Net (#655), tensorflow based nn of SR0 (#651), tensorflow based nn (#650), New TF Json with bad PMT List (#659)
* Bug FIX: Don't include inactive PMTs in S1 pattern fit (#663), Bugfix for AFT calculation (#657),  * Enhencement: Refactor S1 AFT probability calculation (#660), Add SR-dependent configs from processing (#652), More reasonable electron lifetime default (#647), Add sr1 pmt after pulse configs from beginning/middle/end of the runs (#645)

------------------
6.8.3 (2017-09-26)
------------------
* saving per channel contribution to tight coincidence (#641)

------------------
6.8.2 (2017-09-26)
------------------
* New SR0 Net for SR0 reprocessing (#612)
* Tuning clustering for SR1 (#615) 
* New maps for SR1 and SR0 (#625, #626 #628, #631)
* Time dependent 3-D FDC (#632)
* FAX update: single electron shape (#616), New PMT after pulse model for SR1 (#635), Update single-PE and S1 waveform model (#639)
* No FDC in pax (#620)
* Delete old maps (#627, #621, #623, #624)
* Bug fixes (#618, #638, #633)
* Code cleaning and style changes (#614, #619, #630, #634)

------------------
6.8.1 (2017-09-26)
------------------
* New 3D data-driven FDC map (#611)
* Fix de-saturation bug for very rare event (#601)
* SR1 PMT gains set to default in fax config (#606)
* S1 optical photon hit patterns with improved binning (#603)
* New S1 area fraction top map (#598)
* Prepare for supernova trigger (#568)
* Don't keep LED data in buffer (#600)
* Add random initial pickle to acquisition monitor file (#608)


------------------
6.8.0 (2017-08-07)
------------------
* Two step gap clustering (#596) 
* Add data-driven SPE distributions for each channel (fax) (#595)
* InterpolatingMap bug fix (#582)

------------------
6.6.6 (2017-06-28)
------------------
* add S2 correction in task list (#573)
* S1 area fraction top plugin (#572)
* Update package versions in environment creation (#578)
* New per-PMT LCE maps for full LXe+GXe volume (#577, #588)
* Fax update for better photo-ionisation model and memory leak fix (#574, #581, #589)
* Classification update (#576, #584)
* Disabling pulse-based desaturation for S1 (#580)
* New Position reconstruction (#591, #592)

------------------
6.6.5 (2017-05-04)
------------------
* S2 map update for SR1 with new gains  (#570)
* S1 LCE map update for SR1 with new gains  (#571)
* Re-enable reduction of processed data (#569)
* Coherent noise and maximum S2 afterpulse delay (#562)

------------------
6.6.4 (2017-04-24)
------------------
* New Neural network training results with PMT 2 off in SR1 (#560)
* Update XENON1T_MV.ini file for gain calibration (#561)
* Field Distortion Correction and LCE maps for SR1 (#563)
* New S2 correction map for SR1 (#565)
* Add threshold on saturation correction reference region to avoid biased correction for S1 (#566)
* Fix event builder error related to UTC timestamp (#567)

------------------
6.6.3 (2017-03-30)
------------------
* Do not compute S1 AFT for large S1 (#558)
* Don't delete area<0 hits in fixed range hitfinder mode (#557)
  
------------------
6.6.2 (2017-03-15)
------------------
* Fix peak.hits_fraction_top calculation (#555)
    * Bug found in S1AreaFractionTopProbability calculation (#553)

------------------
6.6.1 (2017-03-11)
------------------
* Updated field correction and S1 LCE map (#551)
* Fix typo in S1 Area fraction top computation (#550)

------------------
6.6.0 (2017-03-08)
------------------
* Updates to data files:
    * S2(x,y) relative light yield map (#535)
    * Neural net (#531)
* S1 area fraction top probability computed (#533)
* Fix bug in peak.area_midpoint computation
* Simulator: Use realistic SPE spectrum in fax (#528)
* Simulator: Fix bug in self-trigger emulation, AmBe noise file (#537)

------------------
6.5.1 (2017-02-28)
------------------
* Minor bugfixes (#526, #527, #529)

------------------
6.5.0 (2017-02-21)
------------------
* Change S1/S2 pairing loop order (#522)
* Minor speed improvements (#520)
* Add S1 hit-based pattern fit variable (#525)
* Simulator: extend afterpules to just after the initial signal (#521)
* Simulator: effective S1 shape model (#524)
* Travis builds fixed (#523)

------------------
6.4.2 (2017-02-19)
------------------
* Update author list
* Fix drift velocity fetched in the runs database (no change in pax)

------------------
6.4.1 (2017-02-18)
------------------
* Fix classification minus sign error
* Fix waveform simulator internal bug

------------------
6.4.0 (2017-02-17)
------------------
* New classification based on new properties. Order S1s by coincidence. (#510)
* Cluster based on central regions of hits (#504)
* Simulator: aft truth info, update for S2, afterpulse gains, fix dead PMT double counting (#516)
* Simulator: Update SE gain, double PE emission (#507)
* Update (r,z) correction map (#512)
* Update S2(x,y) correction map, top and bottom correction map (#515)
* Fix saturated PMT accounting config typo (#513)

------------------
6.3.3 (2017-02-14)
------------------
* Fix for rare edge cases (#499)
* ROOT-based event display (#503)
* Fax update: afterpulses, gains, S2 AFT (#502)
* Add run number to plot_dir filename (#500)

------------------
6.3.2 (2017-02-13)
------------------
* Fix (r,z) correction bug (#498)
* Neural network update (#495)

------------------
6.3.1 (2017-02-09)
------------------
* Fix crash due to edge case due to negative samples in hits (#490)
* Remove AWS support (#489)

------------------
6.3.0 (2017-02-08)
------------------
* Hitfinder upgrade to accomodate long-tailed pe-pulses (#479)
* r-z position correction from field simulation (#482)
* Pulse-based desaturation correction (#480, #484)
* New S1(x,y,z) correction map (#488) and QE rescaling fix
* S2(x,y) correction (#487)
* Realistic photoelectron pulse model (#478)
* Peak property cleanup (#485)
* Pickle output for plotting (#481)
* Plugin to apply software prescale retroactively (#486)

------------------
6.2.1 (2017-01-11)
------------------
* Fax: first realistic noise and ZLE thresholds (#471)
* Fax: first realistic PMT after-pulse configuration (#475)

------------------
6.2.0 (2016-12-20)
------------------
* Low-energy classification update, new peak properties (#467)
* Corrected error in peak.mean_area_over_noise computation
* Fix timezone offset for event time in plots (#466)
* Pattern goodness of fit is now a likelihood ratio (#465)
* Event builder robustness to db connectivity issues (#461)
* Fixed-range hit finder (#458)

------------------
6.1.1 (2016-11-24)
------------------
* Fax update - PMT and S2 afterpulses (#452)
* Include factorial in posrec likelihood (#459)

------------------
6.1.0 (2016-11-11)
------------------
* Improved clustering (#450)
* Make processed data smaller (#456)
* Write fax output to its own ROOT file (#453)
* Neural network update (#457)
* Fix for acquisition monitor pulse rescue logic (#454)
* Bug related to decimal types in datastructure (#447)
* Assorted event builder improvements after major changes

------------------
6.0.2 (2016-10-31)
------------------
* More lenient timeouts for batch queue processing

------------------
6.0.1 (2016-10-31)
------------------
* Fixed memory leak
    
------------------
6.0.0 (2016-10-26)
------------------
* Remote/distributed multiprocessing (#439)
* New S1 overall LCE maps (#436) and S2 per-pmt LCE maps (#431)
* Amazon Dynamo support (#438)
* Saturation correction computed, not added to total for XENON1T (#437)
* Matched waveform simulator S2 model (#428)
* Event display improvements (#440, #427)
* Event builder speed improvements (#434)
* Fax naming changes (#443)


------------------
5.7.0 (2016-10-07)
------------------
* Eventbuilder improvements
 * Trigger base logic improvements (#407, 418)
 * do_not_trigger option (#419)
 * Fix for online dead time calculation (#416)
 * Trigger monitor improvements (#412, #424, #429)
 * Save acquisition monitor pulses (#414)
* pulses_per_channel and n_pulses attributes added to Event class (#425, #422)
* Fax saves GEANT4 id to ROOT file, more noise options (#426)

------------------
5.6.5 (2016-09-21)
------------------

* Screwed up release, fixed HISTORY.rst

------------------
5.6.4 (2016-09-21)
------------------

* Mistaken release

------------------
5.6.3 (2016-09-21)
------------------

* MV processing support further

------------------
5.6.2 (2016-09-01)
------------------

* MV processing support

------------------
5.6.1 (2016-09-01)
------------------

* Muon veto processing support using run database information

------------------
5.6.0 (2016-08-31)
------------------

* Deal correctly with gains almost zero but see hit (#415)
* LCE map from Kr83m (#413)
* Raw data from MV had suffix (#411)

------------------
5.5.1 (2016-08-15)
------------------

* Fixes for trigger (#409)
* Adjust drift velocity (#410)


------------------
5.5.0 (2016-08-08)
------------------

* Muon veto integration (#406)

------------------
5.4.0 (2016-08-06)
------------------

* New raw data format
* Reconstruction fix (#401)
* Event builder changes (#404)

------------------
5.3.6 (2016-07-25)
------------------

* Small DAQ fixes.


------------------
5.3.5 (2016-07-25)
------------------

* Trying to get a DOI

------------------
5.3.4 (2016-07-25)
------------------

* Small tweak to description in setup.py (which I actually made because I want to get a DOI for pax).


------------------
5.3.3 (2016-07-25)
------------------

* Small DAQ fixes.

------------------
5.3.2 (2016-07-20)
------------------

* Only TPC errors can stop DAQ.

------------------
5.3.1 (2016-07-19)
------------------

* Fix MV bug in ini (#397).

------------------
5.3.0 (2016-07-19)
------------------

* Trigger changes for deadtime and acquisition monitor - no backword compatbility for new data (#395)
* Muon veto DAQ changes (#396)

------------------
5.2.1 (2016-07-06)
------------------

* Fix release

------------------
5.2.0 (2016-07-06)
------------------

* Store acquisition monitor info (#392)

------------------
5.1.0 (2016-06-27)
------------------

* S1 relative light yield map update (#382)
* Makefile now writes to stable branch (#379)
* Geant4 interface simulates from below cathode if desired (#389)
* Trigger changes with error handling (#386)
* Trigger prevent invalid event ranges (#388)
* Equalized gains (#387)

------------------
5.0.2 (2016-06-20)
------------------

* Unintentional release.

------------------
5.0.1 (2016-06-20)
------------------

* Minor DAQ fixes including error handling (#384)

------------------
5.0.0 (2016-06-08)
------------------

* Ready for XENON1T data
* New clustering (#372)
* Extended trigger window (#369)

------------------
4.11.0 (2016-06-06)
------------------

* Run database interface (#366)
* Revive PMTs that were masked (#371)

------------------
4.10.2 (2016-06-01)
------------------

* ROOT fix (#370)
* Configurable low-level info (#368)
* Event builder fixes (scattered commits)

------------------
4.10.1 (2016-05-30)
------------------

* Add PMT information (#364)
* Event builder changes (#365 plus other commits), including processing related changes.

------------------
4.10.0 (2016-05-20)
------------------

* Initial pax tuning for XENON1T #361

------------------
4.9.3 (2016-05-12)
------------------

* Temporarily downgrade scipy due to issues with latest build on some systems
* Event builder: split collections handling, save-all-pulses / mega event / timed trigger mode
* Lowered threshold in XENON1T-LED config until we can specify optimal threshold (#357)
* Waveform simulator bugfix (#354), LED signal simulation (#355)


------------------
4.9.2 (2016-05-03)
------------------

* Poisson likelihood statistic for position reconstruction, confidence contour improvement (#342)
* Event builder: parallel queries, delete-as-we-go, optimized queries, better config / run_doc handling
* Lock-based race condition prevention for ROOT class compilation (see #351)
* Fix wrong numbers in connector map (#349)

------------------
4.9.1 (2016-04-25)
------------------

* Neural net uses correct QEs
* Small changes for event builder
* Split S2 afterpulse models so independent for XENON100 and XENON1T

------------------
4.9.0 (2016-04-18)
------------------

* XENON1T: gains to 1 in LED mode, amplifiers and positions in pmts config dictionary (#339)
* XENON100 S2(x,y) map, XENON100 S2 simulation bugfix (#334)
* Event builder fixes, cax integration
* Pax version no longer append to output filename (0f26ac0)
* Multiprocessing and ROOT fix (#337)
* Waveform simulator afterpulses fix (#341)


------------------
4.8.0 (2016-03-29)
------------------

* New event builder version (#336)

------------------
4.7.0 (2016-03-21)
------------------

* Geant4 input to waveform simulator
* Tuning classification for XENON1T gas-mode zero-field.

------------------
4.6.1 (2016-03-07)
------------------

* Screwed up release, fixing...

------------------
4.6.0 (2016-03-07)
------------------

* Confidence levels on position reconstruction
* Saturation correction bug
* Several small bug fixes
* Minor event builder changes

------------------
4.5.0 (2016-02-26)
------------------

* .cpp classes now included within the ROOT output file (#323)
* Area corrections stored separately in datastructure (#322)
* Waveform simulator refactor, PMT afterpulses support (#321)
* Small event builder changes (#316, several loose commits)

------------------
4.4.1 (2016-02-05)
------------------

* Weird outlier bug fixes found in bulk processing

------------------
4.4.0 (2016-02-02)
------------------

* New event builder iteration (#297)
* Configuration bugs fixed

------------------
4.3.2 (2016-01-31)
------------------

* Small argument fixes for default configuration.

------------------
4.3.1 (2016-01-28)
------------------

* Nasty multiprocessing bug fix

------------------
4.3.0 (2016-01-25)
------------------

* Parallelization refactor (#298)
* Store meta data in ROOT output (#303)
* z coordinate system now negative in liquid (#302)
* Neural net reconstruction (#296)

------------------
4.2.0 (2016-01-11)
------------------

* Fixes for ROOT output: memory leak (#282), LED output (#283), long int fields (#289)
* Event builder changes (#278)
* 3D position reconstruction for S1s (#277)
* Hits and Pulses for S1s saved by default (#283)
* Raw data filename format changed, progress bar fix (#289)

------------------
4.1.2 (2015-11-30)
------------------

* Docs fixes
* TableWriter bug
* Saturation bug #274

------------------
4.1.0 (2015-11-17)
------------------

* ROOT class output
* Signal processing speedup (#245)
* S1 3d pattern simulation & goodness of fit computation (#237)
* Modifications for working with other TPCs (#247)
* Improvements to / fixes for noisy channel hit rejection
* Assorted bug fixes (#241, #244) and documentation fixes

------------------
4.0.1 (2015-10-17)
------------------

* Memory leak fixed
* Corrections to position reconstruction (#244)
* Documentation fixes

------------------
4.0.0 (2015-10-02)
------------------

* Add/remove several peak properties (#223, #214, #203), such as the peak's hits-only sum waveform.
* Clustering changes: separate plugins, better goodness of split, faster (#223, #213)
* Python 2 support (#217)
* Paxer options to switch input and output type (#212)
* Position reconstruction before classification (#223)
* Fast PatternFitter for position reconstruction (#233)
* Irregular correction map support, XENON100 S1(x,y,z) correction (#219)
* S1 vs S2 classification fix (#221)
* Several bugfixes and documentation improvements (e.g. #230)


------------------
3.3.0 (2015-08-03)
------------------

* Natural break declustering (#187)
* Improvements to chi2gamma accuracy and speed (#193, #196)
* Non-continuous events in ZippedBSON format (#192)
* XED writing (#177)
* Refactor plugin base and timing code (#190)
* S2 LCE in waveform simulator (#185)
* Cleanup plugin folders and names (#202)
* Minor improvements to logging (#155, #86) and plotting (#98, #144, #200)
* Documentation improvements


------------------
3.2.0 (2015-07-06)
------------------

* Multithreading of paxer (see --help)
* Clustering bug fixed (#186)
* Contribution section for non-XENON TPCs in examples.
* Chi2 algorithm now runs by default (and has energy cutoff for speed)
* Event builder pretrigger merged into pax
* Units now statically defined
* Various docs improvements


------------------
3.1.2 (2015-06-07)
------------------

* Update requirements.txt

  * Require new numba version since use new features
  * Pymongo3 required for all our Mongo setups

------------------
3.1.1 (2015-06-07)
------------------

* Fixed merge issue with minor release (mea culpa)

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

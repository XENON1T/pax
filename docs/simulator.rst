Waveform simulator docs – very early draft
==========================================

------------
What it does
------------
Simulate ADC waveforms resulting from S1 photons and/or S2 electrons released in a liquid xenon TPC and feed them into pax. 

It does not do yield estimation, i.e. convert energy depositions to number of photons and electrons -- use GEANT4/NEST/whatever for that.

-------------
How to use it
-------------

**Case 1: use all of the simulator to make waveforms.** For this, use pax with `–config XENON100 Simulation –input your_instructions.csv` (replace XENON100 with your TPC settings file if you’re simulating for another TPC). See below for the format of this instructions file.You’ll also want to specify an output format that is convenient for whatever you actually want to do with the waveforms: formats like pickle, JSON and BSON are easy to use (if you use python), but we also offer Avro, HDF5, CSV, HTML, ROOT, and soon XED. If you need a specific format not in this list, let us know, we may be bribable. Of course, if you use pax for plotting and analyzing the simulated waveforms, you may not need to output raw waveforms at all.

**Case 2: use just a small part of the simulator library**, such as the S1 photon time distribution, or the S2 per-PMT light distribution. In this case it is faster to make your own script with::

    from pax import core
    mypax = core.Processor(config_dict={'pax': {'plugin_group_names':[]}})
    sim = mypax.simulator
Now `sim` is an instance of the Simulator class, and you can call the methods from the 'library reference' section below. See examples/PosRecTest for an example of using the simulator like this.


-----------------------
Instruction file format
-----------------------
The instruction files read by the simulator are csv files with one row per simulated interaction (S1 AND S2, although you can turn of either -- or even both, but that's silly). 

The meaning of the columns is as follows:

 * **instruction**: All interactions with the same instruction number will be put into the same event. In principle these are simulated once, unless `event_repetitions` is set (to an integer larger than 1).
 * **recoil_type**: NR or ER. This will only affect the S1 pulse shape.
 * **x**: the x-position of the interaction in cm (in the coordinate system also used for the top PMT array), or the string `random`. In the latter case, the x & y position will be sampled randomly from a uniform distribution in a circle with radius `tpc_radius` (an option in the DEFAULT section of the configuration).
 * **y**: the y-position of the interaction in cm (in the coordinate system also used for the top PMT array). If x was set to `random`, whatever you put here is ignored. So you can put `random` here too, or your favourite band names, whatever. You cannot choose x randomly but not y, or vice versa. 
 * **depth**: the z-position of the interaction in cm below the gate mesh.
 * **s1_photons**: how many S1 photons to simulate. Whether these will all be detected by the PMTs will depend on the configuration settings, in particular `s1_detection_efficiency`.
 * **s2_electrons**: how many S2 electrons to simulate. Whether these will all arrive in the gas gap will depend on the configuration settings, in particular `electron_lifetime_liquid`.
 * **t**: Time of the interaction in nanoseconds since the start of the event. Keep in mind the S2 electrons take time to drift as well, so the S2 will appear (much) later in the event.
 
----------------------
Implementation details
----------------------

The waveform simulator consists of two parts:
 - An **input plugin** (WaveformSimulator) which lives in pax.plugins.io. This takes care of processing the instructions file, then calling the right commands for s1 and s2 generation, gluing the stages of the simulation process together, and finally outputting a file `fax_truth.csv` for your convenience which tells you what was actually simulated (i.e. where the each signal actually ended up in the event). This file is a csv file with one row per simulated peak (S1 or S2); the columns should speak for themselves.
 - The **Simulator class** which lives in pax.simulation. Every instance of the main Processor class has one simulator instance accessible by its `simulator` attribute. This does all the heavy lifting; here you’ll find all the physics code: random sampling distributions, diffusion model, etc.
 
Roughly, the life of a simulated waveform consists of several stages

1. **Instruction**. A line in the csv which says how many S2 electons and how many S1 photons to generate, at which place in the TPC and at which time relative to the start of an event.
2. **Photon production time lists** for each signal (S1 and S2): a list of times (in ns since the start of the event) when a photon is produced for a specific signal.  For S2 generation, there is an intermediary step: a list of times when an electron arrives in the gas. 
3. **Hitpattern(s)**: an object which knows which PMTs get hit by a photon at which time. Each signal first gets its own hitpattern object; the photon distribution takes into account (if possible) the location at which they are produced. Finally, all hitpatterns in an event are merged into a single hitpattern object.
4. A **pax event** containing the actual simulated ADC waveforms. This is passed to the next plugin in pax (either a processing plugin or straight to an output plugin), from whose point of view it is indistinguishable from real data. 

Electrical noise is added in the final step. ZLE emulation is not performed in the simulator, but by a separate plugin (SoftwareZLE) which you may or may not choose to run. [1]_


Photon production time simulation
---------------------------------
[FIXME] See https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:fax:presentation4june2014 for some outdated info. Also see the library reference below.


Hitpattern building
-------------------

Now that we know when the photons will be produced, we must figure out which PMTS see these photons. This depends on the location of the photon production, and can only be done properly if per-PMT position-dependent LCE maps are available. Information about the identity of the signal (S1 or S2) is no longer present at this stage; we get a list of photon production times from the previous stage, and must additionally pass the position at which these photons are produced.

For XENON100, LCE maps (generated from the GEANT4 Monte Carlo) are only available for S2 signals, and only for the top PMTs (which see just slightly more than 50% of the S2 light). These were extracted from Xerawdp, from a datafile used by the chi2 position reconstruction. This is a set of two-dimensional maps, since S2 is are always generated at the same z-position. [2]_

To trigger the use of these S2 LCE maps, the z-position of the anode (`-gate_anode_distance`) must be specified as the photon production location (the WaveformSimulator I/O plugin will take care of this behind the scenes), and the config option `s2_lce_map` must be set to the filename of the map file. This map file format is documented in InterpolatingMap; it must contain a 2d (x, y) map for each PMT number in the top array containing the PMTs relative LCE at that position (where relative is understood so the total for all PMTs in top array is 1 at each position, modulo small interpolation artefacts). The number of S2 photons that arrive at the top array is drawn from a binomial with p= `s2_mean_area_fraction_top`; the rest go to the bottom array, and are randomly distributed over all (see next paragraph for qualifying remark) PMTs there. [3]_ 

If any other z-position is specified, the x y and z positions you pass are ignored and the photons are distributed randomly and uniformly over all PMTs. Well, not necessarily all PMTs… if `pmt_0_is_fake` is set, channel 0 will never receive photons; if `magically_avoid_dead_pmts = True`, dead PMTs (gain=0) will also never receive photons, if `magically_avoid_s1_excluded_pmts = True` and `channels_excluded_for_s1` is present in the config (only for XerawdpImitation), the channels in channels_excluded_for_s1 also don’t receive any photons. While it is not the default, you often want to set magically_avoid_dead_pmts = True: this way the number of photons specified in the instruction file matches the photons observed. 

Internally, the hitpattern is stored as an object (`SimulatedHitpattern`).This holds the photon arrival times per channel in the dictionary attribute `s2_mean_area_fraction_top`. There are  also attributes for convenient access to the min and max hit time (needed in the waveform building stage to determine the event length) and the total n_photons.  On initializing the hitpattern, the photon times are corrected for the PMT transition time spread. In the future we might correct them for capture time as well, but this will require a set of 3d per-PMT histograms, or running an actual ray tracer for every photon (very slow). [FIXME, Cyril sent me per-array xyz arrival time maps a while ago].

In the WaveformSimulator IO plugin, the SimulatedHitpattern’s for several signals (S1 and S2 for every simulated interaction) are added (by overloaded + on the object) into a single SimulatedHitpattern object, which is passed to the next stage. 

Waveform building
-----------------
The final stage of the waveform simulator emulates the response of the PMT’ s and digitizers to the detected photons. Each PMT’s/channel's waveform is simulated separately, then the results are combined into a single pax event containing a full (XENON100: 400 us) waveform for every channel. In this final step, real electrical noise is added. ZLE emulation is not performed in the simulator, but by a separate plugin (SoftwareZLE).

1.	For each photon which arrives in the channel, the charge it deposits in digitizer bins close to its arrival time is calculated.

 a.	The transition time of the photon signal in the PMT is added to the arrival time. A Gaussian model is used with configurable mean (of no consequence) and sigma. The sigma can be deduced from PMT transition time spread measurements by dividing by 2.35 (the FWHM of a Gaussian in units of its standard deviation). Gaussians are notorious for poorly modeling tails of distributions [ref Taleb for fun?], so this model is ill-suited for studies sensitive to outliers in transition time -- for example, low-energy S1 pulse shape studies in a detector which, like Xenon1T , use PMTs with transition time spread comparable to the liquid xenon excimer decay times.
 
 b.	A PMT gain value is sampled, separately for each photon, from a Gaussian truncated at zero, with mean and sigma as determined by the PMT gain calibration. The use of a Gaussian model again implies we ignore details of the tail of high photon-pulse area -- no known studies are particularly sensitive to this. By treating each photon separately and independently, we ignore PMT saturation effects. [not important: PMTs linear, have enough time to recover in long stretches of nothing between S1s/S2s, if single S2s get large the ADCs will saturate first?]
 
 c.	Using the photon arrival time, we calculate in which digitiser bin in the centre of the photon signal is to fall, as well as the offset of the signal centre in that bin. Since PMT pulse generation is the most performance-critical part, a key optimization is made: the offset in the bin is rounded to the nearest nanosecond (or a different, configurable precision), so only a fixed number of pulse shapes are produced in the next step, which can be cached.

 d.	The charge deposited in each bin is computed. We integrate a normalized model PMT pulse –  see figure …, two exponentials stitched together –between the boundaries of several digitizer bins close to the signal center. The rise and fall time of the exponentials are set to agree with photon-pulse shape measurements, described in […]. The charge deposited in bins further away from the center than a configurable number of rise / fall times from the center is ignored. Finally, the pulse is multiplied with the gain drawn in step [b].

 e.	The pulse is added to an initially empty waveform, in the right place.

2.	Extra white noise current can now be added to the waveform, drawing for each sample from a Gaussian with 0 mean configurable sigma. By default this feature is turned off.

3.	The waveform is converted to ADC counts deviation from baseline, using the load resistance,  digitizer voltage resolution, and external amplification.

4.	Real digitizer background output is superposed on the waveform. Small 150-sample data segments, taken from LED calibration events just before the LED starts firing, are randomly selected and concatenated. ((Noise data is BIG. Including 100 different 400us events would take a >1GB noise databank (ref Sander's dataset). You can do it, but don't come to me complaining about the speed you'll get. A more practical way is to take many small samples occasionally, then concatenate them in different combinations.)) This concatenation leads to more and more sudden baseline shifts than in real background output, but ensures the simulated background is sufficiently varied to make hitfinder tests robust. Each channel’s noise data is chosen separately, so any inter-channel noise correlations are currently not reproduced in the simulator! [FIXME] Alternatively, a fixed baseline can be added instead.

5.	The waveform is clipped to fall within values the digitizer can show, to model ADC saturation correctly.
This process is repeated for each channel -- even those that receive no photons, as they will still record noise. Finally, the waveforms for all channels are combined into a Pax event object, ready to be processed further.


------------------------------------
Waveform simulator library reference
------------------------------------

This is a reference for all methods in pax's waveform simulator library. To use the methods of the Simulator class, see "Case 2" at the start of this file.

.. automodule:: pax.simulation
    :members:
    :undoc-members:
    :show-inheritance:

-----------------------------------------
Waveform simulator input plugin reference
-----------------------------------------

This is a reference for the WaveformSimulator input plugin. You probably won't have to deal with this directly.

.. automodule:: pax.plugins.io.WaveformSimulator
    :members:
    :undoc-members:
    :show-inheritance:


.. [1] There is an option (called cheap_zle), off by default, to limit noise generation to just around detected photons. This will increases the speed of the simulation, at a loss of correctness, since noise can also trigger the ZLE by itself.

.. [2] Well, actually every S2 results from track of electron-Xenon interactions along the 2.5mm gas gap, but we usually neglect differences in light distribution from different parts of the track. This is not entirely accurate, in particular, because photons generated near the end of the track have a higher likelihood to be detected by the bottom PMT array, most likely because the wires of the anode are shadowing the top array at this point. See https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon100:analysis:single_e_waveform_model.

.. [3] This is an approximation, necessary due to the lack of S2 LCE maps for the bottom array. From what I've seen from S2s, this isn't a bad approximation, but I haven't tested it quantitatively.

"""
Settings file for Xenon100
"""
from __future__ import division
from pax.units import *

#Basics
tpc_name                  =           'Xenon100'
liquid_depth              =           30*cm     #elsewhere?

#Processor settings
baseline_sample_size      =           46                 #Take this # of samples from beginning of waveform to establish baseline: XeRawDP wiki note
peakfinder_treshold_type  =           'absolute'         #Absolute: tresholds in pe/ns. Relative: tresholds in sigmas from baseline
peak_treshold             =           20                 #For Xams: 7 sigmas instead
base_treshold             =           0.5                #For Xams: -0.5 sigmas instead
subbase_count_for_end     =           5                  #Peaks end when they come below base_treshold for the nth time (since last coming below peak_treshold)

#FaX performance settings
pulse_width_cutoff        =           3                  #Neglect pmt pulse before/after this many rise/fall times.
"""
Increasing this cutoff slows down the simulation by an approximately linear factor.
3 should be fine (see validation : area test)
Excessively high values are in fact bad, too: you don't know how the single-pe pulse looks at those long times
                                              very low voltages -> exacerbate effect of finite V-resolution
"""
default_split_time        =           10*ms              #By default, split waveform once an empty region of this size is encountered.

#Digitizer / PMT operating circuit
digitizer_baseline        =           16000              #?????
digitizer_V_resolution    =           2.25*V/2**(14)     #2.2V/2**14
digitizer_t_resolution    =           10*ns
digitizer_resistor        =           50*Ohm
digitizer_amplification   =           10                 #Amplification factor of any amplifiers placed AFTER the PMT
clipping                  =           True               #If False, negative digitizer readout is possible. Otherwise, clip large peaks to 0.

#PMT channel groupings
def range_inc(a,b): return list(range(a,b+1))
channel_groups           = {
    'top'           : range_inc(1,98),
    'bottom'        : range_inc(99,178),
    'top_veto'      : range_inc(179,210),
    'bottom_veto'   : range_inc(211,242),
    'bad'           : [1, 2, 145, 148, 157, 171, 177]
}
channel_groups['summed']      = channel_groups['top']      + channel_groups['bottom']
channel_groups['summed_veto'] = channel_groups['top_veto'] + channel_groups['bottom_veto']

#PMT positions and calibration data
#This should really be read from the PMT calibration database...
channel_data              =          {}
for i in range(256):
    channel_data[i] = {
        'gain': 2*10**6,
        'name': i,
        #XY position, other calibration data
    }

#TODO: Light collection map


##
## Simulator-only settings
##

#PMT pulse characteristics
pmt_transit_time_mean     =           50*ns              #PLACEHOLDER - PMT handbook upper limit for linear focussed pmt type
                                                         #upper limit chosen as transit time spread is way above limits quoted in this table
                                                         #Not a big issue I think, this merely shifts the entire waveform. Could even put it 0.
pmt_transit_time_spread   =           6*ns               #Lung et al. 2012, measured for Xenon1T PMTs at -1750V.. what is it for Xenon100 PMTs?
pmt_rise_time             =           2.6*ns             #Lung et al. 2012 ".. again, what is it for the Xenon100 PMTs?
pmt_fall_time             =           9.7*ns             #Lung et al. 2012 "
#pmt_gain                  =           2*10**6            #X100... GPlante's thesis
                                                         #X1T: 6.28*10**6 (Lung et al. 2012)

#S2 electron drift and extraction
electron_lifetime_liquid  =           350*us             #NSort has ms... GPlante's thesis has us.
drift_velocity_liquid     =           1.73*um/ns         #Andrea says 1.73 um/ns. Ethan's code has 1.8 mm/us. X100 single electron study has 0.27 cm/us... but at the same field as the DM data?
diffusion_constant_liquid =           12*cm**(2)/s       #Sorensen 2011, longitudinal diffusion. Ethan's code uses 70*cm**(2)/s! (0.007*mm**2/us)
electron_trapping_time    =           140*ns             #Nest 2014, but was obtained through fitting data
electron_extraction_yield =           1                  #"above 0.96" xenon:xenon100:analysis:maxime:s2afterpulses

#S2 electroluminescence
elr_length                =           2.5*mm             #Xenon100 Analysis paper, page 4, "h_g ~ 2.5 mm"
s2_secondary_sc_yield_density =       19.7/elr_length    #"secondary scintillation gain" per length unit. 19.7 from NSort. This automatically includes detection efficiencies.
drift_velocity_gas        =           8*mm/us            #Nest S2widthposter.pdf 'physically well motivated value' for Xenon10 (what is it for Xenon100?). They note 5 mm/us produced a better fit...
anode_wire_radius         =           125/2*um           #GPlante p 98
anode_mesh_pitch          =           2.5*mm             #GPlante p 98
wire_field_parameter      =           0.5                #field becomes wire-dominated (~1/r) at wire_field_parameter*anode_mesh_pitch
singlet_lifetime_gas      =           5.88*ns            #Nest 2014
triplet_lifetime_gas      =           100.1*ns           #Nest 2014
singlet_fraction_gas      =           0.697              #PLACEHOLDER... from Nest source code? what was it again?

#S1
s1_primary_eximer_fraction=           1/(1+1/(0.06))     #Nest 2011: Nex/Ni=0.06 "theoretically" ... wait, but I want Nex_primary/Nex_tot, not Nex/Ni!!!
s1_detection_efficiency   =           0.08               #% photons detected, NSort
singlet_lifetime_liquid   =           3.1*ns             #Nest 2014
triplet_lifetime_liquid   =           24*ns              #Nest 2014

#S1 defaults for interaction-dependent quantities
s1_default_eximer_fraction=           0.697              #Nest source code for gammas and alphas.... right?
s1_default_recombination_time =       5*ns               #PLACEHOLDER, \hat{\tau}=3.5ns, so reasonable?
s1_default_singlet_fraction=          0.697              #PLACEHOLDER.. ?

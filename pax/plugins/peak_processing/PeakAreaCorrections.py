import numpy as np
from pax import plugin, exceptions
from pax.dsputils import saturation_correction
import sys
from numpy import mean, sqrt, square, arange
import pandas as pd
from pax import configuration, dsputils


# Must be run before 'BuildInteractions.BasicInteractionProperties'


class S2SpatialCorrection(plugin.TransformPlugin):
    """Compute S2 spatial(x,y) area correction
    """

    def startup(self):
        if 'xy_posrec_preference' not in self.config:
            raise ValueError('Configuration for %s must contain xy_posrec_preference' % self.name)
        self.s2_light_yield_map = self.processor.simulator.s2_light_yield_map

    def transform_event(self, event):

        for peak in event.peaks:
            # check that there is a position
            if not len(peak.reconstructed_positions):
                continue
            else:
                try:
                    # Get x,y position from peak
                    xy = peak.get_position_from_preferred_algorithm(self.config['xy_posrec_preference'])

                    # S2 area correction: divide by relative light yield at the position
                    peak.s2_spatial_correction /= self.s2_light_yield_map.get_value_at(xy)
                    if 'map_top'in self.s2_light_yield_map.map_names:
                        peak.s2_top_spatial_correction /= self.s2_light_yield_map.get_value_at(xy,
                                                                                               map_name='map_top')
                        peak.s2_bottom_spatial_correction /= self.s2_light_yield_map.get_value_at(xy,
                                                                                                  map_name='map_bottom')
                except ValueError:
                    self.log.debug("Could not find any position from the chosen algorithms")
        return event


class S2SaturationCorrection(plugin.TransformPlugin):
    """Compute S2 saturation(x,y,pmtpattern) area correction
    """

    def startup(self):
        self.s2_patterns = self.processor.simulator.s2_patterns
        self.zombie_pmts_s2 = np.array(self.config.get('zombie_pmts_s2', []))

    def transform_event(self, event):

        for peak in event.peaks:
            # check that there is a position
            if not len(peak.reconstructed_positions):
                continue
            try:
                # Get x,y position from peak
                xy = peak.get_position_from_preferred_algorithm(self.config['xy_posrec_preference'])
            except ValueError:
                self.log.debug("Could not find any position from the chosen algorithms")
                continue
            try:
                peak.s2_saturation_correction *= saturation_correction(
                    peak=peak,
                    channels_in_pattern=self.config['channels_top'],
                    expected_pattern=self.s2_patterns.expected_pattern((xy.x, xy.y)),
                    confused_channels=np.union1d(peak.saturated_channels, self.zombie_pmts_s2),
                    log=self.log)
            except exceptions.CoordinateOutOfRangeException:
                self.log.debug("Expected light pattern at coordinates "
                               "(%f, %f) consists of only zeros!" % (xy.x, xy.y))

        return event



class S2SaturationCorrectionByWF(plugin.TransformPlugin):
    """
        Compute S2 saturation area correction based on WF model derived in data
    """
    def transform_event(self, event):
        # Getting the pulses to construct a WF
        pulses = event.pulses
        p_left=np.array([p.left for p in pulses])
        p_right=np.array([p.right for p in pulses])
        p_channel=np.array([p.channel for p in pulses])
        p_id=np.array([pid for pid in range(0,len(pulses))])
        pulses_pd=pd.DataFrame({"p_id":p_id,"p_left":p_left,"p_right":p_right,"p_channel":p_channel})
#        pulses_pd=pulses_pd[(pulses_pd["p_left"] <= end_sample) & (pulses_pd["p_right"] >= start_sample)]    
        print('total number of pulses:',len(pulses_pd))
        dt = event.sample_duration # ns per sample
        for peak in event.peaks :
            # only analize saturated peaks
            if((peak.n_saturated_channels>0) & (peak.area>10000.) & (peak.n_contributing_channels>=30)): 
                print('peak type:',peak.type)
                start_sample=peak.left
                end_sample=peak.right
                
                # defining waveforms
                nPmts=248
                nPmtsTop=127
                wf_raw=np.zeros((nPmts,end_sample-start_sample))
                channel_sat_status=np.zeros(nPmts) # set to 1 if channels has saturated channels
                channel_sat_samples=np.zeros(nPmts) # total number of saturated samples per channel
                channel_sat_left=np.zeros(nPmts) # total number of saturated samples per channel
                channel_sat_right=np.zeros(nPmts) # total number of saturated samples per channel

                # select pulses inside this peak window
                pulses_pd=pulses_pd[(pulses_pd["p_left"] <= end_sample) & (pulses_pd["p_right"] >= start_sample)]
                # Start constructing the WFs
                for pmtid in range(0, nPmts):
                    channel_sat_left[pmtid]=end_sample
                    channel_sat_right[pmtid]=start_sample
                    index_all=pulses_pd[(pulses_pd.p_channel==pmtid)].p_id.values # all pulses in this WF
                    adc_conversion=dsputils.adc_to_pe(self.config, channel=pmtid)
                    
                    for index in index_all:
                        p=pulses[int(index)]
                        p_left=p.left
                        p_right=p.right
                        raw_data_adc = p.raw_data.astype(np.float16)
                        for sample in range(start_sample,end_sample):
                            if(p_left<=sample and p_right>=sample):
                                adc=raw_data_adc[sample-p.left]
                                adc=16000.0-adc 
                                wf_raw[pmtid,sample-start_sample]=adc*adc_conversion
                            
                                # determine whether there is saturation
                                if (adc>=16000.0-0.5):
                                    channel_sat_status[pmtid]=1
                                    channel_sat_samples[pmtid]+=1
                                    channel_sat_left[pmtid]=np.minimum(sample, channel_sat_left[pmtid])
                                    channel_sat_right[pmtid]=np.maximum(sample, channel_sat_right[pmtid])
                                    
                # sum of WFs
                wf_sum_all=np.sum(wf_raw,axis=0) # all PMTs including the saturated channels
                # first deal with non-saturated PMTS, define Sum of nan_saturated WF 
                wf_sum_nan_saturate=np.zeros_like(wf_sum_all) # Sum of non-saturated 
                wf_sum_saturate=np.zeros_like(wf_sum_all) # Sum of non-saturated
                for sample in range(start_sample,end_sample):
                    for pmtid in range(0, nPmts):
                        # first deal with non-saturated PMTS
                        if(channel_sat_samples[pmtid]<1):
                            wf_sum_nan_saturate[sample-start_sample]+=wf_raw[pmtid,sample-start_sample]
                        else:
                            wf_sum_saturate[sample-start_sample]+=wf_raw[pmtid,sample-start_sample]
                area_nan_saturate=np.sum(wf_sum_nan_saturate) # Area of Total non-saturated WFs 
                # The apply correction on saturated samples, need the shape of non-saturated WFs
                # re-define pulse shape based on the derived model data
                max_sample=np.argmax(wf_sum_nan_saturate) #sample sample at maximum
                height=wf_sum_nan_saturate[max_sample]
                peak_threshold_left=0.01*height
                peak_threshold_right=0.02*height
                
                # search for more precise peak left/right edges
                peak_left_edge=max_sample
                for sample in reversed(range(start_sample,max_sample+start_sample)):
                    if(wf_sum_nan_saturate[sample-start_sample]<=peak_threshold_left):
                        peak_left_edge=sample
                        break
                if(peak_left_edge==max_sample): # use old edges if new edges now found
                    peak_left_edge=start_sample 
                    
                peak_right_edge=max_sample
                for sample in range(max_sample+start_sample,end_sample):
                    if(wf_sum_nan_saturate[sample-start_sample]<=peak_threshold_right):
                        peak_right_edge=sample
                        break
                if(peak_right_edge==max_sample): # use old edges if new edges now found
                    peak_right_edge=end_sample

                
                # re-calculate area in the new window
                area_nan_saturate=0.0
                peak_left_edge=np.maximum(peak_left_edge,start_sample)
                peak_right_edge=np.minimum(peak_right_edge,end_sample)
                for sample in range(peak_left_edge,peak_right_edge):
                    area_nan_saturate+=wf_sum_nan_saturate[sample-start_sample]
                
                # Area per pmt w/o saturation correction
                area_pmt_before_correction=np.sum(wf_raw,axis=1)
                area_pmt_after_correction=np.zeros(nPmts)
                area_pmt_correction_factor=np.zeros(nPmts)
                   
                areatop=0.0
                for pmtid in range(0,nPmts):
                    area_model_left,area_model_right=0.0,0.0
                    area_data_left,area_data_right=0.0,0.0
                    #for sample in range(start_sample,end_sample):
                    for sample in range(peak_left_edge,peak_right_edge):
                        if(sample<channel_sat_left[pmtid]):
                            area_model_left+=wf_sum_nan_saturate[sample-start_sample]
                            area_data_left+=wf_raw[pmtid,sample-start_sample]   
                                                        
                        elif(sample>channel_sat_right[pmtid]):
                            area_model_right+=wf_sum_nan_saturate[sample-start_sample]
                            area_data_right+=wf_raw[pmtid,sample-start_sample]
                            
                    area_data=area_data_left+area_data_right
                    area_model=area_model_left+area_model_right

                    # prefer to make corrections based on rising edge, this is possibly helpful for based saturation correction as well.                    
                    if (channel_sat_status[pmtid]>0) & (area_data_left>10) & (area_model>200) & (area_model_left>100) :
                        if((area_data_left>10) or (area_data_left/area_data>0.3) ):
                            area_pmt_correction_factor[pmtid]=area_data_left/area_model_left
                        # if rising edge is too small, use wfs after saturation
                        elif (area_data>10):
                            area_pmt_correction_factor[pmtid]=area_data/area_model
                        area_pmt_after_correction[pmtid]=area_nan_saturate*area_pmt_correction_factor[pmtid]
                    # for non saturated channels, update the calculation based on new peak edges
                    elif(peak_right_edge>peak_left_edge):
                        area_new=0.0
                        for sample in range(peak_left_edge,peak_right_edge):
                            area_new+=wf_raw[pmtid,sample-start_sample]
                        area_pmt_after_correction[pmtid]=area_new
                    
                    # re_compute the s2 area per channel, based on correction factor and the new edges
                    #area_pmt_after_correction[pmtid]=area_nan_saturate*area_pmt_correction_factor[pmtid]
                    peak.area_per_channel[pmtid]=area_pmt_after_correction[pmtid]
                   
                    # Re compute area fraction on top
                    if (pmtid<nPmtsTop):
                        areatop+=peak.area_per_channel[pmtid]
                       
                print('area before correction: ',peak.area)                
                peak.area=np.sum(peak.area_per_channel)
                print('area after correction: ',peak.area)
                
                #print('aft before correction: ',peak.area_fraction_top)
                if(peak.area>0):
                    peak.area_fraction_top=areatop/peak.area
                    #print('aft after correction: ',peak.area_fraction_top)
                
                    
                #print('width before correction: ',peak.range_area_decile)
                # recompute peak properties
                if(area_nan_saturate>1e3):
                    peak.area_midpoint, peak.range_area_decile = compute_area_deciles(wf_sum_nan_saturate)
                    peak.range_area_decile*=dt
                    #print('width after correction: ',peak.range_area_decile)
                
        return event
    
    
    
def compute_area_deciles(w):
    """Return (index of mid area, array of the 0th ... 10 th area decile ranges in samples) of w
    e.g. range_area_decile[5] = range of 50% area = distance (in samples)
    between point of 25% area and 75% area (with boundary samples added fractionally).
    First element (0) of array is always zero, last element (10) is the length of w in samples.
    """
    fractions_desired = np.linspace(0, 1, 21)
    index_of_area_fraction = np.ones(len(fractions_desired)) * float('nan')
    integrate_until_fraction(w, fractions_desired, index_of_area_fraction)
    return index_of_area_fraction[10], (index_of_area_fraction[10:] - index_of_area_fraction[10::-1]),


# @numba.jit(numba.void(numba.float32[:], numba.float64[:], numba.float64[:]),
#            nopython=True, cache=True)
# For some reason numba doesn't clean up its memory properly for this function... leave it in python for now
def integrate_until_fraction(w, fractions_desired, results):
    """For array of fractions_desired, integrate w until fraction of area is reached, place sample index in results
    Will add last sample needed fractionally.
    eg. if you want 25% and a sample takes you from 20% to 30%, 0.5 will be added.
    Assumes fractions_desired is sorted and all in [0, 1]!
    """
    area_tot = w.sum()
    fraction_seen = 0
    current_fraction_index = 0
    needed_fraction = fractions_desired[current_fraction_index]
    for i, x in enumerate(w):
        # How much of the area is in this sample?
        fraction_this_sample = x/area_tot
        # Will this take us over the fraction we seek?
        # Must be while, not if, since we can pass several fractions_desired in one sample
        while fraction_seen + fraction_this_sample >= needed_fraction:
            # Yes, so we need to add the next sample fractionally
            area_needed = area_tot * (needed_fraction - fraction_seen)
            results[current_fraction_index] = i + area_needed/x
            # Advance to the next fraction
            current_fraction_index += 1
            if current_fraction_index > len(fractions_desired) - 1:
                return
            needed_fraction = fractions_desired[current_fraction_index]
        # Add this sample's area to the area seen, advance to the next sample
        fraction_seen += fraction_this_sample
    if needed_fraction == 1:
        results[current_fraction_index] = len(w)
    else:
        # Sorry, can't add the last fraction to the error message: numba doesn't allow it
        raise RuntimeError("Fraction not reached in waveform? What the ...?")



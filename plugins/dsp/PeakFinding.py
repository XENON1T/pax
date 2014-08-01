import numpy as np

from pax import plugin, units

class FindPeaksXeRawDPStyle(plugin.TransformPlugin):

    def startup(self):
        self.log.debug(("If you compare the isolation test intervals with xerawdp,it looks like ours are too large.\n" +
                        "In reality, Xerawdp's math is off, and we are testing the same intervals."
        ))

    def transform_event(self, event):
        # Load settings here, so we are sure dynamic threshold stuff etc gets reset
        self.settings_for_peaks = [ #We need a specific order, so we can't use a dict
            ('large_s2', {
                'threshold':        0.6241506363,
                'left_boundary_to_height_ratio':    0.005,
                'right_boundary_to_height_ratio':   0.002,
                'min_length':       35,
                'max_length':       float('inf'),
                'min_base_interval_length': 60,
                'max_base_interval_length': float('inf'),
                'source_waveform':  'filtered_for_large_s2',
                # For isolation test on top-level interval
                'around_interval_to_height_ratio_max': 0.05,
                'test_around_interval': 21,
                # For isolation test on every peak
                'around_peak_to_height_ratio_max': 0.25,
                'test_around_peak': 21,
                'min_crossing_length':1,
                'stop_if_start_exceeded':False, # Not for large s2s? Bug?
            }),
            ('small_s2', {
                'threshold':                        0.06241506363,
                'left_boundary_to_height_ratio':    0.01,
                'right_boundary_to_height_ratio':   0.01,
                'min_base_interval_length':         40,
                'max_base_interval_length':          200,
                # Xerawdp bug: small_s2 filtered wv inadvertently only used for threshold crossing detection:
                # I hardcoded filtered_for_small_s2 there and changed the source_waveform here.
                # This is the solution with as little hardcoding as possible
                'source_waveform':                  'filtered_for_large_s2',
                'around_to_height_ratio_max':       0.05,
                'test_around':                      10,
                'min_crossing_length':              1,
                'stop_if_start_exceeded':           True,
            }),
            ('s1', {
                'threshold':                        0.1872451909,
                'left_boundary_to_height_ratio':    0.005,
                'right_boundary_to_height_ratio':   0.005,
                'min_crossing_length':              3,
                'min_base_interval_length':         0, #???!
                'max_base_interval_length':         float('inf'), #???!
                'stop_if_start_exceeded':           True,
                'source_waveform':                 'uncorrected_sum_waveform_for_s1',
            }),
            # TODO: Veto S1 peakfinding
        ]
        self.seeker_position = None
        self.highest_s2_height_ever = 0
        event['peaks'] = []
        for (peak_type, settings) in self.settings_for_peaks:
            # Get the signal out
            signal = event['processed_waveforms'][settings['source_waveform']]

            # Code for dumping waveforms:
            # if peak_type == 'large_s2':
            #     with open('s2_dump_pax.txt','w') as output:
            #         output.write("\n".join(map(str,signal)))
            # if peak_type == 'small_s2':
            #     with open('s2small_dump_pax.txt','w') as output:
            #         output.write("\n".join(map(str,signal)))
            # if peak_type == 's1':
            #     with open('s1_dump_pax.txt','w') as output:
            #         output.write("\n".join(map(str,signal)))
            #     exit()

            # Determine when we should stop looking for this type of peaks
            stop_looking_after = float('inf')
            if peak_type == 'small_s2':
                # For small s2s, we stop looking after a sufficiently large s2 (height in large_s2 waveform) is seen
                # Don't have to test these peaks are actually s2s, those are the only peaks in here
                huge_s2s = [p for p in event['peaks'] if p['height'] > 624.150636300]
                if huge_s2s:
                    stop_looking_after = min([p['left'] for p in huge_s2s])
            if peak_type == 's1':
                s2s = [p for p in event['peaks'] if p['peak_type'] in ('large_s2', 'small_s2')]
                if s2s:
                    # We stop looking for s1s after the s2 with the largest INTEGRAL
                    # Very confusing, and undocumented!
                    stop_looking_after = s2s[ int(np.argmax([p['integral'] for p in s2s])) ]['left']
                    #  Also stop looking after large enough s2s
                    #  Size of s2s is determined from s1 peakfinding waveform!
                    large_enough_s2s = [p for p in s2s if event['processed_waveforms']['uncorrected_sum_waveform_for_s1'][p['index_of_max_in_waveform']] > 3.1207531815]
                    if large_enough_s2s:
                        stop_looking_after = min(
                            stop_looking_after,
                            min([p['left'] for p in large_enough_s2s])
                        )

            self.log.debug("Starting %s search, stop looking after %s" % (peak_type, stop_looking_after))

            # Find peaks in all the free regions
            free_regions = self.get_free_regions(event) # Can't put this after 'for .. in', peaks get added during the loop!
            for region_left, region_right in free_regions:

                # S1s have a protection against starting too soon
                # (I guess S2s don't need this because of the filters applied to the waveform)
                if peak_type == 's1' and region_left < 12:
                    region_left = 12

                # Are we still interested?
                if region_left >= stop_looking_after:
                    self.log.debug("Stopping %s search, region starts at %s, but we were asked to stop at %s" %(
                        peak_type, region_left, stop_looking_after
                    ))
                    break

                # Need these in the find_peaks_in sub later
                self.current_region_left = region_left
                self.current_region_right = region_right
                self.left_extent_small_s2_search_limit = self.current_region_left
                # Reset of this is needed
                self.this_s1_alert_position = 0

                # Search for threshold crossings
                self.last_s1_left_boundary = region_left
                self.seeker_position = region_left
                while 1:
                    # Hack for xerawdp matching: small s2 has right signal here, for once
                    if peak_type == 'small_s2':
                        signal = event['processed_waveforms']['filtered_for_small_s2']

                    if self.seeker_position >= region_right:
                        self.log.debug("Seeker position ended up at %s, right region boundary is at %s!" % (
                            self.seeker_position, region_right,
                        ))
                        break

                    # Find the left_boundary of the peak candidate interval
                    # If we're looking for s1s, the seeker position can count as starting an s1
                    # For S2s, there is a check to see if we were above threshold already...
                    # We could implement this as:
                        # For S2s, we need to check if we are above threshold already.
                        # If so, move along until we're not
                        # (weird Xerawdp behaviour, but which of the two options is the most sensible?)
                        # while signal[self.seeker_position] > settings['threshold']:
                        #     self.seeker_position = find_next_crossing(signal, settings['threshold'],
                        #                                         start=self.seeker_position, stop=region_right)
                        #     if self.seeker_position == region_right:
                        #         self.log.warning('Entire %s search region from %s to %s is above threshold %s!' %(
                        #             peak_type, region_left, region_right, settings['threshold']
                        #         ))
                        #         break
                    # ... however, the most common case this happens is after a peak that fails tests
                    # in this case Xerawdp sets above-threshold to false, even though it really may be true...
                    if signal[self.seeker_position] > settings['threshold']:
                        left_boundary = self.seeker_position
                    else:
                        # We're not currently above threshold, so find the next threshold crossing, if it exists
                        left_boundary = find_next_crossing(signal, threshold=settings['threshold'],
                                                          start=self.seeker_position, stop=region_right)
                    if left_boundary == region_right:
                        break #No other crossing found, move on to the next region

                    if peak_type == 's1':
                        # Keep a record of the s1 alert positions: needed for isolation tests
                        self.this_s1_alert_position = left_boundary

                    # Determine the peak window: the interval in which peaks are actually searched for
                    # Also set the seeker position for the next peak search, if needed
                    if peak_type == 's1':
                        # Determine the maximum in the 'preliminary peak window' (next 60 samples) that follows
                        max_pos = int(np.argmax(signal[left_boundary:min(left_boundary + 60, region_right)]))
                        max_idx = left_boundary + max_pos
                        # Xerawdp keeps this, but we throw it away and recover it later... hopefully??

                        # Set a revised peak window based on the max position
                        # Should we also check for previous S1s? (Xerawdp does!) or prune overlap later?
                        #TODO: should this be done here? Or  is it clearer if it's done in find_peaks_in? Now I'm confused...
                        left_boundary = max(max_idx - 10 - 2, 10)    #NB overwrites! # 10 should be region_left, but isn't in Xerawdp...
                        right_boundary = max_idx + 60   # should be min(region_right, ""), but isn't in Xerawdp...
                        # The next seeker position will be set in find_peaks_in
                    else:
                        # Find where we drop below threshold again
                        right_boundary = find_next_crossing(signal, threshold=settings['threshold'],
                                                            start=left_boundary, stop=region_right)
                        max_idx = None #This will be determined later
                        # The next seeker position will be set in find_peaks_in, except for large_s2s....
                        if peak_type == 'large_s2':
                            self.seeker_position = right_boundary

                    # Did the peak actually end? (the tail of a big S2 sometimes doesn't)
                    if right_boundary == region_right:
                        #No, it didn't. For s1s, that means Xerawdp doesn't consider the peak...
                        if peak_type == 's2':
                            if right_boundary == event['len']-1:
                                 # But s2s at the end of the waveform will always end due to the Xerawdp convolution bug
                                pass
                            else:
                                self.log.debug("Peak starting at %s didn't end at region boundary %s" % (left_boundary, right_boundary))
                                break
                    # right_boundary -= 1 # Peak end is just before crossing # But Xerawdp doesn't do this either...

                    # Hack for xerawdp matching: small s2 has wrong signal from here on
                    if peak_type=='small_s2':
                        signal = event['processed_waveforms']['filtered_for_large_s2']

                    # Hand over to a function: this is needed because Xerawdp recurses for large s2s
                    self.find_peaks_in(event, peak_type, signal, settings, left_boundary, right_boundary,
                                       toplevel=True, max_idx=max_idx)

                    # For large s2, update the dynamic threshold
                    if peak_type=='large_s2':
                        settings['threshold'] = max(
                            settings['threshold'], 0.001*self.highest_s2_height_ever
                        )

        return event

    def find_peaks_in(self, event, peak_type, signal, settings, left_boundary, right_boundary, toplevel=False, max_idx=None):
        self.log.debug("%s candidate interval %s - %s" % (peak_type, left_boundary, right_boundary))

        # Check if the interval is large enough to contain the peak
        # We must start with this for large s2s.
        failed_interval_test = False
        if not settings['min_base_interval_length'] <= right_boundary - left_boundary <= settings['max_base_interval_length']:
            self.log.debug("    Interval failed width test %s <= %s <= %s" %(
                settings['min_base_interval_length'], right_boundary - left_boundary, settings['max_base_interval_length']
            ))
            if peak_type == 'large_s2':
                return
            else:
                # We can't return yet, we need the right extent of the never-to-be-accepted peak 
                # for the new self.seeker_position for small_s2 and s1...
                failed_interval_test = True

        # Find the maximum index and height
        if max_idx is None:
            max_idx = left_boundary + np.argmax(signal[left_boundary:right_boundary+1])  # Remember silly python indexing
        height = signal[max_idx]

        # How is the point to stop looking for the peak's extent related to the interval we're searching in?
        if peak_type=='large_s2':
            left_extent_search_limit  = nearest_s2_boundary(event, max_idx, left_boundary, 'left')
            right_extent_search_limit = nearest_s2_boundary(event, max_idx, right_boundary, 'right')
        elif peak_type=='small_s2':
            left_extent_search_limit  = self.left_extent_small_s2_search_limit
            right_extent_search_limit = self.current_region_right
        else:       # s1
            left_extent_search_limit = left_boundary
            right_extent_search_limit = right_boundary

        # Find the peak extent
        (left, right) = interval_until_threshold(
            signal,
            start=max_idx,
            left_threshold=settings['left_boundary_to_height_ratio'] * height,
            right_threshold=settings['right_boundary_to_height_ratio'] * height,
            left_limit=left_extent_search_limit,
            right_limit=right_extent_search_limit,
            stop_if_start_exceeded=settings['stop_if_start_exceeded'],
            min_crossing_length=settings['min_crossing_length'],
            activate_xerawdp_hacks_for=peak_type,
        )
        self.log.debug("    %s peak candidate: %s-%s-%s" % (peak_type, left, max_idx, right))

        # Except for large_s2 (which uses an interval recursion system), set the seeker position based on the right extent of the peak found
        if peak_type != 'large_s2':
            # Next search should start after this peak - do this before testing the peak or the arcane +2
            self.seeker_position = right + 1

        # And now, finally, we can (and should) return if we failed the interval test already above.
        if failed_interval_test: return

        # Store some quantities for large s2
        if peak_type == 'large_s2':
            # Update highest s2 height ever for dynamic threshold determination.
            # This has to get done AFTER the interval test, but before the peak has a chance to fail any other test
            self.highest_s2_height_ever = max(self.highest_s2_height_ever, height)
            # We need this for tests later, specifically, to see if child intervals contain peaks which can skip some of the large_s2
            if toplevel: self.last_toplevel_max_val = height

        # Check for pathological peaks; Xerawdp doesn't do this, I'm going to because it prevents crashes
        # Hmm, that probably means you're wrong about other things, right?
        if left >= right or left < 0 or right > len(signal)-1:
            self.log.warning("%s at %s-%s-%s has invalid extent -- not testing or appending it! Go debug the processor!" % (peak_type, left, max_idx, right))
            return

        # Apply various tests to the peaks
        if peak_type == 'large_s2':

            # For large S2 from top-level interval: do the isolation check on the top interval
            if toplevel:
                if not self.isolation_test(signal, max_idx,
                                      right_edge_of_left_test_region=min(left_boundary, left)-1,
                                      test_length_left=settings['test_around_interval'],
                                      left_edge_of_right_test_region=max(right_boundary, right)+1,
                                      test_length_right=settings['test_around_interval'],
                                      before_avg_max_ratio=settings['around_interval_to_height_ratio_max'],
                                      after_avg_max_ratio=settings['around_interval_to_height_ratio_max'],
                                      can_fail_one = True): #NB Different from all the others!!!! (also s1?)
                    self.log.debug("    Toplevel interval failed isolation test")
                    return

            # The peak FWHM is tested only for large s2s #TODO: HMMZ a duplicate fwhm computation... same in Xerawdp however
            fwhm = extent_until_threshold(signal, start=max_idx, threshold=height / 2)
            if not settings['min_length'] <= fwhm <= settings['max_length']:
                self.log.debug("    Failed width test")
                return

            # An additional, different, isolation test is applied to every individual peak < 0.05 the toplevel peak
            if height < 0.05 * self.last_toplevel_max_val:
                if not self.isolation_test(signal, max_idx,
                                      right_edge_of_left_test_region=left-1,
                                      test_length_left=settings['test_around_peak'],
                                      left_edge_of_right_test_region=right+1,
                                      test_length_right=settings['test_around_peak'],
                                      before_avg_max_ratio=settings['around_peak_to_height_ratio_max'],
                                      after_avg_max_ratio=settings['around_peak_to_height_ratio_max']):
                    self.log.debug("    Failed isolation test")
                    return


        elif peak_type == 'small_s2':

            # Test for aspect ratio of the UNFILTERED WAVEFORM, probably to avoid misidentifying s1s as small s2s
            unfiltered_signal = event['processed_waveforms']['uncorrected_sum_waveform_for_s2']
            height_for_aspect_ratio_test = unfiltered_signal[left + int(np.argmax(unfiltered_signal[left:right+1]))]
            aspect_ratio_threshold = 0.062451  # 1 mV/bin = 0.1 mV/ns
            peak_width = (right - left) / units.ns
            aspect_ratio = height_for_aspect_ratio_test / peak_width
            if aspect_ratio > aspect_ratio_threshold:
                self.log.debug('    Failed aspect ratio test: max/width ratio %s is higher than %s' % (aspect_ratio, aspect_ratio_threshold))
                return

            # For small s2's the isolation test is slightly different
            # This is insane, and probably a bug, but I swear it's in Xerawdp
            left_isolation_test_window  = min(settings['test_around'],right_boundary-left_extent_search_limit)
            right_isolation_test_window = min(settings['test_around'],right_extent_search_limit-right_boundary)
            self.log.debug("    Isolation test windows used: left %s, right %s" % (left_isolation_test_window, right_isolation_test_window))
            if not self.isolation_test(signal, max_idx,
                                  right_edge_of_left_test_region=left-1,
                                  left_edge_of_right_test_region=right+1,
                                  test_length_left=left_isolation_test_window,
                                  test_length_right=right_isolation_test_window,
                                  before_avg_max_ratio=settings['around_to_height_ratio_max'],
                                  after_avg_max_ratio=settings['around_to_height_ratio_max']):
                self.log.debug("    Failed the isolation test.")
                return


        elif peak_type == 's1':
            # Test for non-isolated peaks
            #TODO: dynamic window size if several s1s close together, check with Xerawdp now that free_regions
            if not self.isolation_test(signal, max_idx,
                                  right_edge_of_left_test_region=left-1,
                                  test_length_left=min(50,self.this_s1_alert_position-self.last_s1_left_boundary),
                                  left_edge_of_right_test_region=right+1,
                                  test_length_right=min(10,self.current_region_right-self.this_s1_alert_position),
                                  before_avg_max_ratio=0.01,
                                  after_avg_max_ratio=0.04):
                return

            # Test for nearby negative excursions #Xerawdp bug: no check if is actually negative..
            negex = min(signal[
                max(0,max_idx-50 +1) :              #Need +1 for python indexing, Xerawdp doesn't do 1-correction here
                min(len(signal)-1,max_idx + 10 +1)
            ])
            if not height > 3 * abs(negex):
                self.log.debug('    Failed negative excursion test')
                return

            #Test for too wide s1s
            filtered_wave = event['processed_waveforms']['filtered_for_large_s2']   #I know, but that's how Xerawdp...
            max_in_filtered = left + int(np.argmax(filtered_wave[left:right]))
            filtered_width = extent_until_threshold(filtered_wave,
                                                    start=max_in_filtered,
                                                    threshold=0.25*filtered_wave[max_in_filtered])
            if filtered_width > 50:
                self.log.debug('    Failed filtered width test')
                return

        # Update values for later isolation tests
        # This gets done only if the current peak passes all tests!
        # It has to get done AFTER the isolation test of the current peak of course.
        # TODO: maybe we can keep track of this in a more intelligent way?
        if peak_type == 'small_s2':
            # For small s2s, we need to store the right peak boundary as the left boundary for later isolation tests.
            self.left_extent_small_s2_search_limit = right
        elif peak_type == 's1':
            #For s1, update the last_s1_left boundary, also used for later isolation tests
            self.last_s1_left_boundary = self.this_s1_alert_position

        # If we're still here, append the peak found
        if peak_type == 's1':
            #Arcane Xerawdp stuff, probably to compensate for after crossing timer stuff?
            left -= 2
            right += 2
        self.log.debug("! Appending %s (%s-%s-%s) to peaks" % (peak_type, left, max_idx, right))
        event['peaks'].append( {
            'left':         left,
            'right':        right,
            'peak_type':    peak_type,
            'height':       height,
            'index_of_max_in_waveform': max_idx,
            'source_waveform':          settings['source_waveform'],
            'integral':     np.sum(signal[left:right]),     #Yeah, that's a waste of time! But it is really needed for s2s at least...
        })

        # For large s2s, recurse on the remaining intervals
        if peak_type == 'large_s2':
            # Recurse on remaining intervals right, then left
            # Ordering is different from Xerawdp because it uses a stack, we use recursion
            self.find_peaks_in(event, peak_type, signal, settings,
                               left_boundary=left_boundary, right_boundary=left) #Not left-1? Who is off by one?
            self.find_peaks_in(event, peak_type, signal, settings,
                               left_boundary=right, right_boundary=right_boundary) #Not right+1? Who is off by one?

    def isolation_test(
            self,
            signal, max_idx,
            right_edge_of_left_test_region, test_length_left,
            left_edge_of_right_test_region, test_length_right,
            before_avg_max_ratio,
            after_avg_max_ratio,
            can_fail_one=False,
        ):
        """
        Does XerawDP's arcane isolation test. Returns if test is passed or not.
        NB: edge = first index IN region to test!
        TODO: Regions seem to come out empty sometimes... when? warn?
        """
        # Xerawdp reports 1 tighter boundaries in the debug output than it actually tests.
        # Do the same so the log files match also.
        # self.log.debug("        Running isolation test: 'RofLeft' %s, 'LofRight' %s, LengthLeft %s, LengthRight %s" % (right_edge_of_left_test_region+1, left_edge_of_right_test_region-1, test_length_left, test_length_right))
        # Bug in Xerawdp: floor in waveform->average causes different behaviour for even&odd window sizes
        # Only case which is different is in pre-avg test (left test region), even window case
        right_edge_of_left_test_region -= (1 if test_length_right % 2 == 0 else 0)
        self.log.debug("        Running isolation test: RofLeft %s, LofRight %s, LengthLeft %s, LengthRight %s" % (right_edge_of_left_test_region, left_edge_of_right_test_region, test_length_left, test_length_right))
        # +1s are to compensate for python's indexing conventions...
        pre_avg = np.mean(signal[
            right_edge_of_left_test_region - (test_length_left-1):
            right_edge_of_left_test_region + 1
        ])
        post_avg = np.mean(signal[
            left_edge_of_right_test_region:
            left_edge_of_right_test_region + test_length_right
        ])
        height = signal[max_idx]
        if can_fail_one:
            failed = pre_avg > height * before_avg_max_ratio and post_avg > height * after_avg_max_ratio
        else:
            failed = pre_avg > height * before_avg_max_ratio or post_avg > height * after_avg_max_ratio
        if failed:
            self.log.debug("        Nope: %s > %s and/or %s > %s..." % (pre_avg, height * before_avg_max_ratio, post_avg, height * after_avg_max_ratio))
            return False
        else:
            self.log.debug("        Passed!")
            return True

    # TODO: Maybe this can get moves to the event class?
    def get_free_regions(self, event):
        lefts = sorted([0] + [p['left'] for p in event['peaks']])
        rights = sorted([p['right'] for p in event['peaks']] + [event['length']-1])
        free_regions = list(zip(*[iter(sorted(lefts + rights))]*2))   #hack from stackoverflow
        self.log.debug("Free regions: " + str(free_regions))
        return free_regions


class ComputePeakProperties(plugin.TransformPlugin):

    """Compute various derived quantities of each peak (full width half maximum, etc.)


    """

    def transform_event(self, event):
        """For every filtered waveform, find peaks
        """

        # Compute relevant peak quantities for each pmt's peak: height, FWHM, FWTM, area, ..
        # Todo: maybe clean up this data structure? This way it was good for csv..
        peaks = event['peaks']

        for i, p in enumerate(peaks):
            # Hack for Xerawdp matching: we need to compute the area of EVERY CHANNEL in EVERY PEAK
            # The only reason we do this is because channels with negative area don't get contribute to a peak's area...
            p['areas_per_pmt'] = {}
            for channel, wave_data in event['channel_waveforms'].items():
                #TODO: Don't hardcode this...!!!
                if channel > 178: continue
                if p['peak_type'] == 's1' and channel in self.config['pmts_excluded_for_s1']: continue
                integral = np.sum(wave_data[p['left']:p['right']]) # No +1, Xerawdp forgets the right edge also
                p['areas_per_pmt'][channel] = integral
            p['area_for_xerawdp_matching'] = sum([area for _, area in p['areas_per_pmt'].items() if area > 0])
            #Nicer computations, probably don't need them?
            #continue
            for channel, wave_data in event['processed_waveforms'].items():
                # Todo: use python's handy arcane naming/assignment convention to beautify this code
                peak_wave = wave_data[p['left']:p['right'] ]#+ 1] Xerawdp bug/feature: does not include right edge in integral...
                peaks[i][channel] = {}
                maxpos = peaks[i][channel]['position_of_max_in_peak'] = np.argmax(peak_wave)
                maxval = peaks[i][channel]['height'] = peak_wave[maxpos]
                peaks[i][channel]['position_of_max_in_waveform'] = p['left'] + maxpos
                peaks[i][channel]['area'] = np.sum(peak_wave)
                if channel == 'top_and_bottom':
                    # Expensive stuff...
                    # Have to search the actual whole waveform, not peak_wave:
                    # TODO: Can search for a VERY VERY LONG TIME when there are weird peaks, e.g in afterpulse tail..
                    # Fix by computing baseline for inividual peak, or limiting search region...
                    # how does XerawDP do this?
                    samples_to_ns = self.config['digitizer_t_resolution'] / units.ns
                    peaks[i][channel]['fwhm'] = extent_until_threshold(wave_data, start=p['index_of_max_in_waveform'],
                                                                       threshold=maxval / 2) * samples_to_ns
                    peaks[i][channel]['fwtm'] = extent_until_threshold(wave_data, start=p['index_of_max_in_waveform'],
                                                                       threshold=maxval / 10) * samples_to_ns
                    # if 'top' in peaks[i] and 'bottom' in peaks[i]:
                    # peaks[i]['asymmetry'] = (peaks[i]['top']['area'] - peaks[i]['bottom']['area']) / (
                    # peaks[i]['top']['area'] + peaks[i]['bottom']['area'])

        return event


# Helper functions for peakfinding



def find_next_crossing(signal, threshold,
                       start=0, direction='right', min_length=1,
                       stop=None, stop_if_start_exceeded=False, activate_xerawdp_hacks_for=None):
    """Returns first index in signal crossing threshold, searching from start in direction
    A 'crossing' is defined as a point where:
        start_sample < threshold < this_sample OR start_sample > threshold > this_sample

    Arguments:
    signal            --  List of signal samples to search in
    threshold         --  Threshold defining crossings
    start             --  Index to start search from: defaults to 0
    direction         --  Direction to search in: 'right' (default) or 'left'
    min_length        --  Crossing only counts if stays above/below threshold for min_length
                          Default: 1, i.e, a single sample on other side of threshold counts as a crossing
                          The first index where a crossing happens is still returned.
    stop_at           --  Stops search when this index is reached, THEN RETURNS THIS INDEX!!!
    #Hack for Xerawdp matching:
    stop_if_start_exceeded  -- If true and a value HIGHER than start is encountered, stop immediately
    activate_large_s2_hacks -- If true, also checks slope; if fails due to slope|boundary, return argmin instead

    This is a pretty crucial function for several DSP routines; as such, it does extensive checking for
    pathological cases. Please be very careful in making changes to this function, their effects could
    be felt in unexpected ways in many places.

    TODO: add lots of tests!!!
    TODO: Allow specification of where start_sample should be (below or above treshold),
          only use to throw error if it is not true? Or see start as starting crossing?
           -> If latter, can outsource to new function: find_next_crossing_above & below
              (can be two lines or so, just check start)
    TODO: Allow user to specify that finding a crossing is mandatory / return None when none found?

    """

    # Set stop to last index along given direction in signal
    if stop is None:
        stop = 0 if direction == 'left' else len(signal) - 1

    # Check for errors in arguments
    if not 0 <= stop <= len(signal) - 1:
        #TEMP HACK, this should become an error...
        print("!!!!!!!!!!!!! Invalid crossing search stop point: %s (signal has %s samples)" % (stop, len(signal)))
        return len(signal)-1
    if not 0 <= start <= len(signal) - 1:
        raise ValueError("Invalid crossing search start point: %s (signal has %s samples)" % (start, len(signal)))
    if direction not in ('left', 'right'):
        raise ValueError("Direction %s is not left or right" % direction)
    if not 1 <= min_length:
       raise ValueError("min_length must be at least 1, %s specified." %  min_length)
    if (direction == 'left' and start < stop) or (direction == 'right' and stop < start):
        # When Xerawdp matching is done, this should become a runtime error
        raise RuntimeError("Search region (start: %s, stop: %s, direction: %s) has negative length!" % (
            start, stop, direction
        ))
        #return stop #I hope this is what the Xerawdp algorithm does in this case...

    # Check for pathological cases which can arise, not serious enough to throw an exception
    # Can't raise a warning from here, as I don't have self.log...
    if stop == start:
        return stop
    if signal[start] == threshold:
        print("Threshold %s equals value in start position %s: crossing will never happen!" % (
            threshold, start
        ))
        return stop
    if not min_length <= abs(start - stop):
        # This is probably ok, can happen, remove warning later on
        print("Minimum crossing length %s will never happen in a region %s samples in size!" % (
            min_length, abs(start - stop)
        ))
        return stop

    # Do the search
    i = start
    after_crossing_timer = 0
    start_sample = signal[start]
    while 1:
        this_sample = signal[i]
        if i == stop:
            # stop_at reached, have to return something right now
            if activate_xerawdp_hacks_for == 'large_s2':
                # We need to index of the minimum before & including this point instead...
                if direction=='left':
                    return i + np.argmin(signal[i:start+1])
                else:    # direction=='right'
                    return start + np.argmin(signal[start:i+1])
            elif activate_xerawdp_hacks_for == 's1':
                # Xerawdp keeps going, but always increments after_crossing_timer, so we know what it'll give
                # This is a completely arcane hack due to several weird interactions of boundary errors
                if not this_sample < threshold:
                    #The counter gets reset on this sample
                    after_crossing_timer = 0
                else:
                    #This sample increments the counter
                    after_crossing_timer += 1
                return stop + (-1+after_crossing_timer if direction == 'left' else 1-after_crossing_timer)
            elif activate_xerawdp_hacks_for == 'small_s2':    #small_s2
                return stop + (-1 if direction=='left' else 1)
            else:
                return stop     # Sane case, doesn't happen in Xerawdp I think
        if stop_if_start_exceeded and this_sample > start_sample:
            #print("Emergency stop of search at %s: start value exceeded" % i)
            return i
        if start_sample < threshold < this_sample or start_sample > threshold > this_sample:
            # We're on the other side of the threshold that at the start!
            after_crossing_timer += 1
            if after_crossing_timer == min_length:
                return i + (min_length - 1 if direction == 'left' else 1 - min_length)
        else:
            # We're back to the old side of threshold again
            after_crossing_timer = 0

        # Dirty hack for Xerawdp matching
        if activate_xerawdp_hacks_for == 'large_s2':
            # Check also for slope inversions
            if this_sample > 7.801887059:   #'0.125 V'
                # We need to check for slope inversions. How bad is it allowed to be?
                if this_sample < 39.00943529: #'0.625 V'
                    log_slope_threshold = 0.02 #Xerawdp says '0.02 V/bin', but it is a log slope threshold...
                else:
                    log_slope_threshold = 0.005 #Idem '0.005 V/bin'

                #Todo: check if enough samples exist to compute slope..
                try:
                    # Calculate the slope at this point using XeRawDP's '9-tap derivative kernel'
                    log_slope = np.sum(signal[i-4:i+5] * np.array([-0.003059, -0.035187, -0.118739, -0.143928, 0.000000, 0.143928, 0.118739, 0.035187, 0.003059]))/this_sample
                    #print("Slope is %s, threshold is %s" % (slope, slope_threshold))
                    # Left slopes of peaks are positive, so a negative slope indicates inversion
                    # If slope inversions are seen, return index of the minimum before this.
                    if direction=='left' and log_slope < - log_slope_threshold:
                        print("    ! Inversion found on rising (left) slope at %s: log_slope %s < - %s. Started from %s" %  (i, log_slope, log_slope_threshold, start))
                        return i + np.argmin(signal[i:start+1])
                    elif direction=='right' and log_slope > log_slope_threshold:
                        print("    ! Inversion found on falling (right) slope at %s: log_slope %s > %s. started from %s" % (i, log_slope, log_slope_threshold, start))
                        return start + np.argmin(signal[start:i+1])
                except ValueError:
                    print(" !! Slope test crashed, you should check if you have enough samples... really.. ")
        # Increment the search position in the right direction
        i += -1 if direction == 'left' else 1


def interval_until_threshold(signal, start,
                             left_threshold, right_threshold=None, left_limit=0, right_limit=None,
                             min_crossing_length=1, stop_if_start_exceeded=False, activate_xerawdp_hacks_for=None
):
    """Returns (l,r) indices of largest interval including start on same side of threshold
    ... .bla... bla
    """
    if right_threshold is None:
        right_threshold = left_threshold
    if right_limit is None:
        right_limit = len(signal) - 1
    l_cross = find_next_crossing(signal, left_threshold,  start=start, stop=left_limit,
                           direction='left',  min_length=min_crossing_length,
                           stop_if_start_exceeded=stop_if_start_exceeded, activate_xerawdp_hacks_for=activate_xerawdp_hacks_for)
    r_cross = find_next_crossing(signal, right_threshold, start=start, stop=right_limit,
                           direction='right', min_length=min_crossing_length,
                           stop_if_start_exceeded=stop_if_start_exceeded, activate_xerawdp_hacks_for=activate_xerawdp_hacks_for)
    # BADNESS: this ins't interval_until_threshold, but 1 more on each side!
    return (l_cross, r_cross)


def extent_until_threshold(signal, start, threshold):
    a = interval_until_threshold(signal, start, threshold)
    return a[1] - a[0]


def nearest_s2_boundary(event, peak_position, edge_position, direction):
    boundaries = []
    if direction=='left':
        # Will take the max of 0, edge_position-100, and any s2 right boundaries before peak
        boundaries = [0, edge_position-100]
        for p in event['peaks']:
            if p['peak_type'] in ('small_s2', 'large_s2') and p['right'] <= peak_position:  # = case actually happens!
                boundaries.append(p['right'])
        return max(boundaries)
    elif direction=='right':
        # Will take the min of length-1, edge_position+100, and any s2 left boundaries after peak
        boundaries = [event['length']-1, edge_position+100]
        for p in event['peaks']:
            if p['peak_type'] in ('small_s2', 'large_s2') and peak_position <= p['left']:    # = case actually happens!
                boundaries.append(p['right'])
        return min(boundaries)
    raise RuntimeError("direction %s isn't left or right" % direction)
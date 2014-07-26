import numpy as np

from pax import plugin, units

class FindPeaksXeRawDPStyle(plugin.TransformPlugin):

    def startup(self):
        self.settings_for_peaks = [ #We need a specific order, so we can't use a dict
            ('large_s2', {
                'threshold':        0.624509324,
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
                'threshold':                        0.0624509324,
                'left_boundary_to_height_ratio':    0.01,
                'right_boundary_to_height_ratio':   0.01,
                'min_base_interval_length':         40,
                'max_base_interval_length':          200,
                'source_waveform':                  'filtered_for_small_s2',
                'around_to_height_ratio_max':       0.05,
                'test_around':                      10,
                'min_crossing_length':              1,
                'stop_if_start_exceeded':           True,
            }),
            ('s1', {
                'threshold':                        0.1872452894,
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

    def transform_event(self, event):
        event['peaks'] = []
        # Do everything first for large s2s, then for small s2s
        for (peak_type, settings) in self.settings_for_peaks:
            # Get the signal out
            signal = event['processed_waveforms'][settings['source_waveform']]

            # Determine when we should stop looking for this type of peaks
            stop_looking_after = float('inf')
            if peak_type == 'small_s2':
                # For small s2s, we stop looking after a sufficiently large s2 is seen
                # Don't have to test for p['peak_type'] == 'large_s2', these are the only peaks in here
                huge_s2s = [p for p in event['peaks'] if p['height'] > 624.151]
                if huge_s2s:
                    stop_looking_after = min([p['left'] for p in huge_s2s])
            if peak_type == 's1':
                # We stop looking for s1s after the largest s2, or any sufficiently large s2
                s2s = [p for p in event['peaks'] if p['peak_type'] in ('large_s2', 'small_s2')]
                if s2s:
                    stop_looking_after = s2s[ int(np.argmax([p['height'] for p in s2s])) ]['left']
                    large_enough_s2s = [p for p in s2s if p['height'] > 3.12255]
                    if large_enough_s2s:
                        stop_looking_after = min(
                            stop_looking_after,
                            min([p['left'] for p in large_enough_s2s])
                        )

            # Find peaks in all the free regions
            free_regions = get_free_regions(event) # Can't put this after 'for .. in', peaks get added during the loop!
            for region_left, region_right in free_regions:

                # Are we still interested?
                if region_left >= stop_looking_after: break

                # hacks ... hmzz
                event['region_right_end_for_small_peak_isolation_test'] = region_right
                event['left_boundary_for_small_peak_isolation_test'] = region_left

                # Search for treshold crossings
                self.seeker_position = region_left
                while 1:

                    # Are we perhaps above threshold already? if so, move along until we're not
                    while signal[self.seeker_position] > settings['threshold']:
                        self.seeker_position = find_next_crossing(signal, settings['threshold'],
                                                            start=self.seeker_position, stop=region_right)
                        if self.seeker_position == region_right:
                            self.log.warning('Entire %s search region from %s to %s is above threshold %s!' %(
                                peak_type, region_left, region_right, settings['threshold']
                            ))
                            break
                    if self.seeker_position == region_right: break
                    assert signal[self.seeker_position] < settings['threshold']

                    # Find the next threshold crossing, if it exists
                    left_boundary= find_next_crossing(signal, threshold=settings['threshold'],
                                                      start=self.seeker_position, stop=region_right)
                    if left_boundary == region_right:
                        break # There wasn't another crossing

                    # Determine the peak window: the interval in which peaks are actually searched for
                    if peak_type == 's1':
                        # Determine the maximum in the 'preliminary peak window' (next 60 samples) that follows
                        max_pos = int(np.argmax(signal[left_boundary:min(left_boundary + 60, region_right)]))
                        max_idx = left_boundary + max_pos
                        # Xerawdp keeps this, but we throw it away and recover it later... hopefully??

                        # Set a revised peak window based on the max position
                        # Should we also check for previous S1s? (Xerawdp does!) or prune overlap later?
                        left_boundary = max(max_idx - 10 -2, region_left) #NB overwrites!
                        right_boundary = min(max_idx + 60, region_right)
                    else:
                        # Find where we drop below threshold again
                        right_boundary = find_next_crossing(signal, threshold=settings['threshold'],
                                                            start=left_boundary, stop=region_right)
                        max_idx = None #This will be determined later

                    # Did the peak actually end? (in the tail of a big S2 it often doesn't)
                    if right_boundary == region_right:
                        break   # Dont search this interval: Xerawdp behaviour
                    # right_boundary -= 1 # Peak end is just before crossing # But Xerawdp doesn't do this either...

                    # Hand over to a function: this is needed because Xerawdp recurses for large s2s
                    self.find_peaks_in(event, peak_type, signal, settings, left_boundary, right_boundary,
                                       toplevel=True, max_idx=max_idx)

                    # Prepare for finding the next peak
                    if peak_type!='s1':
                        self.seeker_position = right_boundary

        del event['left_boundary_for_small_peak_isolation_test']
        del event['last_toplevel_max_val']
        del event['region_right_end_for_small_peak_isolation_test']
        return event

    def find_peaks_in(self, event, peak_type, signal, settings, left_boundary, right_boundary, toplevel=False, max_idx=None):
        self.log.debug("Searching for %s in %s - %s" % (peak_type, left_boundary, right_boundary))

        # Check if the interval is large enough to contain the peak
        if not settings['min_base_interval_length'] <= right_boundary - left_boundary <= settings['max_base_interval_length']:
            self.log.debug("    Interval failed width test %s <= %s <= %s" %(
                settings['min_base_interval_length'], right_boundary - left_boundary, settings['max_base_interval_length']
            ))
            return

        # Find the maximum index and height
        if max_idx is None:
            max_idx = left_boundary + np.argmax(signal[left_boundary:right_boundary+1])  # Remember silly python indexing
        height = signal[max_idx]
        if toplevel: event['last_toplevel_max_val'] = height # Hack for undocumented condition for skipping large_s2 tests
                                                             # Dirty, should perhaps pass yet another argument around..

        # Find the peak extent
        (left, right) = interval_until_threshold(
            signal,
            start=max_idx,
            left_threshold=settings['left_boundary_to_height_ratio'] * height,
            right_threshold=settings['right_boundary_to_height_ratio'] * height,
            left_limit=(nearest_s2_boundary(event, left_boundary, 'left') if peak_type=='large_s2' else left_boundary),
            right_limit=(nearest_s2_boundary(event, right_boundary, 'right') if peak_type=='large_s2' else right_boundary),
            stop_if_start_exceeded=settings['stop_if_start_exceeded'],
            min_crossing_length=settings['min_crossing_length'],
            activate_xerawdp_hacks_for=peak_type,
        )
        self.log.debug("    Possible %s found at %s: %s - %s" % (peak_type, max_idx, left, right))

        # For s1s, Xerawdp sets the seeker position now... right?
        if peak_type=='s1':
            # Next search should start after this peak - do this before testing the peak or the arcane +2
            self.seeker_position = right

        # Apply various tests to the peaks
        failed_some_test = False
        if peak_type == 'large_s2':

            # For large S2 from top-level interval: do the isolation check on the top interval
            # If this fails, we don't even search for additional s2s in the interval!
            if toplevel:
                if not isolation_test(signal, max_idx,
                                      right_edge_of_left_test_region=min(left_boundary, left)-1,
                                      test_length_left=settings['test_around_interval'],
                                      left_edge_of_right_test_region=max(right_boundary, right)+1,
                                      test_length_right=settings['test_around_interval'],
                                      before_avg_max_ratio=settings['around_interval_to_height_ratio_max'],
                                      after_avg_max_ratio=settings['around_interval_to_height_ratio_max']):
                    self.log.debug("    Toplevel interval failed isolation test")
                    return

            # If any of the below tests fail, we DO search the rest of the interval later, so don't return!

            # For large s2, the peak width is also tested (base interval width tested in both cases)
            if not settings['min_length'] <= right - left <= settings['max_length']:
                self.log.debug("    Failed width test")
                failed_some_test = True

            # An additional, different, isolation test is applied to every individual peak < 0.05 the toplevel peak
            if height < 0.05 * event['last_toplevel_max_val']:
                if not isolation_test(signal, max_idx,
                                      right_edge_of_left_test_region=left-1,
                                      test_length_left=settings['test_around_peak'],
                                      left_edge_of_right_test_region=right+1,
                                      test_length_right=settings['test_around_peak'],
                                      before_avg_max_ratio=settings['around_peak_to_height_ratio_max'],
                                      after_avg_max_ratio=settings['around_peak_to_height_ratio_max']):
                    self.log.debug("    Failed isolation test")
                    failed_some_test = True


        elif peak_type == 'small_s2':
            # The Small s2 search doesn't recurse, so returning when tests fail is fine

            # For small s2's the isolation test is slightly different
            if not isolation_test(signal, max_idx,
                                  right_edge_of_left_test_region=left-1,
                                  left_edge_of_right_test_region=right+1,
                                  # This is insane, and probably a bug, but I swear it's in Xerawdp
                                  test_length_left=min(settings['test_around'],right_boundary-event['left_boundary_for_small_peak_isolation_test']),
                                  test_length_right=min(settings['test_around'],event['region_right_end_for_small_peak_isolation_test']-right_boundary),
                                  before_avg_max_ratio=settings['around_to_height_ratio_max'],
                                  after_avg_max_ratio=settings['around_to_height_ratio_max']):
                self.log.debug("    Failed the isolation test.")
                return

            # Test for aspect ratio, probably to avoid misidentifying s1s as small s2s
            aspect_ratio_threshold = 0.062451  # 1 mV/bin = 0.1 mV/ns
            peak_width = (right - left) / units.ns
            aspect_ratio = height / peak_width
            if aspect_ratio > aspect_ratio_threshold:
                self.log.debug('    Failed aspect ratio test: max/width ratio %s is higher than %s' % (aspect_ratio, aspect_ratio_threshold))
                return

        elif peak_type == 's1':
            # Test for non-isolated peaks
            #TODO: dynamic window size if several s1s close together, check with Xerawdp now that free_regions
            if not isolation_test(signal, max_idx,
                                  right_edge_of_left_test_region=left-1,
                                  test_length_left=50,
                                  left_edge_of_right_test_region=right+1,
                                  test_length_right=10,
                                  before_avg_max_ratio=0.01,
                                  after_avg_max_ratio=0.04):
                return

            # Test for nearby negative excursions #Xerawdp bug: no check if is actually negative..
            negex = min(signal[
                max(0,max_idx-50 +1) :              #Need +1 for python indexing, Xerawdp doesn't do 1-correction here
                min(len(signal)-1,max_idx + 10 +1)
            ])
            if not height > 3 * abs(negex):
                return

            #Test for too wide s1s
            filtered_wave = event['processed_waveforms']['filtered_for_large_s2']   #I know, but that's how Xerawdp...
            max_in_filtered = left + int(np.argmax(filtered_wave[left:right]))
            filtered_width = extent_until_threshold(filtered_wave,
                                                    start=max_in_filtered,
                                                    threshold=0.25*filtered_wave[max_in_filtered])
            if filtered_width > 50:
                return

        # Append the peak found, if we haven't failed the tests
        if not failed_some_test:
            event['peaks'].append( {
                'left':         left - (2 if peak_type == 's1' else 0),     # Xerawdp weirdness
                'right':        right + (2 if peak_type == 's1' else 0),    # ditto
                'peak_type':    peak_type,
                'height':       height,
                'index_of_max_in_waveform': max_idx,
                'source_waveform':          settings['source_waveform'],
            })
        # For large s2s, recurse on the remaining intervals
        if peak_type == 'small_s2':
            event['left_boundary_for_small_peak_isolation_test'] = right
        elif peak_type == 'large_s2':
            # Recurse on remaining intervals right, then left
            # Ordering is different from Xerawdp because it uses a stack, we use recursion
            self.find_peaks_in(event, peak_type, signal, settings,
                               left_boundary=left_boundary, right_boundary=left) #Not left-1? Who is off by one?
            self.find_peaks_in(event, peak_type, signal, settings,
                               left_boundary=right, right_boundary=right_boundary) #Not right+1? Who is off by one?

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
            for channel, wave_data in event['processed_waveforms'].items():
                # Todo: use python's handy arcane naming/assignment convention to beautify this code
                peak_wave = wave_data[p['left']:p['right'] + 1]  # Remember silly python indexing
                peaks[i][channel] = {}
                maxpos = peaks[i][channel]['position_of_max_in_peak'] = np.argmax(peak_wave)
                maxval = peaks[i][channel]['height'] = peak_wave[maxpos]
                peaks[i][channel]['position_of_max_in_waveform'] = p['left'] + maxpos
                peaks[i][channel]['area'] = np.sum(peak_wave)
                if channel == 'top_and_bottom':
                    # Expensive stuff...
                    # Have to search the actual whole waveform, not peak_wave:
                    # TODO: Searches for a VERY VERY LONG TIME for weird peaks in afterpulse tail..
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

def isolation_test(
        signal, max_idx,
        right_edge_of_left_test_region, test_length_left,
        left_edge_of_right_test_region, test_length_right,
        before_avg_max_ratio,
        after_avg_max_ratio,
    ):
    """
    Does XerawDP's arcane isolation test. Returns if test is passed or not.
    TODO: Regions seem to come out empty sometimes... when? warn?
    """
    print("        Running isolation test: RofLeft %s, LofRight %s" % (right_edge_of_left_test_region, left_edge_of_right_test_region))
    # +1s are to compensate for python's indexing conventions...
    pre_avg = np.mean(signal[
         right_edge_of_left_test_region - (test_length_left-1):
         right_edge_of_left_test_region +1
    ])
    post_avg = np.mean(signal[
         left_edge_of_right_test_region:
         left_edge_of_right_test_region + test_length_right
    ])
    height = signal[max_idx]
    if max_idx == 23099:
        print("Pre avg %s (threshold %s), Post avg %s (threshold %s)" %(
            pre_avg, height * before_avg_max_ratio,
            post_avg, height * after_avg_max_ratio
        ))
    if pre_avg > height * before_avg_max_ratio or post_avg > height * after_avg_max_ratio:
        print("        Nope: %s > %s or %s > %s..." % (pre_avg, height * before_avg_max_ratio, post_avg, height * after_avg_max_ratio))
        return False
    else:
        print("        Passed!")
        return True

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
        raise ValueError("Invalid crossing search stop point: %s (signal has %s samples)" % (stop, len(signal)))
    if not 0 <= start <= len(signal) - 1:
        raise ValueError("Invalid crossing search start point: %s (signal has %s samples)" % (start, len(signal)))
    if direction not in ('left', 'right'):
        raise ValueError("Direction %s is not left or right" % direction)
    if (direction == 'left' and start < stop) or (direction == 'right' and stop < start):
        raise ValueError("Search region (start: %s, stop: %s, direction: %s) has negative length!" % (
            start, stop, direction
        ))
    if not 1 <= min_length:
       raise ValueError("min_length must be at least 1, %s specified." %  min_length)

    # Check for pathological cases not serious enough to throw an exception
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
        if i == stop:
            # stop_at reached, have to return something right now
            # Xerawdp keeps going, but always increments after_crossing_timer, so we know what it'll give
            if activate_xerawdp_hacks_for == 'large_s2':
                # We need to index of the minimum before & including this point instead...
                if direction=='left':
                    return i + np.argmin(signal[i:start+1])
                else: #direction=='right'
                    return start + np.argmin(signal[start:i+1])
            elif activate_xerawdp_hacks_for == 'small_s2':
                # Bug where it returns 1 too much!
                return stop + (-1 if direction == 'left' else 1)
            else:
                return stop + after_crossing_timer if direction == 'left' else stop - after_crossing_timer
        this_sample = signal[i]
        if stop_if_start_exceeded and this_sample > start_sample:
            #print("Emergency stop of search at %s: start value exceeded" % i)
            return i
        if start_sample < threshold < this_sample or start_sample > threshold > this_sample:
            # We're on the other side of the threshold that at the start!
            after_crossing_timer += 1
            if after_crossing_timer == min_length:
                if activate_xerawdp_hacks_for in ('small_s2', 'large_s2'):
                    # Again, bug in Xerawdp: it returns 1 too much.
                    return i + (-1 if direction == 'left' else 1)
                else:
                    return i + (min_length - 1 if direction == 'left' else 1 - min_length)  #
        else:
            # We're back to the old side of threshold again
            after_crossing_timer = 0

        # Dirty hack for Xerawdp matching
        if activate_xerawdp_hacks_for == 'large_s2':
            # Check also for slope inversions
            if this_sample > 7.801887059:   #'0.125 V' #Todo: check if enough samples exist to compute slope..
                # We need to check for slope inversions. How bad is it allowed to be?
                if this_sample < 39.00943529: #'0.625 V'
                    log_slope_threshold = 0.02 #Xerawdp says '0.02 V/bin', but it is a log slope threshold...
                else:
                    log_slope_threshold = 0.005 #Idem '0.005 V/bin'

                # Calculate the slope at this point using XeRawDP's '9-tap derivative kernel'
                log_slope = np.sum(signal[i-4:i+5] * np.array([-0.003059, -0.035187, -0.118739, -0.143928, 0.000000, 0.143928, 0.118739, 0.035187, 0.003059]))/this_sample
                #print("Slope is %s, threshold is %s" % (slope, slope_threshold))
                # Left slopes of peaks are positive, so a negative slope indicates inversion
                # If slope inversions are seen, return index of the minimum before this.
                if direction=='left' and log_slope < - log_slope_threshold:
                    print("    ! Inversion found on left slope at %s: log_slope %s > %s. Started from %s" %  (i, log_slope, log_slope_threshold, start))
                    return i + np.argmin(signal[i:start+1])
                elif direction=='right' and log_slope > log_slope_threshold:
                    print("    ! Inversion found on right slope at %s: log_slope %s > %s. started from %s" % (i, log_slope, log_slope_threshold, start))
                    return start + np.argmin(signal[start:i+1])


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
    if l_cross != left_limit:
        l_cross += 1    #We want the last index still on ok-side!
    if r_cross != right_limit:
        r_cross -= 1    #We want the last index still on ok-side!
    return (l_cross, r_cross)


def extent_until_threshold(signal, start, threshold):
    a = interval_until_threshold(signal, start, threshold)
    return a[1] - a[0]

# TODO: Move this to event class?
def get_free_regions(event):
    lefts = sorted([0] + [p['left'] for p in event['peaks']])
    rights = sorted([p['right'] for p in event['peaks']] + [event['length']-1])
    free_regions = list(zip(*[iter(sorted(lefts + rights))]*2))   #hack from stackoverflow
    print("Free regions: " + str(free_regions))
    return free_regions

def nearest_s2_boundary(event, position, direction='left'):
    if direction=='left':
        boundaries = [p['right'] for p in event['peaks'] if p['peak_type'] in ('small_s2', 'large_s2') and p['right']<position]
        return max(boundaries) if len(boundaries) > 0 else 0
    elif direction=='right':
        boundaries = [p['left']  for p in event['peaks'] if p['peak_type'] in ('small_s2', 'large_s2') and p['left']>position]
        return min(boundaries) if len(boundaries) > 0 else event['length']-1
    raise RuntimeError("direction %s isn't left or right" % direction)


# if peak_type == 'large_s2':
#     with open('s2_dump_pax.txt','w') as output:
#         output.write("\n".join(map(str,signal)))
# if peak_type == 'small_s2':
#     with open('s2small_dump_pax.txt','w') as output:
#         output.write("\n".join(map(str,signal)))
#     exit()
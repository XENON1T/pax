import numpy as np
from copy import copy

from pax import plugin, units

class FindS1_XeRawDPStyle(plugin.TransformPlugin):

    def transform_event(self, event):
        signal = event['processed_waveforms']['uncorrected_sum_waveform_for_s1']
        s1_alert_treshold = 0.1872452894  # "3 mV"

        # We can stop looking for s1s after the largest s2, or after any sufficiently large s2
        s2s = [p for p in event['peaks'] if p['peak_type'] in ('large_s2', 'small_s2')]
        if s2s:
            stop_looking_after = s2s[ np.argmax([p['height'] for p in s2s]) ]['left']    # Left boundary of largest s2
            large_enough_s2s = [p for p in s2s if p['height'] > 3.12255]
            if large_enough_s2s:
                stop_looking_after = min(
                    stop_looking_after,
                    min([p['left'] for p in large_enough_s2s])
                )
        else:
            stop_looking_after = float('inf')

        # Find all intervals around the (s2) peaks that have already been found, search for s1s in these
        free_regions = get_free_regions(event) # Can't put this in loop, peaks get added!
        for region_left, region_right in free_regions:

            #Do we actually need to search this and any further regions?
            if region_left >= stop_looking_after: break
            seeker_position = region_left
            while 1:
                # Are we perhaps above threshold already? if so, move along until we're not
                while signal[region_left] > s1_alert_treshold:
                    seeker_position = find_next_crossing(signal, s1_alert_treshold,
                                                        start=seeker_position, stop=region_right)
                    if seeker_position == region_right:
                        self.log.warning('Entire s1 search region from %s to %s is above s1_alert threshold %s!' %(
                            region_left, region_right, s1_alert_treshold
                        ))
                        break
                if seeker_position == region_right: break

                # Seek until we go above s1_alert_threshold
                potential_s1_start = find_next_crossing(signal, s1_alert_treshold,
                                                        start=seeker_position, stop=region_right)
                if potential_s1_start == region_right:
                    # Search has reached the end of the waveform
                    break

                # Determine the maximum in the 'preliminary peak window' (next 60 samples) that follows
                max_pos = np.argmax(signal[potential_s1_start:min(potential_s1_start + 60, region_right)])
                max_idx = potential_s1_start + max_pos
                height = signal[max_idx]

                # Set a revised peak window based on the max position
                # Should we also check for previous S1s? or prune overlap later? (Xerawdp doesn't do either?)
                peak_window = (
                    max(max_idx - 10 -2, region_left),
                    min(max_idx + 60, region_right)
                )

                # Find the peak boundaries
                peak_bounds = interval_until_threshold(signal, start=max_idx,
                                                       left_threshold=0.005*height, #r-trh automatically =
                                                       left_limit=peak_window[0], right_limit=peak_window[1],
                                                       min_crossing_length=3, stop_if_start_exceeded=True)

                # Next search should start after this peak - do this before testing the peak or the arcane +2
                seeker_position = copy(peak_bounds[1])

                # Test for non-isolated peaks
                #TODO: dynamic window size if several s1s close together, check with Xerawdp now that free_regions
                if not isolation_test(signal, max_idx,
                                      right_edge_of_left_test_region=peak_bounds[0]-1,
                                      test_length_left=50,
                                      left_edge_of_right_test_region=peak_bounds[1]+1,
                                      test_length_right=10,
                                      before_avg_max_ratio=0.01,
                                      after_avg_max_ratio=0.04):
                    continue

                # Test for nearby negative excursions #Xerawdp bug: no check if is actually negative..
                negex = min(signal[
                    max(0,max_idx-50 +1) :              #Need +1 for python indexing, Xerawdp doesn't do 1-correction here
                    min(len(signal)-1,max_idx + 10 +1)
                ])
                if not height > 3 * abs(negex):
                    continue

                #Test for too wide s1s
                filtered_wave = event['processed_waveforms']['filtered_for_large_s2']   #I know, but that's how Xerawdp...
                max_in_filtered = peak_bounds[0] + np.argmax(filtered_wave[peak_bounds[0]:peak_bounds[1]])
                filtered_width = extent_until_threshold(filtered_wave,
                                                        start=max_in_filtered,
                                                        threshold=0.25*filtered_wave[max_in_filtered])
                if filtered_width > 50:
                    continue

                # Xerawdp weirdness
                peak_bounds = (peak_bounds[0]-2,peak_bounds[1]+2)

                # Don't have to test for s1 > 60: can't happen due to limits on peak window
                # That's nonsense btw! TODO?

                #Add the s1 we found
                event['peaks'].append({
                    'peak_type':                's1',
                    'left':                     peak_bounds[0],
                    'right':                    peak_bounds[1],
                    's1_peakfinding_window':    peak_window,
                    'index_of_max_in_waveform': max_idx,
                    'height':                   height,
                    'source_waveform':          'uncorrected_sum_waveform_for_s1',
                })

        return event


class FindS2_XeRawDPStyle(plugin.TransformPlugin):

    def startup(self):
        self.settings_for_peaks = [ #We need large_s2 first, so can't use a dict
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
            }),
            ('small_s2', {
                'threshold':        0.0624509324,
                'left_boundary_to_height_ratio':    0.01,
                'right_boundary_to_height_ratio':   0.01,
                'min_base_interval_length':       40,
                'max_base_interval_length':       200,
                'source_waveform':  'filtered_for_small_s2',
                'around_to_height_ratio_max': 0.05,
                'test_around': 10,
            })
        ]

    def transform_event(self, event):
        event['peaks'] = []
        # Do everything first for large s2s, then for small s2s
        for (peak_type, settings) in self.settings_for_peaks:
            # Get the signal out
            signal = event['processed_waveforms'][settings['source_waveform']]

            # For small s2s, we looking stop after a sufficiently large s2 is seen
            # We can stop looking for s1s after the largest s2, or after any sufficiently large s2
            stop_looking_after = float('inf')
            if peak_type == 'small_s2':
                #Don't have to test for p['peak_type'] == 'large_s2', these are the only peaks in here
                huge_s2s = [p for p in event['peaks'] if p['height'] > 624.151]
                if huge_s2s:
                    stop_looking_after = min([p['left'] for p in huge_s2s])

            # Find peaks in all the free regions
            free_regions = get_free_regions(event) # Can't put this in loop, peaks get added!
            for region_left, region_right in free_regions:
                # Don't look for small S2s after a very large S2
                if peak_type=='small_s2' and region_left >= stop_looking_after: break
                event['region_right_end_for_small_peak_isolation_test'] = region_right
                event['left_boundary_for_small_peak_isolation_test'] = region_left
                seeker_position = region_left
                while 1:
                    #TODO: don't copy paste from s1...
                    # Are we perhaps above threshold already? if so, move along until we're not
                    while signal[seeker_position] > settings['threshold']:
                        seeker_position = find_next_crossing(signal, settings['threshold'],
                                                            start=seeker_position, stop=region_right)
                        if seeker_position == region_right:
                            self.log.warning('Entire %s search region from %s to %s is above threshold %s!' %(
                                peak_type, region_left, region_right, settings['threshold']
                            ))
                            break
                    if seeker_position == region_right: break
                    assert signal[seeker_position] < settings['threshold']

                    # Find the next threshold crossing, if it exists
                    left_boundary= find_next_crossing(signal, threshold=settings['threshold'],
                                                      start=seeker_position, stop=region_right)
                    if left_boundary == region_right:
                        break # There wasn't another crossing

                    # There was a crossing, so we're starting a maybe-peak, find where this peak ends
                    right_boundary = find_next_crossing(signal, threshold=settings['threshold'],
                                                        start=left_boundary, stop=region_right)
                    seeker_position = right_boundary

                    # Did the peak actually end? (in the tail of a big S2 it often doesn't)
                    if right_boundary == region_right:
                        break   # Dont search this interval: Xerawdp behaviour
                    right_boundary -= 1 # Peak end is just before crossing

                    # Hand over to a function, needed because Xerawdp recurses for large s2s
                    s2s_found = self.find_s2s_in(event, peak_type, signal, settings, left_boundary, right_boundary, toplevel=True)
                    if s2s_found is not []:
                        event['peaks'] += s2s_found

        del event['left_boundary_for_small_peak_isolation_test']
        del event['last_toplevel_max_val']
        del event['region_right_end_for_small_peak_isolation_test']
        return event

    #TODO: Probably want to make a find_small_s2s_in and find_large_s2s_in, gets rid of all the ifs
    def find_s2s_in(self, event, peak_type, signal, settings, left_boundary, right_boundary, toplevel=False):
        self.log.debug("Searching for %s in interval %s to %s" % (peak_type, left_boundary, right_boundary))

        # Check if the interval is large enough to contain an s2
        if not settings['min_base_interval_length'] <= right_boundary - left_boundary <= settings['max_base_interval_length']:
            return []

        # Find the maximum index and height
        max_idx = left_boundary + np.argmax(signal[left_boundary:right_boundary+1])  # Remember silly python indexing
        height = signal[max_idx]
        if toplevel: event['last_toplevel_max_val'] = height # Hack for undocumented condition for skipping large_s2 tests
                                                             # Dirty, should perhaps pass yet another argument around..

        # Todo: for large s2's, also check for slope changes!
        (left, right) = interval_until_threshold(
            signal,
            start=max_idx,
            left_threshold=settings['left_boundary_to_height_ratio'] * height,
            right_threshold=settings['right_boundary_to_height_ratio'] * height,
            left_limit=left_boundary,
            right_limit=right_boundary,
            stop_if_start_exceeded=(peak_type=='small_s2'), # Not for large s2s? Bug?
            activate_xerawdp_hacks_for=peak_type,
        )
        self.log.debug("    S2 found at %s: %s - %s" % (max_idx, left, right))

        # If we pass the tests, this is what we'll store
        this_s2 = {
            'left':         left,
            'right':        right,
            'peak_type':    peak_type,
            'height':       height,
            'index_of_max_in_waveform': max_idx,
            'source_waveform':          settings['source_waveform'],
        }

        # Apply various tests to the peaks
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
                    return []

            # If any of the below tests fail, we DO search the rest of the interval later, so don't return!

            # For large s2, the peak width is also tested (base interval width tested in both cases)
            if not settings['min_length'] <= right - left <= settings['max_length']:
                self.log.debug("    Failed width test")
                this_s2 = None

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
                    this_s2 = None


        elif peak_type == 'small_s2':
            # The Small s2 search doesn't recurse, so returning [] when tests fail is fine

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
                return []

            # Test for aspect ratio, probably to avoid misidentifying s1s as small s2s
            aspect_ratio_threshold = 0.062451  # 1 mV/bin = 0.1 mV/ns
            peak_width = (right - left) / units.ns
            aspect_ratio = height / peak_width
            if aspect_ratio > aspect_ratio_threshold:
                self.log.debug('    For peak at %s, Max/width ratio %s is higher than %s' % (max_idx, aspect_ratio, aspect_ratio_threshold))
                return []

        #Return the s2 found, and for large s2s, recurse on the remaining intervals
        if peak_type == 'small_s2':
            event['left_boundary_for_small_peak_isolation_test'] = right
            return [this_s2]
        elif peak_type == 'large_s2':
            # Recurse on remaining intervals left & right, build list
            other_s2s =   self.find_s2s_in(event, peak_type, signal, settings,
                                           left_boundary=left_boundary, right_boundary=left-1) \
                        + self.find_s2s_in(event, peak_type, signal, settings,
                                           left_boundary=right+1, right_boundary=right_boundary)
            if this_s2 is None:
                return other_s2s
            else:
                return [this_s2] + other_s2s


# TODO: Veto S1 peakfinding

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
        #self.log.debug("        Nope: %s > %s or %s > %s..." % (pre_avg, height * before_avg_max_ratio, post_avg, height * after_avg_max_ratio))
        return False
    else:
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
            if this_sample > 7.801887059:   #'0.125 V' #Todo: check if enough samples exist to compute slope..
                # We need to check for slope inversions. How bad is it allowed to be?
                if this_sample < 39.00943529: #'0.625 V'
                    slope_threshold = 1.248301929 #'0.02 V/bin' -> pe/bin
                else:
                    slope_threshold = 0.3120754824 #'0.005 V/bin' -> pe/bin

                # Calculate the slope at this point using XeRawDP's '9-tap derivative kernel'
                slope = np.sum(signal[i-4:i+5] * np.array([-0.003059, -0.035187, -0.118739, -0.143928, 0.000000, 0.143928, 0.118739, 0.035187, 0.003059]))/this_sample
                #print("Slope is %s, threshold is %s" % (slope, slope_threshold))
                # Left slopes of peaks are positive, so a negative slope indicates inversion
                # If slope inversions are seen, return index of the minimum before this.
                if direction=='left' and slope < - slope_threshold:
                    print("    ! Inversion found on left slope at %s: slope %s > %s. Started from %s" %  (i, slope, slope_threshold, start))
                    return i + np.argmin(signal[i:start+1])
                elif direction=='right' and slope > slope_threshold:
                    print("    ! Inversion found on right slope at %s: slope %s > %s. started from %s" % (i, slope, slope_threshold, start))
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
    return [(lefts[i], rights[i]) for i in range(len(lefts))]    # There's probably a zip hack for this
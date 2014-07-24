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
                if signal[region_left] > s1_alert_treshold:
                    seeker_position = find_next_crossing(signal, s1_alert_treshold,
                                                        start=seeker_position, stop=region_right)
                    if seeker_position == region_right:
                        self.log.warning('Entire s1 search region from %s to %s is above s1_alert threshold %s!' %(
                            region_left, region_right, s1_alert_treshold
                        ))
                        break

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
                'threshold':        0.624509647,
                'left_boundary_to_height_ratio':    0.005,
                'right_boundary_to_height_ratio':   0.002,
                'min_length':       60,
                'max_length':       float('inf'),
                'source_waveform':  'filtered_for_large_s2',
                'around_to_height_ratio_max': 0.05,
                'test_around': 21,
            }),
            ('small_s2', {
                'threshold':        0.0624509647,
                'left_boundary_to_height_ratio':    0.01,
                'right_boundary_to_height_ratio':   0.01,
                'min_length':       40,
                'max_length':       200,
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
                left_boundary_for_small_peak_isolation_test = region_left
                seeker_position = region_left
                while 1:
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

                    # If we are searching for large s2, hand over to a function
                    # neefed because Xerawdp recurses
                    if peak_type == 'large_s2':
                        s2s_found = self.find_large_s2s_in(event, signal, settings, left_boundary, right_boundary, toplevel=True)
                        if s2s_found is not None:
                            event['peaks'] += s2s_found
                        continue

                    # We only get here if we are searching for small s2s

                    # For small s2s, the INTERVAL width is tested
                    if not settings['min_length'] <= right_boundary - left_boundary <= settings['max_length']:
                        continue

                    # Find the maximum and the peak extent
                    max_idx = left_boundary + np.argmax(signal[left_boundary:right_boundary+1])    # Remember silly python indexing
                    height = signal[max_idx]

                    # Find the peak extent
                    (left, right) = interval_until_threshold(
                        signal,
                        start=max_idx,
                        left_threshold=settings['left_boundary_to_height_ratio'] * height,
                        right_threshold=settings['right_boundary_to_height_ratio'] * height,
                        left_limit=left_boundary,
                        right_limit=right_boundary,
                    )
                    #Xerawdp bug: S2 sizes are reported 1 too large on both sides. Simulating this bug:
                    left -=1
                    right += 1

                    the_thing = False
                    if peak_type=='small_s2' and max_idx == 23099:
                        the_thing = True
                        print("Found our test peak! L: %s, R: %s" % (left, right))

                    # For small s2's the isolation test is slightly different
                    if not isolation_test(signal, max_idx,
                                          right_edge_of_left_test_region=left-1,
                                          left_edge_of_right_test_region=right+1,
                                          # This is insane, and probably a bug, but I swear it's in Xerawdp
                                          test_length_left=min(settings['test_around'],right_boundary-left_boundary_for_small_peak_isolation_test),
                                          test_length_right=min(settings['test_around'],region_right-right_boundary),
                                          before_avg_max_ratio=settings['around_to_height_ratio_max'],
                                          after_avg_max_ratio=settings['around_to_height_ratio_max']):
                        continue

                    # Test for aspect ratio, probably to avoid misidentifying s1s as small s2s
                    aspect_ratio_threshold = 0.062451  # 1 mV/bin = 0.1 mV/ns
                    peak_width = (right - left) / units.ns
                    aspect_ratio = height / peak_width
                    if aspect_ratio > aspect_ratio_threshold:
                        #print('For peak at %s, Max/width ratio %s is higher than %s' % (max_idx, aspect_ratio, aspect_ratio_threshold))
                        continue
                    if the_thing: print("It passed the aspect ratio test.")

                    #Append the peak we've found (copy paste again...)
                    event['peaks'].append({
                        'left':         left,
                        'right':        right,
                        'peak_type':    'small_s2',
                        'height':       height,
                        'index_of_max_in_waveform': max_idx,
                        'source_waveform': settings['source_waveform'],
                    })
                    left_boundary_for_small_peak_isolation_test = right

        return event

    #TODO: Probably want to make a find_small_s2s_in and find_large_s2s_in, gets rid of all the ifs
    @staticmethod
    def find_large_s2s_in(event, signal, settings, left_boundary, right_boundary, toplevel=False):

        # Copy-Paste from small s2
        max_idx = left_boundary + np.argmax(signal[left_boundary:right_boundary+1])    # Remember silly python indexing
        height = signal[max_idx]

        # Todo: for large s2's, also check for slope changes!
        (left, right) = interval_until_threshold(
            signal,
            start=max_idx,
            left_threshold=settings['left_boundary_to_height_ratio'] * height,
            right_threshold=settings['right_boundary_to_height_ratio'] * height,
            left_limit=left_boundary,
            right_limit=right_boundary,
        )
        #Xerawdp bug: S2 sizes are reported 1 too large on both sides. Simulating this bug:
        left -=1
        right += 1

        # For large S2 from top-level interval: do the isolation check
        if toplevel:
            if not isolation_test(signal, max_idx,
                                  right_edge_of_left_test_region=min(left_boundary, left)-1,
                                  test_length_left=settings['test_around'],
                                  left_edge_of_right_test_region=max(right_boundary, right)+1,
                                  test_length_right=settings['test_around'],
                                  before_avg_max_ratio=settings['around_to_height_ratio_max'],
                                  after_avg_max_ratio=settings['around_to_height_ratio_max']):
                return

        # For large s2, the PEAK width is tested
        if not settings['min_length'] <= right - left <= settings['max_length']:
            return

        # Todo: for large_s2s, recurse on remaining intervals left & right, build list
        return [{
            'left':         left,
            'right':        right,
            'peak_type':    'large_s2',
            'height':       height,
            'index_of_max_in_waveform': max_idx,
            'source_waveform':          settings['source_waveform'],
        }]


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


# #
# Utils: these are helper functions for the plugins
# TODO: Can we put these functions in the transform base class? (Tunnell: yes, or do multiple inheritance)
# #

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
        return False
    else:
        return True

def find_next_crossing(signal, threshold,
                       start=0, direction='right', min_length=1,
                       stop=None, stop_if_start_exceeded=False):
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
    stop_if_start_exceeded -- If true and a value HIGHER than start is encountered, stop immediately

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
            return stop + after_crossing_timer if direction == 'left' else stop - after_crossing_timer
        this_sample = signal[i]
        if stop_if_start_exceeded and this_sample > start_sample:
            #print("Emergency stop of search at %s: start value exceeded" % i)
            return i
        if start_sample < threshold < this_sample or start_sample > threshold > this_sample:
            # We're on the other side of the threshold that at the start!
            after_crossing_timer += 1
            if after_crossing_timer == min_length:
                return i + (min_length - 1 if direction == 'left' else 1 - min_length)  #
        else:
            # We're back to the old side of threshold again
            after_crossing_timer = 0
        i += -1 if direction == 'left' else 1


def interval_until_threshold(signal, start,
                             left_threshold, right_threshold=None, left_limit=0, right_limit=None,
                             min_crossing_length=1, stop_if_start_exceeded=False
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
                           stop_if_start_exceeded=stop_if_start_exceeded)
    r_cross = find_next_crossing(signal, right_threshold, start=start, stop=right_limit,
                           direction='right', min_length=min_crossing_length,
                           stop_if_start_exceeded=stop_if_start_exceeded)
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
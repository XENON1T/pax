import numpy as np

def baseline_mean_stdev(waveform, sample_size=46):
    """ Returns (baseline, baseline_stdev) calculated on the first sample_size samples of waveform """
    """
    This is how XerawDP does it... but this may be better:
    return (
        np.mean(sorted(baseline_sample)[
                int(0.4 * len(baseline_sample)):int(0.6 * len(baseline_sample))
                ]),  # Ensures peaks in baseline sample don't skew computed baseline
        np.std(baseline_sample)  # ... but do count towards baseline_stdev!
    )
    Don't want to just take the median as V-resolution is finite
    Don't want the mean either: this is not robust against large fluctuations (eg peaks in sample)
    See also comment in JoinAndConvertWaveforms
    """

    baseline_sample = waveform[:sample_size]
    return np.mean(baseline_sample), np.std(baseline_sample)


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
            # stop_at reached, have to result something
            return stop
        this_sample = signal[i]
        if stop_if_start_exceeded and this_sample > start_sample:
            print("Emergency stop of search at %s: start value exceeded" % i)
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
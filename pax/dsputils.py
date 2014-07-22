import numpy as np

#UNUSED REMOVE
def baseline_mean_stdev(waveform, sample_size=46, is_first_occurence=False):
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
    if len(waveform)<2*sample_size:
        if not is_first_occurence:
            raise RuntimeError("Occurence has length %s, should be at least 2*46!" % len(waveform))
        print("XeRawDP feature: computing baseline from latest samples")
        baseline_sample = waveform[len(waveform)-sample_size:]
    else:
        baseline_sample = waveform[:sample_size]
    return np.mean(baseline_sample), np.std(baseline_sample)



Definite:

    Peak finding is done on a waveform that is summed, but not gain corrected
        Properly summed&corrected waveform is used to compute areas etc
        Dead channels were specified by putting gain=0, so junk in there contributes to peak finding!

    Baseline computation is very primitive: just mean of every occurrence's 46 first samples individually
        Better: take median.
        Better: take mean of median 20% or so.
        Better: consider all pulses in a channel to compute a single baseline.
                (assuming baseline doesn't fluctuate within the event)
        Better: Keep a running mean/median over several events.
                (assuming baseline fluctuates at most slowly between events)
        Maybe better: use prior knowledge of baseline (it always seems to come out near 16 000)
        How big a difference does this make


Potential:

    S2 tiny peak finding code occasionally uses large s2 giltered waveform
        Probably a bug from copy-pasting code, not a big deal (I think?) as waveforms are very similar

    Maybe misses some S1 due to strong pruning of non-isolated peaks, especially if one photon comes late

    Some places appear to miss checks for array index overruns

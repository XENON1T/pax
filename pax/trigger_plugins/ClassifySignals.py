import numba
from pax.trigger import TriggerPlugin


class ClassifySignals(TriggerPlugin):

    def process(self, data):
        classify_signals(data.signals,
                         s1_max_rms=self.config['s1_max_rms'],
                         s2_min_pulses=self.config['s2_min_pulses'])


@numba.jit(nopython=True)
def classify_signals(signals, s1_max_rms, s2_min_pulses):
    """Set the type field of signals to 0 (unknown), 1 (s1) or 2 (s2). Modifies signals in-place.
    """
    for signal_i, s in enumerate(signals):
        sigtype = 0
        if s.time_rms > s1_max_rms:
            if s.n_pulses >= s2_min_pulses:
                sigtype = 2
        else:
            sigtype = 1
        signals[signal_i].type = sigtype

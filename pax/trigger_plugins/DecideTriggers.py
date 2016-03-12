import numpy as np
import numba
import random
from pax.trigger import TriggerPlugin


class DecideTriggers(TriggerPlugin):

    def startup(self):
        # Convert trigger_probability dictionary of dictionaries to 2d array, with intermediate values filled in
        self.log.info("\tTrigger probabilities: %s" % str(self.config['trigger_probability']))
        p_length = max([max(ps.keys()) for ps in self.config['trigger_probability'].values()]) + 1
        p_matrix = np.zeros((3, p_length), dtype=np.float)
        for sig_type, ps in self.config['trigger_probability'].items():
            for i, n in enumerate(sorted(ps.keys())):
                p_matrix[sig_type][n:] = ps[n]
        self.p_matrix = p_matrix

    def process(self, data):
        flag_triggers(data.signals, p_matrix=self.p_matrix)


@numba.jit(nopython=True)
def flag_triggers(signals, p_matrix):
    """Decide which signals trigger, modifying signals in-place.
    p_matrix[signal_type][n_pulses] is the probability of a signal of type signal_type and n_pulses pulses to trigger
    The last entry of each signal type is used for signals with more pulses than the matrix is wide.
    """
    largest_n_in_p = len(p_matrix[0]) - 1
    for i, s in enumerate(signals):
        p = p_matrix[s.type][min(largest_n_in_p, s.n_pulses)]

        if p == 1.0:
            s.trigger = True
        elif p == 0.0:
            s.trigger = False
        else:
            s.trigger = random.random() < p

import numpy as np
import numba
import random
from pax.trigger import TriggerPlugin
from pax.trigger import TriggerSignal


class DecideTriggers(TriggerPlugin):

    def startup(self):
        # Convert trigger_probability dictionary keys to ints (due to json arcana they must be strings...)
        trig_prob = dict()
        for k, v in self.config['trigger_probability'].items():
            trig_prob[int(k)] = dict()
            for k2, v2 in v.items():
                trig_prob[int(k)][int(k2)] = v2

        # Convert trigger_probability dictionary of dictionaries to 2d array, with intermediate values filled in
        self.log.info("\tTrigger probabilities: %s" % str(trig_prob))
        p_length = max([max(ps.keys()) for ps in trig_prob.values()]) + 1
        p_matrix = np.zeros((3, p_length), dtype=np.float)
        for sig_type, ps in trig_prob.items():
            for i, n in enumerate(sorted(ps.keys())):
                p_matrix[sig_type][n:] = ps[n]
        self.p_matrix = p_matrix

    def process(self, data):
        if self.config.get('do_not_trigger', False):
            return
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


class AnyPulseFromDetectorTrigger(TriggerPlugin):
    """Make a triggering signal for EVERY individual pulse in a particular detector"""

    def process(self, data):
        for detector, typecode in self.config['typecodes_for_detector'].items():
            channels = self.trigger.pax_config['DEFAULT']['channels_in_detector'][detector]
            pulses = data.pulses[np.in1d(data.pulses['pmt'], channels)]
            if not len(pulses):
                continue
            self.log.debug("Triggering on every individual %d pulses from detector %s" % (len(pulses), detector))

            new_sigs = np.zeros(len(pulses), dtype=TriggerSignal.get_dtype())

            new_sigs['trigger'] = True
            new_sigs['type'] = typecode
            new_sigs['left_time'] = new_sigs['right_time'] = new_sigs['time_mean'] = pulses['time']
            new_sigs['area'] = pulses['area']

            new_sigs['time_rms'] = 0
            new_sigs['n_pulses'] = 1
            new_sigs['n_contributing_channels'] = 1

            data.signals = np.concatenate((data.signals, new_sigs))
            data.signals.sort(order='left_time')

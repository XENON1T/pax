import numpy as np
import time
import csv
try:
    import cPickle as pickle
except:
    import pickle
from pax import plugin, units, datastructure, simulation


class WaveformSimulator(plugin.InputPlugin):
    """ Common I/O for waveform simulator: truth file writing, pax event creation, etc.
    There is no physics here: all that is in pax.simulation
    """
    def startup(self):
        # Open the truth file
        self.truth_file = open(self.config['truth_filename'], 'wb')
        self.config = simulation.init_config(self.config)

    def s2(self, electrons, t=0., z=0., x=0., y=0.):
        electron_times = simulation.s2_electrons(electrons_generated=electrons, t=t, z=z)
        photon_times = simulation.s2_scintillation(electron_times)
        self.truth_peaks.append({
            'signal_type': 's2',
            'x': x, 'y': y, 'z': z,
            't_interaction': t,
            'photons': len(photon_times),
            't_photons': np.mean(photon_times),
            'electrons': len(electron_times),
            'e_cloud_sigma': np.std(electron_times),
        })
        return simulation.hitlist_to_waveforms(
            simulation.photons_to_hitlist(photon_times),
        )

    def s1(self, photons, t=0, recombination_time=None, singlet_fraction=None):
        """
        :param photons: total # of photons generated in the S1
        :param t: Time at which the interaction occurs, i.e. offset for arrival times. Defaults to s1_default_recombination_time
        :param recombination_time: Fraction of recombining eximers that decay as singlets. Defaults to s1_default_eximer_fraction
        :param singlet_fraction: Recombination time (\tau_r in Nest papers).
        :return: start_time, pmt_waveforms
        """
        photon_times = simulation.s1_photons(photons, t, recombination_time, singlet_fraction)
        self.truth_peaks.append({
            'peak_type': 's1',
            #'x': x, 'y': y, 'z': z,
            't_interaction': t,
            'photons': len(photon_times),
            't_photons': np.mean(photon_times),
        })
        return simulation.hitlist_to_waveforms(
            simulation.photons_to_hitlist(photon_times),
        )

    def get_events(self):

        dt = self.config['digitizer_t_resolution']
        event_number = 0
        for instructions in self.get_instructions_for_next_event():
            for repetition_i in range(self.config['event_repetitions']):
                self.truth_peaks = []

                # Make (start_time, waveform matrix) tuples for every s1/s2 we have to generate
                signals = []
                for q in instructions:
                    self.log.debug("Simulating %s photons and %s electrons at %s cm depth, at t=%s ns" % (
                        q['s1_photons'], q['s2_electrons'], q['depth'], q['t']
                    ))
                    if int(q['s1_photons']):
                        signals += [self.s1(
                            int(q['s1_photons']),
                            t=float(q['t'])
                        )]
                    if int(q['s2_electrons']):
                        signals += [self.s2(
                            int(q['s2_electrons']),
                            z=float(q['depth']) * units.cm,
                            t=float(q['t'])
                        )]

                # Remove empty signals (None) from signal list
                signals = [s for s in signals if s is not None]
                if len(signals) == 0:
                    self.log.warning(
                        "Fax simulation returned no signals, can't return an event...")
                    continue

                # Combine everything into a single waveform matrix
                # Baseline addition & flipping down is done here
                self.log.debug("Combining %s signals into a single matrix" % len(signals))
                start_time_offset = min([s[0] for s in signals])
                # Compute event length in samples
                event_length = int(
                     2 * self.config['event_padding'] / dt +
                     max([s[0] - start_time_offset for s in signals]) / dt +
                     max([s[1].shape[1] for s in signals])
                )
                pmt_waveforms = self.config['digitizer_baseline'] *\
                                np.ones((len(self.config['channels']), event_length), dtype=np.int16)
                for s in signals:
                    start_index = int(
                        (s[0] - start_time_offset + self.config['event_padding']) / dt
                    )
                    # NOTE: MINUS!
                    pmt_waveforms[:, start_index:start_index + s[1].shape[1]] -= s[1]
                # Clipping
                pmt_waveforms = np.clip(pmt_waveforms, 0, 2 ** (self.config['digitizer_bits']))

                # Create and yield a pax event
                self.log.debug("Creating pax event")
                event = datastructure.Event()
                event.event_number = event_number
                now = int(time.time() * units.s)
                event.start_time = int(now)
                event.stop_time = event.start_time + int(event_length * dt)
                event.sample_duration = dt
                # Make a single occurrence for the entire event... yeah, this
                # should be changed
                event.occurrences = {
                    ch: [(0, pmt_waveforms[ch])]
                    for ch in self.config['channels']
                }
                self.log.debug("These numbers should be the same: %s %s %s %s" % (
                    pmt_waveforms.shape[1], event_length, event.length(), event.occurrences[1][0][1].shape))
                yield event

                # Write the truth of the event
                # Remove start time offset from all times in the peak
                for p in self.truth_peaks:
                    p['t_interaction'] -= start_time_offset
                    p['t_photons'] -= start_time_offset
                self.write_truth({
                    'event_number' : event_number,
                    'peaks' : self.truth_peaks
                })

                event_number += 1

    def write_truth(self, stuff):
        pickle.dump(stuff, self.truth_file)



class WaveformSimulatorFromCSV(WaveformSimulator):
    def startup(self):
        # Open the instructions file
        self.instructions_file = open(self.config['instruction_filename'], 'r')
        self.instructions = csv.DictReader(self.instructions_file)
        WaveformSimulator.startup(self)

    def shutdown(self):
        self.instructions_file.close()

    def get_instructions_for_next_event(self):
        this_event = 0
        this_event_peaks = []
        for p in self.instructions:
            if int(p['event']) == this_event:
                this_event_peaks.append(p)
            else:
                # New event reached!
                # How often do we need to repeat the old event?
                yield this_event_peaks
                this_event = int(p['event'])
                this_event_peaks = [p]
        # For the final event...
        yield this_event_peaks

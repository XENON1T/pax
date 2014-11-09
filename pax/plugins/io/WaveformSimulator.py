import numpy as np
import os
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
        self.truth_file = None
        self.config = simulation.init_config(self.config)

    def write_truth(self, event_number, peaks):
        """ Write stuff to the truth file
        """
        tff = self.config['truth_file_format']

        if tff == 'stacked_pickles':
            if self.truth_file is None:
                self.truth_file = open(self.config['truth_filename'], 'wb')
            pickle.dump({'event_number' : event_number, 'peaks' : peaks}, self.truth_file)

        elif tff == 'csv_peaklist':
            if self.truth_file is None:

                headers = [
                    'event_number', 'type',
                    'x', 'y', 'z',
                    't_interaction',
                    'n_photons', 't_mean_photons', 't_first_photon', 't_last_photon', 't_sigma_photons',
                    'n_electrons', 't_sigma_electrons', 't_mean_electrons', 't_first_electron', 't_last_electron'
                ]
                self.truth_file = csv.DictWriter(
                    f = open(self.config['truth_file_name'], 'w'),
                    fieldnames = headers,
                    lineterminator='\n'
                )
                self.truth_file.writeheader()
            for p in peaks:
                p['event_number'] = event_number
                self.truth_file.writerow(p)
        else:
            raise ValueError('Unsupported waveform simulator truth file format %s' % tff)

    def store_true_peak(self, type, t, x, y, z, photon_times, electron_times=[]):
        """ Saves the truth information about a peak (s1 or s2)
        """
        true_peak = {
            'type': type,
            'x': x, 'y': y, 'z': z,
            't_interaction':     t,
        }
        for name, times in (('photon', photon_times), ('electron', electron_times)):
            if len(times) != 0:
                true_peak.update({
                    ('n_%ss' % name):         len(times),
                    ('t_mean_%ss' % name):    np.mean(times),
                    ('t_first_%s' % name):    np.min(times),
                    ('t_last_%s' % name):     np.max(times),
                    ('t_sigma_%ss' % name):   np.std(times),
                })
            else:
                true_peak.update({
                    ('n_%ss' % name):         '',
                    ('t_mean_%ss' % name):     '',
                    ('t_first_%s' % name):    '',
                    ('t_last_%s' % name):     '',
                    ('t_sigma_%ss' % name):    '',
                })
        self.truth_peaks.append(true_peak)

    def s2(self, electrons, t=0., z=0., x=0., y=0.):
        electron_times = simulation.s2_electrons(electrons_generated=electrons, t=t, z=z)
        photon_times = simulation.s2_scintillation(electron_times)
        self.store_true_peak('s2', t, x, y, z, photon_times, electron_times)
        return simulation.hitlist_to_waveforms(
            simulation.photons_to_hitlist(photon_times),
        )

    def s1(self, photons, recoil_type, t=0., x=0., y=0., z=0.):
        """
        :param photons: total # of photons generated in the S1
        :param recoil_type: 'ER' for electronic recoil, 'NR' for nuclear recoil
        :param t: Time at which the interaction occurs, i.e. offset for arrival times. Defaults to s1_default_recombination_time
        :return: start_time, pmt_waveforms
        """
        photon_times = simulation.s1_photons(photons, recoil_type, t)
        self.store_true_peak('s1', t, x, y, z, photon_times)
        return simulation.hitlist_to_waveforms(
            simulation.photons_to_hitlist(photon_times),
        )

    def get_instructions_for_next_event(self):
        raise NotImplementedError()

    def get_events(self):

        dt = self.config['digitizer_t_resolution']
        for instruction_number, instructions in enumerate(self.get_instructions_for_next_event()):
            for repetition_i in range(self.config['event_repetitions']):
                event_number = instruction_number * self.config['event_repetitions'] + repetition_i
                self.log.debug('Instruction %s, iteration %s, event number %s' % (
                    instruction_number, repetition_i, event_number
                ))
                self.truth_peaks = []

                # Make (start_time, waveform matrix) tuples for every s1/s2 we have to generate
                signals = []
                for q in instructions:
                    self.log.debug("Simulating %s photons and %s electrons at %s cm depth, at t=%s ns" % (
                        q['s1_photons'], q['s2_electrons'], q['depth'], q['t']
                    ))
                    if int(q['s1_photons']):
                        signals.append(
                            self.s1( photons=int(q['s1_photons']), recoil_type=q['recoil_type'], t=float(q['t']) )
                        )
                    if int(q['s2_electrons']):
                        signals.append(
                            self.s2(
                                electrons=int(q['s2_electrons']),
                                z=float(q['depth']) * units.cm,
                                t=float(q['t'])
                            )
                        )

                # Remove empty signals (None) from signal list
                signals = [s for s in signals if s is not None]
                if len(signals) == 0:
                    self.log.warning(
                        "Fax simulation returned no signals, can't return an event...")
                    continue

                # Compute start time and event length in samples)
                start_time_offset = min([s[0] for s in signals])
                event_length = int(
                     2 * self.config['event_padding'] / dt +
                     max([s[0] - start_time_offset for s in signals]) / dt +
                     max([s[1].shape[1] for s in signals])
                )

                # Make a single waveform matrix
                # Baseline addition & flipping down is done here
                self.log.debug("Combining %s signals into a single matrix" % len(signals))
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

                # Setup the pax event 'header'
                self.log.debug("Creating pax event")
                event = datastructure.Event()
                if hasattr(self, 'dataset_name'):
                    event.dataset_name = self.dataset_name
                event.event_number = event_number
                event.start_time = int(time.time() * units.s)
                event.stop_time = event.start_time + int(event_length * dt)
                event.sample_duration = dt

                # Make a single occurrence for the entire event... yeah, this is wonky
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
                    for key in p.keys():
                        if key[:2] == 't_' and key[2:7] != 'sigma':
                            if p[key] == '':
                                continue
                            p[key] -= start_time_offset
                self.write_truth(event_number=event_number, peaks=self.truth_peaks)

                event_number += 1



class WaveformSimulatorFromCSV(WaveformSimulator):
    def startup(self):
        # Open the instructions file
        if 'input_override' in self.config and self.config['input_override'] is not None:
            filename = self.config['input_override']
        else:
            filename = self.config['instruction_filename']
        self.dataset_name = os.path.basename(filename)
        self.instructions_file = open(filename, 'r')
        self.instructions = csv.DictReader(self.instructions_file)
        WaveformSimulator.startup(self)

    def shutdown(self):
        self.instructions_file.close()

    def get_instructions_for_next_event(self):
        this_instruction = None
        this_instruction_peaks = []
        for p in self.instructions:
            if int(p['instruction']) == this_instruction:
                this_instruction_peaks.append(p)
            else:
                # New event reached!
                if this_instruction_peaks:
                    yield this_instruction_peaks
                this_instruction = int(p['instruction'])
                this_instruction_peaks = [p]
        # For the final event...
        yield this_instruction_peaks


class WaveformSimulatorFromNEST(WaveformSimulator):

    variables = (
             #Fax name        #Root name    #Conversion factor (multiplicative)
            ('x',             'Nest_x',     0.1),
            ('y',             'Nest_y',     0.1),
            ('depth',         'Nest_z',     -0.1),
            ('s1_photons',    'Nest_nph',   1),
            ('s2_electrons',  'Nest_nel',   1),
            ('t',             'Nest_t',     10**9),
            ('recoil_type',   'Nest_nr',    1),
    )

    def startup(self):
        self.log.warning('This plugin is completely untested and will probably crash!')
        if 'input_override' in self.config and self.config['input_override'] is not None:
            filename = self.config['input_override']
        else:
            filename = self.config['input_file']
        import ROOT
        f = ROOT.TFile(filename)
        self.t = f.Get("t1") # For Xerawdp use T1, for MC t1
        WaveformSimulator.startup(self)

    def get_instructions_for_next_event(self):
        for event_i in range(self.t.GetEntries()):
            self.t.GetEntry(event_i)

            # Get stuff from root files
            values = {}
            for (variable_name, root_thing_name, _) in self.variables:
                values[variable_name] = getattr(self.t, root_thing_name)

            # Convert to peaks dictionary
            npeaks = len(values[self.variables[0][0]])
            peaks = []
            for i in range(npeaks):
                peaks.append({'instruction' : event_i})
                for (variable_name, _, conversion_factor) in self.variables:
                    peaks[-1][variable_name] = values[variable_name][i] * conversion_factor

            for p in peaks:
                # Subtract depth of gate mesh, see xenon:xenon100:mc:roottree, bottom of page
                p['depth'] -= 2.15+0.25
                # Fix ER / NR label
                if p['recoil_type'] != 0:
                    p['recoil_type'] = 'NR'
                else:
                    p['recoil_type'] = 'ER'
            # Sort by time
            peaks.sort(key = lambda p:p['t'])

            yield peaks

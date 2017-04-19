"""
Plugins to interface with the integrated waveform simulator (FaX)
This is I/O stuff only: truth file writing, instruction reading, etc.
There is no physics here, all that is in pax.simulation.
Added by Qing: Sorry there's a little physics (S2 after pulses)
is implemented here for it is easier.
"""

import os
import csv

import numpy as np
import pandas

from pax import plugin, units, utils

try:
    import ROOT
    import root_pandas  # noqa
    have_root = True
except ImportError:
    print("You don't have ROOT or root_pandas, root truth file output is disabled")
    have_root = False


def uniform_circle_rv(radius, n_samples=None):
    """Sample n_samples from x,y uniformly in a circle with radius"""

    if n_samples is None:
        just_give_one = True
        n_samples = 1
    else:
        just_give_one = False

    xs = []
    ys = []

    for sample_i in range(n_samples):
        while True:
            x = np.random.uniform(-radius, radius)
            y = np.random.uniform(-radius, radius)
            if x**2 + y**2 <= radius**2:
                break
        xs.append(x)
        ys.append(y)

    if just_give_one:
        return xs[0], ys[0]
    else:
        return xs, ys


class WaveformSimulator(plugin.InputPlugin):
    """Common input plugin for waveform simulator plugins. Do not use directly, won't work.
    Takes care of truth file writing as well.
    """

    def startup(self):
        self.all_truth_peaks = []
        self.simulator = self.processor.simulator
        # The simulator's internal config was already intialized in the core

    def shutdown(self):
        self.log.debug("Write the truth peaks to %s" % self.config['truth_file_name'])
        output = pandas.DataFrame(self.all_truth_peaks)
        output.to_csv(self.config['truth_file_name']+".csv", index_label='fax_truth_peak_id')
        if have_root:
            output.to_root(self.config['truth_file_name']+".root", 'fax_truth')

    def store_true_peak(self, peak_type, g4_id, t, x, y, z, photon_times, electron_times=(), peak_top_fraction=0):
        """ Saves the truth information about a peak (s1 or s2)
        """
        true_peak = {
            'instruction':      self.current_instruction,
            'repetition':       self.current_repetition,
            'event':            self.current_event,
            'peak_type':        peak_type,
            'g4_id':          g4_id,
            'x': x, 'y': y, 'z': z,
            't_interaction':     t,
            'top_fraction':     peak_top_fraction,
        }
        for name, times in (('photon', photon_times), ('electron', electron_times)):
            if len(times) != 0:
                # This signal type doesn't exist in this peak
                true_peak.update({
                    ('n_%ss' % name):         len(times),
                    ('t_mean_%ss' % name):    np.mean(times),
                    ('t_first_%s' % name):    np.min(times),
                    ('t_last_%s' % name):     np.max(times),
                    ('t_sigma_%ss' % name):   np.std(times),
                })
            else:
                true_peak.update({
                    ('n_%ss' % name):         float('nan'),
                    ('t_mean_%ss' % name):    float('nan'),
                    ('t_first_%s' % name):    float('nan'),
                    ('t_last_%s' % name):     float('nan'),
                    ('t_sigma_%ss' % name):   float('nan'),
                })
        self.truth_peaks.append(true_peak)

    def s2(self, electrons, g4_id=-1, t=0., x=0., y=0., z=0.):
        r = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)

        electron_times = self.simulator.s2_electrons(electrons_generated=electrons, t=t, z=z, r=r)
        if not len(electron_times):
            return None
        photon_times = self.simulator.s2_scintillation(electron_times, x, y)
        if not len(photon_times):
            return None

        # Compute the xy for the S2 using the radial distortion map
        dmap = self.simulator.rz_position_distortion_map
        if dmap:
            r += dmap.get_value(r, z, map_name='to_distorted_r')
            x = r * np.cos(phi)
            y = r * np.sin(phi)

        arriving_photon_times_top,  arriving_photon_times_bottom = self.simulator.queue_signal(
            photon_times,
            x,
            y,
            # Generate S2 hitpattern "at the anode": cue for simulator to use S2 LCE map
            z=-self.config['gate_to_anode_distance'])
        peak_top_fraction = float(len(arriving_photon_times_top)) / float(
            len(arriving_photon_times_bottom) + len(arriving_photon_times_top))
        self.store_true_peak(
            's2',
            g4_id,
            t,
            x,
            y,
            z,
            np.concatenate((
                arriving_photon_times_top,
                arriving_photon_times_bottom
                )
            ),
            electron_times,
            peak_top_fraction,
            )
        return None

    def s1(self, photons, recoil_type, g4_id=-1, t=0., x=0., y=0., z=0.):
        """
        :param photons: total # of photons generated in the S1
        :param recoil_type: 'ER' for electronic recoil, 'NR' for nuclear recoil
        :param t: Time at which the interaction occurs, i.e. offset for arrival times.
                Defaults to s1_default_recombination_time.
        :return: start_time, channel_waveforms
        """
        photon_times = self.simulator.s1_photons(photons, recoil_type, x, y, z, t)
        if not len(photon_times):
            return None
        arriving_photon_times_top, arriving_photon_times_bottom = self.simulator.queue_signal(
                                                                                               photon_times,
                                                                                               x=x,
                                                                                               y=y,
                                                                                               z=z
                                                                                              )
        peak_top_fraction = float(len(arriving_photon_times_top)) / float(
            len(arriving_photon_times_bottom) + len(arriving_photon_times_top))
        self.store_true_peak(
            's1',
            g4_id,
            t,
            x,
            y,
            z,
            np.concatenate((
                arriving_photon_times_top,
                arriving_photon_times_bottom
                )),
            peak_top_fraction=peak_top_fraction,
            )
        return None

    def s2_after_pulses(self, g4_id=-1):
        """
        :simulate the s2 after pulses
        :the after pulses are assumed to be uniformly distributed in X-Y,
        :so are not related to where the photon is generated
        :We simplify the generation, assuming S2-after pulses
        :will not further generate secondary s2 after pulses
        """
        photon_detection_times = []
        # join all the photon detection times in channels
        for channel_id, single_channel_photon_detection_times \
                in self.simulator.arrival_times_per_channel.items():
            photon_detection_times.extend(
                np.array(single_channel_photon_detection_times)
                )

        # generate the s2 after pulses for each type
        s2_ap_electron_times = []
        for s2_ap_data in self.config['s2_afterpulse_types'].values():
            # calculate how many s2 after pulses
            num_s2_afterpulses = np.random.binomial(
                n=len(photon_detection_times),
                p=s2_ap_data['p']
                )
            if num_s2_afterpulses == 0:
                continue

            # Find the time delay of the after pulses
            dist_kwargs = s2_ap_data['time_parameters']
            dist_kwargs['size'] = num_s2_afterpulses
            delay = getattr(np.random, s2_ap_data['time_distribution'])(**dist_kwargs)

            # Delete afterpulses coming after the maximum delay time (if this option has been enabled)
            delay = delay[delay < self.config.get('max_s2_afterpulse_delay', float('inf'))]
            num_s2_afterpulses = len(delay)

            # Choose the original photons that generated the S2 afterpulse.
            # In fact is it is not a detected photon, but an undetected one, that generated the afterpulse...
            # but we only know the detected photon times anymore at this stage.
            original_photon_times = np.random.choice(photon_detection_times,
                                                     size=num_s2_afterpulses,
                                                     replace=False)
            s2_ap_electron_times.extend(original_photon_times + delay)

        # generate the s2 photons of each s2 pulses one by one
        # the X-Y of after pulse is randomized, Z is set to 0
        # randomize an array of X-Y
        rsquare = np.random.uniform(
            0,
            np.power(self.config['tpc_radius'], 2.),
            len(s2_ap_electron_times)
            )
        theta = np.random.uniform(0, 3.141592653, len(s2_ap_electron_times))
        X = np.sqrt(rsquare)*np.cos(theta)
        Y = np.sqrt(rsquare)*np.sin(theta)
        for electron_id, s2_ap_electron_time \
                in enumerate(s2_ap_electron_times):
            s2_ap_photon_times = self.simulator.s2_scintillation(
                [s2_ap_electron_time],
                X[electron_id],
                Y[electron_id]
                )
            # queue the photons caused by the s2 after pulses
            arriving_photon_times_top, arriving_photon_times_bottom = self.simulator.queue_signal(
                s2_ap_photon_times,
                X[electron_id],
                Y[electron_id],
                -self.config['gate_to_anode_distance']
                )
            peak_top_fraction = float(len(arriving_photon_times_top)) / float(
                len(arriving_photon_times_bottom) + len(arriving_photon_times_top))
            self.store_true_peak(
                'photoionization_afterpulse',
                g4_id,
                t=s2_ap_electron_time,
                x=X[electron_id],
                y=Y[electron_id],
                z=-self.config['gate_to_anode_distance'],
                photon_times=np.concatenate((
                    arriving_photon_times_top,
                    arriving_photon_times_bottom
                    )),
                peak_top_fraction=peak_top_fraction,
                )

    def get_instructions_for_next_event(self):
        raise NotImplementedError()

    def simulate_single_event(self, instructions):
        self.truth_peaks = []

        for q in instructions:
            self.log.debug("Simulating %s photons and %s electrons at %s cm z, at t=%s ns" % (
                q['s1_photons'], q['s2_electrons'], q['z'], q['t']))

            # Should we choose x and yrandomly?
            if q['x'] == 'random':
                x, y = uniform_circle_rv(self.config['tpc_radius'])
            else:
                x = float(q['x'])
                y = float(q['y'])

            if q['z'] == 'random':
                z = - np.random.uniform(0, self.config['tpc_length'])
            else:
                z = float(q['z']) * units.cm

            if int(q['s1_photons']):
                self.s1(photons=int(q['s1_photons']),
                        recoil_type=q['recoil_type'],
                        g4_id=q['g4_id'],
                        t=float(q['t']), x=x, y=y, z=z)

            if int(q['s2_electrons']):
                self.s2(electrons=int(q['s2_electrons']),
                        g4_id=q['g4_id'],
                        t=float(q['t']), x=x, y=y, z=z)

        # Based on the generated photon timing
        # generate the after pulse
        # currently make it simple, assuming s2 after pulses
        # will not generate further s2 after pulses.
        # If there were no instructions, there is no g4_id, but there will also be no photons,
        # so running this is pointless.
        if len(instructions):
            self.s2_after_pulses(g4_id=q['g4_id'])

        event = self.simulator.make_pax_event()
        if hasattr(self, 'dataset_name'):
            event.dataset_name = self.dataset_name
        event.event_number = self.current_event

        # Add start time offset to all peak start times in the truth file
        # Can't be done at the time of peak creation, it is only known now...
        # TODO: That's no longer true! so fix it
        for p in self.truth_peaks:
            for key in p.keys():
                if key[:2] == 't_' and key[2:7] != 'sigma':
                    if p[key] == '':
                        continue
                    p[key] += self.config['event_padding']
        self.all_truth_peaks.extend(self.truth_peaks)

        return event

    def get_events(self):
        for instruction_number, instructions in enumerate(self.get_instructions_for_next_event()):
            self.current_instruction = instruction_number
            for repetition_i in range(self.config['event_repetitions']):
                self.current_repetition = repetition_i
                self.current_event = instruction_number * self.config['event_repetitions'] + repetition_i
                self.log.debug('Instruction %s, iteration %s, event number %s' % (instruction_number,
                                                                                  repetition_i, self.current_event))
                yield self.simulate_single_event(instructions)


class WaveformSimulatorFromCSV(WaveformSimulator):
    """Simulate waveforms from a csv file with instructions, see:
        http://xenon1t.github.io/pax/simulator.html#instruction-file-format
    """

    def startup(self):
        """
        The startup routine of the WaveformSimulatorFromCSV plugin
        """

        # Open the instructions file
        filename = self.config['input_name']
        self.dataset_name = os.path.basename(filename)
        self.instructions_file = open(utils.data_file_name(filename), 'r')
        #
        # Slurp the entire instructions file, so we know the number of events
        self.instruction_reader = csv.DictReader(self.instructions_file)
        self.instructions = []
        #
        # Loop over lines, make instructions
        instruction_number = 0
        instruction = []
        for p in self.instruction_reader:
            p['g4_id'] = -1  # create fake g4_id=-1 for csv input
            if p['depth'] == 'random':
                p['z'] = 'random'
            else:
                p['z'] = -1 * float(p['depth'])
            del p['depth']
            if int(p['instruction']) == instruction_number:
                # Deposition is part of the previous instruction
                instruction.append(p)
            else:
                # New deposition reached!
                if instruction:
                    self.instructions.append(instruction)
                instruction_number = int(p['instruction'])
                instruction = [p]
        # For the final instruction
        self.instructions.append(instruction)

        self.number_of_events = len(self.instructions) * self.config['event_repetitions']
        WaveformSimulator.startup(self)

    def shutdown(self):
        self.instructions_file.close()
        WaveformSimulator.shutdown(self)

    def get_instructions_for_next_event(self):
        for instr in self.instructions:
            yield instr


class WaveformSimulatorFromMC(WaveformSimulator):
    """Simulate waveforms from GEANT4 ROOT file with generated
    S1, S2 information (e.g. from NEST, nSort, or other)
    """

    variables = (
        # Fax name        #Root name    #Conversion factor (multiplicative)
        ('x',             'Fax_x',     0.1),
        ('y',             'Fax_y',     0.1),
        ('z',             'Fax_z',     0.1),
        ('s1_photons',    'Fax_nph',   1),
        ('s2_electrons',  'Fax_nel',   1),
        ('t',             'Fax_t',     10 ** 9),
        ('recoil_type',   'Fax_nr',    1),
    )

    def startup(self):
        if not have_root:
            raise RuntimeError("Can't read MC ROOT files if you do not have root!")

        self.config.setdefault('add_to_z', 0)
        self.log.warning('This plugin is completely untested and will probably crash!')
        filename = self.config['input_name']

        self.f = ROOT.TFile(utils.data_file_name(filename))
        self.t = self.f.Get("events/events")  # new MC structure, 160622
        WaveformSimulator.startup(self)
        self.number_of_events = self.t.GetEntries() * self.config['event_repetitions']

    def get_instructions_for_next_event(self):
        for event_i in range(self.number_of_events):
            self.t.GetEntry(event_i)

            # Get stuff from root files
            values = {}
            for (variable_name, root_thing_name, _) in self.variables:
                values[variable_name] = getattr(self.t, root_thing_name)

            # Convert to peaks dictionary
            npeaks = len(values[self.variables[0][0]])
            interaction_instructions = []
            for i in range(npeaks):
                interaction_instructions.append({'instruction': event_i})
                interaction_instructions[-1]['g4_id'] = self.t.eventid
                for (variable_name, _, conversion_factor) in self.variables:
                    interaction_instructions[-1][variable_name] = values[variable_name][i] * conversion_factor

            for p in interaction_instructions:
                # Correct the z-coordinate system
                p['z'] += self.config['add_to_z']
                # Fix ER / NR / alpha label
                if p['recoil_type'] == 1:
                    p['recoil_type'] = 'NR'
                elif p['recoil_type'] == 0:
                    p['recoil_type'] = 'ER'
                elif p['recoil_type'] == 9:
                    p['recoil_type'] = 'alpha'
                else:
                    raise ValueError("Type of events unknown!")
            # Sort by time
            interaction_instructions.sort(key=lambda p: p['t'])

            yield interaction_instructions


class WaveformSimulatorFromOpticalGEANT(WaveformSimulator):
    """Simulate waveforms from Fabio's GEANT4 optical simulation
    See:
        https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon1t:fvm:waveform_generator
    Currently just pseudocode, awaiting full implementation.
    The truth file produced by this input plugin will be empty (as WaveformSimulator.s1 and .s2 never get called)
    """

    variables = (
        # Fax name        #Root name    #Conversion factor (multiplicative)
        ('photon_arriving_times',             'pmthitTime',     1000000000.),
        ('photon_hit_pmt_ids',             'pmthitID',     1),
    )

    def startup(self):
        if not have_root:
            raise RuntimeError("Can't read MC ROOT files if you do not have root!")

        self.f = ROOT.TFile(self.config['input_name'])
        if not self.f.IsOpen():
            raise ValueError(
                "Cannot open ROOT file %s" % self.config['input_name']
                )
        self.t = self.f.Get("events/events")
        WaveformSimulator.startup(self)
        self.number_of_events = self.t.GetEntries() * self.config['event_repetitions']

    def get_instructions_for_next_event(self):
        for event_i in range(self.number_of_events):
            self.t.GetEntry(event_i)
            instructions = {}
            instructions['instruction'] = event_i
            instructions['g4_id'] = self.t.eventid
            # fill instructions to look like this:
            # {[photon_hit_pmt_ids], [photon_arriving_times]}
            for (
                    variable_name, root_thing_name,
                    conversion_factor
                    ) in self.variables:
                # get stuff from root
                # the two variables in this plugin are all vectors
                values = np.reshape(
                    getattr(self.t, root_thing_name),
                    self.t.npmthits
                    )
                conversion_factors = [conversion_factor]*self.t.npmthits
                instructions[variable_name] = [
                    x*y for x, y in zip(values, conversion_factors)
                    ]
            yield instructions

    def simulate_single_event(self, instructions):
        self.simulator.clear_signals_queue()
        for (channel, arr_time) in zip(
                instructions['photon_hit_pmt_ids'],
                instructions['photon_arriving_times']
                ):
            self.simulator.arrival_times_per_channel[channel].append(arr_time)
        self.s2_after_pulses()
        event = self.simulator.make_pax_event()
        event.event_number = self.current_event
        return event

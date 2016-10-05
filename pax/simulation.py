"""
Waveform simulator ("FaX") - physics backend
The only I/O stuff here is pax event creation, everything else is in the WaveformSimulator plugins
"""

from __future__ import division
import logging
import math
import time

import numpy as np

from scipy import stats

from pax import units, utils, datastructure
from pax.PatternFitter import PatternFitter
from pax.InterpolatingMap import InterpolatingMap
from pax.utils import Memoize

log = logging.getLogger('SimulationCore')


class Simulator(object):

    def __init__(self, config_to_init):
        c = self.config = config_to_init

        # Should we repeat events?
        if 'event_repetitions' not in c:
            c['event_repetitions'] = 1

        # Primary excimer fraction from Nest Version 098
        # See G4S1Light.cc line 298
        density = c['liquid_density'] / (units.g / units.cm ** 3)
        excfrac = 0.4 - 0.11131 * density - 0.0026651 * density ** 2    # primary / secondary excimers
        excfrac = 1 / (1 + excfrac)                                     # primary / all excimers
        # primary / all excimers that produce a photon:
        excfrac /= 1 - (1 - excfrac) * (1 - c['s1_ER_recombination_fraction'])
        c['s1_ER_primary_excimer_fraction'] = excfrac
        log.debug('Inferred s1_ER_primary_excimer_fraction %s' % excfrac)

        # Recombination time from NEST 2014
        # 3.5 seems fishy, they fit an exponential to data, but in the code they use a non-exponential distribution...
        efield = (c['drift_field'] / (units.V / units.cm))
        c['s1_ER_recombination_time'] = 3.5 / 0.18 * (1 / 20 + 0.41) * math.exp(-0.009 * efield)
        log.debug('Inferred s1_ER_recombination_time %s ns' % c['s1_ER_recombination_time'])

        # Calculate particle number density in the gas (ideal gas law)
        number_density_gas = c['pressure'] / (units.boltzmannConstant * c['temperature'])

        # electric field in the gas
        # Formula from xenon:xenon100:analysis:jacob:s2gain_v2
        e_in_gas = c['lxe_dielectric_constant'] * c['anode_voltage'] / (
            c['lxe_dielectric_constant'] * c['elr_gas_gap_length'] +
            (c['gate_to_anode_distance'] - c['elr_gas_gap_length'])
        )

        # Reduced electric field in the gas
        c['reduced_e_in_gas'] = e_in_gas / number_density_gas
        log.debug("Inferred a reduced electric field of %s Td in the gas" % (
            c['reduced_e_in_gas'] / units.Td))

        # Which channels stand to receive any photons?
        channels_for_photons = c['channels_in_detector']['tpc']
        if c['pmt_0_is_fake']:
            channels_for_photons = [ch for ch in channels_for_photons if ch != 0]
        if c.get('magically_avoid_dead_pmts', False):
            channels_for_photons = [ch for ch in channels_for_photons if c['gains'][ch] > 0]
        if c.get('magically_avoid_s1_excluded_pmts', False) and \
           'channels_excluded_for_s1' in c:
            channels_for_photons = [ch for ch in channels_for_photons
                                    if ch not in c['channels_excluded_for_s1']]
        c['channels_for_photons'] = channels_for_photons

        # Determine sensible length of a pmt pulse to simulate
        dt = c['sample_duration']
        c['samples_before_pulse_center'] = math.ceil(
            c['pulse_width_cutoff'] * c['pmt_rise_time'] / dt
        )
        c['samples_after_pulse_center'] = math.ceil(
            c['pulse_width_cutoff'] * c['pmt_fall_time'] / dt
        )
        log.debug('Simulating %s samples before and %s samples after PMT pulse centers.' % (
            c['samples_before_pulse_center'], c['samples_after_pulse_center']))

        # Load real noise data from file, if requested
        if c['real_noise_file']:
            self.noise_data = np.load(utils.data_file_name(c['real_noise_file']))['arr_0']
            # The silly XENON100 PMT offset again: it's relevant for indexing the array of noise data
            # (which is one row per channel)
            self.channel_offset = 1 if c['pmt_0_is_fake'] else 0

        # Load light yields
        self.s1_light_yield_map = InterpolatingMap(utils.data_file_name(c['s1_light_yield_map']))
        self.s2_light_yield_map = InterpolatingMap(utils.data_file_name(c['s2_light_yield_map']))

        # Init s2 per pmt lce map
        qes = np.array(c['quantum_efficiencies'])
        if c.get('s2_patterns_file', None) is not None:
            self.s2_patterns = PatternFitter(filename=utils.data_file_name(c['s2_patterns_file']),
                                             zoom_factor=c.get('s2_patterns_zoom_factor', 1),
                                             adjust_to_qe=qes[c['channels_top']],
                                             default_errors=c['relative_qe_error'] + c['relative_gain_error'])
        else:
            self.s2_patterns = None

        # Init s1 pattern maps
        # NB: do NOT adjust patterns for QE, map is data derived, so no need.
        log.debug("Initializing s1 patterns...")
        if c.get('s1_patterns_file', None) is not None:
            self.s1_patterns = PatternFitter(filename=utils.data_file_name(c['s1_patterns_file']),
                                             zoom_factor=c.get('s1_patterns_zoom_factor', 1),
                                             default_errors=c['relative_qe_error'] + c['relative_gain_error'])
        else:
            self.s1_patterns = None

        self.clear_signals_queue()

    def clear_signals_queue(self):
        """Prepares the waveform simulator for a new event.
        """
        self.arrival_times_per_channel = {ch: [] for ch in range(self.config['n_channels'])}

    def queue_signal(self, photon_timings, x=0, y=0, z=0):
        """Add a signal due to isotropic light emission to the waveform simulator
            photon_timings: list of photon emission times (ns) since start of the event.
                            All photons listed here will be detected!
            x, y, z: position of emission (standard coordinate system, pax units = cm)
        """
        if not len(photon_timings):
            return

        # Correct for PMT Transition Time Spread
        photon_timings += np.random.normal(self.config['pmt_transit_time_mean'],
                                           self.config['pmt_transit_time_spread'],
                                           len(photon_timings))

        # Shuffle all timings in the array, so channel 1 doesn't always get the first photon
        np.random.shuffle(photon_timings)

        # Get the photon counts per channel
        hitp = self.distribute_photons(len(photon_timings), x, y, z)

        # Split photon times over channels, add to the currently queued photon signals
        # Note the last array is always zero  # TODO: did you check this
        q = np.split(photon_timings, np.cumsum(hitp))
        assert len(q[-1]) == 0
        q = q[:-1]
        for channel_i, photon_times in enumerate(q):
            self.arrival_times_per_channel[channel_i] = np.concatenate((self.arrival_times_per_channel[channel_i],
                                                                        photon_times))

    def make_pax_event(self):
        """Simulate PMT response to the queued photon signals
        Returns None if no photons have been queued else returns start_time (in units, ie ns), pmt waveform matrix
        # TODO: Account for random initial digitizer state wrt interaction? Where?
        """
        log.debug("Now performing hitpattern to waveform conversion")
        start_time = int(time.time() * units.s)

        # Find out the duration of the event
        all_times = np.concatenate(list(self.arrival_times_per_channel.values()))
        if not len(all_times):
            log.warning("No photons to simulate: making a noise-only event")
            max_time = 0
        else:
            max_time = np.concatenate(list(self.arrival_times_per_channel.values())).max()

        event = datastructure.Event(n_channels=self.config['n_channels'],
                                    start_time=start_time,
                                    stop_time=start_time + int(max_time + 2 * self.config['event_padding']),
                                    sample_duration=self.config['sample_duration'])
        # Ensure the event length is even (else it cannot be written to XED)
        if event.length() % 2 != 0:
            event.stop_time += self.config['sample_duration']

        # Convenience variables
        dt = self.config['sample_duration']
        dv = self.config['digitizer_voltage_range'] / 2 ** (self.config['digitizer_bits'])

        # Build waveform channel by channel
        for channel, photon_detection_times in self.arrival_times_per_channel.items():
            # If the channel is dead, we don't do anything.
            if self.config['gains'][channel] == 0 or (self.config['pmt_0_is_fake'] and channel == 0):
                continue

            photon_detection_times = np.array(photon_detection_times)

            log.debug("Simulating %d photons in channel %d (gain=%s, gain_sigma=%s)" % (
                len(photon_detection_times), channel,
                self.config['gains'][channel], self.config['gain_sigmas'][channel]))

            # Use a Gaussian truncated to positive values for the SPE gain distribution
            gains = truncated_gauss_rvs(my_mean=self.config['gains'][channel],
                                        my_std=self.config['gain_sigmas'][channel],
                                        left_boundary=0,
                                        right_boundary=float('inf'),
                                        n_rvs=len(photon_detection_times))

            # Add PMT afterpulses
            ap_times = []
            ap_gains = []
            for ap_data in self.config['pmt_afterpulse_types'].values():
                ap_data.setdefault('gain_mean', self.config['gains'][channel])
                ap_data.setdefault('gain_rms', self.config['gain_sigmas'][channel])

                # How many photons will make this kind of afterpulse?
                n_afterpulses = np.random.binomial(n=len(photon_detection_times),
                                                   p=ap_data['p'])
                if not n_afterpulses:
                    continue

                # Find the time and gain of the afterpulses
                dist_kwargs = ap_data['time_parameters']
                dist_kwargs['size'] = n_afterpulses
                ap_times.extend(np.random.choice(photon_detection_times, size=n_afterpulses, replace=False) +
                                getattr(np.random, ap_data['time_distribution'])(**dist_kwargs))
                ap_gains.extend(truncated_gauss_rvs(my_mean=ap_data['gain_mean'],
                                                    my_std=ap_data['gain_rms'],
                                                    left_boundary=0,
                                                    right_boundary=float('inf'),
                                                    n_rvs=n_afterpulses))

            gains = np.concatenate((gains, ap_gains))
            photon_detection_times = np.concatenate((photon_detection_times, ap_times))

            #  Add padding, sort (eh.. or were we already sorted? and is sorting necessary at all??)
            pmt_pulse_centers = np.sort(photon_detection_times + self.config['event_padding'])

            # Build the waveform pulse by pulse (bin by bin was slow, hope this is faster)

            # Compute offset & center index for each pe-pulse
            # 'index' refers to the (hypothetical) event waveform, as usual
            pmt_pulse_centers = np.array(pmt_pulse_centers, dtype=np.int)
            offsets = pmt_pulse_centers % dt
            center_index = (pmt_pulse_centers - offsets) / dt   # Absolute index in waveform of pe-pulse center
            center_index = center_index.astype(np.int)

            # Simulate an event-long waveform in this channel
            # Remember start padding has already been added to times, so just one padding in end_index
            start_index = 0
            end_index = event.length() - 1
            pulse_length = end_index - start_index + 1

            current_wave = np.zeros(pulse_length)

            for i, _ in enumerate(pmt_pulse_centers):
                # Add some current for this photon pulse
                # Compute the integrated pmt pulse at various samples, then
                # do their diffs/dt
                generated_pulse = self.pmt_pulse_current(gain=gains[i], offset=offsets[i])

                # +1 due to np.diff in pmt_pulse_current   #????
                left_index = center_index[i] - start_index + 1
                left_index -= int(self.config['samples_before_pulse_center'])
                righter_index = center_index[i] - start_index + 1
                righter_index += int(self.config['samples_after_pulse_center'])

                # Abandon the pulse if it goes the left/right boundaries
                if len(generated_pulse) != righter_index - left_index:
                    raise RuntimeError(
                        "Generated pulse is %s samples long, can't be inserted between %s and %s" % (
                            len(generated_pulse), left_index, righter_index))
                elif left_index < 0:
                    log.debug("Invalid left index %s: can't be negative" % left_index)
                    continue
                elif righter_index >= len(current_wave):
                    log.debug("Invalid right index %s: can't be longer than length of wave (%s)!" % (
                        righter_index, len(current_wave)))
                    continue

                current_wave[left_index: righter_index] += generated_pulse

            # Did you order some Gaussian current noise with that?
            if self.config['gauss_noise_sigmas']:
                # if the baseline fluc. is defined for each channel
                # use that in prior
                noise_sigma_current = self.config['gauss_noise_sigmas'][channel]*self.config['gains'][channel] / dt
                current_wave += np.random.normal(0, noise_sigma_current, len(current_wave))
            elif self.config['gauss_noise_sigma']:
                # / dt is for charge -> current conversion, as in pmt_pulse_current
                noise_sigma_current = self.config['gauss_noise_sigma'] * self.config['gains'][channel] / dt,
                current_wave += np.random.normal(0, noise_sigma_current, len(current_wave))

            # Convert from PMT current to ADC counts
            adc_wave = current_wave
            adc_wave *= self.config['pmt_circuit_load_resistor']    # Now in voltage
            adc_wave *= self.config['external_amplification']       # Now in voltage after amplifier
            adc_wave /= dv                                          # Now in float ADC counts above baseline
            adc_wave = np.trunc(adc_wave)                           # Now in integer ADC counts "" ""
            # Could round instead of trunc... who cares?

            # PMT signals are negative excursions, so flip them.
            adc_wave = - adc_wave

            # Did you want to superpose onto real noise samples?
            if self.config['real_noise_file']:
                sample_size = self.config['real_noise_sample_size']
                available_noise_samples = self.noise_data.shape[1] / sample_size
                needed_noise_samples = int(math.ceil(pulse_length / sample_size))
                chosen_noise_sample_numbers = np.random.randint(0,
                                                                available_noise_samples - 1,
                                                                needed_noise_samples)
                # Extract the chosen noise samples and concatenate them
                # Have to use a listcomp here, unless you know a way to select multiple slices in numpy?
                #  -- yeah making an index list with np.arange would work, but honestly??
                real_noise = np.concatenate([
                    self.noise_data[channel - self.channel_offset][nsn * sample_size:(nsn + 1) * sample_size]
                    for nsn in chosen_noise_sample_numbers
                ])
                # Adjust the noise amplitude if needed, then add it to the ADC wave
                noise_amplitude = self.config.get('adjust_noise_amplitude', {}).get(str(channel), 1)
                if noise_amplitude != 1:
                    # Determine a rough baseline for the noise, then adjust towards it
                    baseline = np.mean(real_noise[:min(len(real_noise), 50)])
                    real_noise = baseline + noise_amplitude * (real_noise - baseline)
                adc_wave += real_noise[:pulse_length]

            else:
                # If you don't want to superpose onto real noise,
                # we should add a reference baseline
                adc_wave += self.config['digitizer_reference_baseline']

            # Digitizers have finite number of bits per channel, so clip the signal.
            adc_wave = np.clip(adc_wave, 0, 2 ** (self.config['digitizer_bits']))

            event.pulses.append(datastructure.Pulse(
                channel=channel,
                left=start_index,
                raw_data=adc_wave.astype(np.int16)))

        log.debug("Simulated pax event of %s samples length and %s pulses "
                  "created." % (event.length(), len(event.pulses)))
        self.clear_signals_queue()
        return event

    def s2_electrons(self, electrons_generated=None, z=0., t=0.):
        """Return a list of electron arrival times in the ELR region caused by an S2 process.

            electrons             -   total # of drift electrons generated at the interaction site
            t                     -   Time at which the original energy deposition occurred.
            z                     -   Depth below the GATE mesh where the interaction occurs.
        As usual, all units in the same system used by pax (if you specify raw values: ns, cm)
        """

        if not - self.config['tpc_length'] <= z <= 0:
            log.warning("Unphysical depth: %s cm below gate. Not generating S2." % - z)
            return []
        log.debug("Creating an s2 from %s electrons..." % electrons_generated)

        # Average drift time, taking faster drift velocity after gate into account
        drift_time_mean = - z / self.config['drift_velocity_liquid'] + \
            (self.config['gate_to_anode_distance'] - self.config['elr_gas_gap_length']) \
            / self.config['drift_velocity_liquid_above_gate']

        # Diffusion model from Sorensen 2011
        drift_time_stdev = math.sqrt(2 * self.config['diffusion_constant_liquid'] * drift_time_mean)
        drift_time_stdev /= self.config['drift_velocity_liquid']

        # Absorb electrons during the drift
        electron_lifetime_correction = -1 * drift_time_mean / self.config['electron_lifetime_liquid']
        electron_lifetime_correction = math.exp(electron_lifetime_correction)
        prob = self.config['electron_extraction_yield'] * electron_lifetime_correction

        electrons_seen = np.random.binomial(n=electrons_generated,
                                            p=prob)

        log.debug("    %s electrons survive the drift." % electrons_generated)

        # Calculate electron arrival times in the ELR region
        e_arrival_times = t + np.random.exponential(self.config['electron_trapping_time'], electrons_seen)
        if drift_time_stdev:
            e_arrival_times += np.random.normal(drift_time_mean, drift_time_stdev, electrons_seen)
        return e_arrival_times

    def s1_photons(self, n_photons, recoil_type, x=0., y=0., z=0, t=0.):
        """Returns a list of photon detection times at the PMT caused by an S1 emitting n_photons.
        """
        # Apply light yield / detection efficiency
        log.debug("Creating an s1 from %s photons..." % n_photons)
        ly = self.s1_light_yield_map.get_value(x, y, z) * self.config['s1_detection_efficiency']
        n_photons = np.random.binomial(n=n_photons, p=ly)
        log.debug("    %s photons are detected." % n_photons)
        if n_photons == 0:
            return np.array([])

        if recoil_type.lower() == 'er':

            # How many of these are primary excimers? Others arise through recombination.
            n_primaries = np.random.binomial(n=n_photons, p=self.config['s1_ER_primary_excimer_fraction'])

            primary_timings = self.singlet_triplet_delays(
                np.zeros(n_primaries),  # No recombination delay for primary excimers
                t1=self.config['singlet_lifetime_liquid'],
                t3=self.config['triplet_lifetime_liquid'],
                singlet_ratio=self.config['s1_ER_primary_singlet_fraction']
            )

            # Correct for the recombination time
            # For the non-exponential distribution: see Kubota 1979, solve eqn 2 for n/n0.
            # Alternatively, see Nest V098 source code G4S1Light.cc line 948
            secondary_timings = self.config['s1_ER_recombination_time']\
                * (-1 + 1 / np.random.uniform(0, 1, n_photons - n_primaries))
            secondary_timings = np.clip(secondary_timings, 0, self.config['maximum_recombination_time'])
            # Handle singlet/ triplet decays as before
            secondary_timings += self.singlet_triplet_delays(
                secondary_timings,
                t1=self.config['singlet_lifetime_liquid'],
                t3=self.config['triplet_lifetime_liquid'],
                singlet_ratio=self.config['s1_ER_secondary_singlet_fraction']
            )

            timings = np.concatenate((primary_timings, secondary_timings))

        elif recoil_type.lower() == 'nr':

            # Neglible recombination time, same singlet/triplet ratio for primary & secondary excimers
            # Hence, we don't care about primary & secondary excimers at all:
            timings = self.singlet_triplet_delays(
                np.zeros(n_photons),
                t1=self.config['singlet_lifetime_liquid'],
                t3=self.config['triplet_lifetime_liquid'],
                singlet_ratio=self.config['s1_NR_singlet_fraction']
            )

        elif recoil_type.lower() == 'alpha':

            # again neglible recombination time, same singlet/triplet ratio for primary & secondary excimers
            # Hence, we don't care about primary & secondary excimers at all:
            timings = self.singlet_triplet_delays(
                np.zeros(n_photons),
                t1=self.config['alpha_singlet_lifetime_liquid'],
                t3=self.config['alpha_triplet_lifetime_liquid'],
                singlet_ratio=self.config['s1_ER_alpha_singlet_fraction']
            )

        elif recoil_type.lower() == 'led':

            # distribute photons uniformly within the LED pulse length
            timings = np.random.uniform(0, self.config['led_pulse_length'],
                                        size=n_photons)

        else:
            raise ValueError('Recoil type must be ER, NR, alpha or LED, not %s' % type)

        return timings + t * np.ones(len(timings))

    def s2_scintillation(self, electron_arrival_times, x=0.0, y=0.0):
        """Given a list of electron arrival times, returns photon production times"""
        # How many photons does each electron make?
        c = self.config
        photons_produced = np.random.poisson(
            c['s2_secondary_sc_gain_density'] * c['elr_gas_gap_length'] * self.s2_light_yield_map.get_value(x, y),
            len(electron_arrival_times)
        )
        total_photons = np.sum(photons_produced)
        log.debug("    %s scintillation photons will be detected." % total_photons)
        if total_photons == 0:
            return np.array([])

        # Find the photon production times
        # Assume luminescence probability ~ electric field
        s2_pe_times = np.concatenate([
            t0 + self.get_luminescence_times(photons_produced[i])
            for i, t0 in enumerate(electron_arrival_times)
        ])

        # Account for singlet/triplet excimer decay times
        return self.singlet_triplet_delays(
            s2_pe_times,
            t1=c['singlet_lifetime_gas'],
            t3=c['triplet_lifetime_gas'],
            singlet_ratio=c['singlet_fraction_gas']
        )

    def singlet_triplet_delays(self, times, t1, t3, singlet_ratio):
        """
        Given a list of eximer formation times, returns excimer decay times.
            t1            - singlet state lifetime
            t3            - triplet state lifetime
            singlet_ratio - fraction of excimers that become singlets
                            (NOT the ratio of singlets/triplets!)
        """
        n_singlets = np.random.binomial(n=len(times), p=singlet_ratio)
        return times + np.concatenate([
            np.random.exponential(t1, n_singlets),
            np.random.exponential(t3, len(times) - n_singlets)
        ])

    def get_luminescence_times(self, n):

        dg = self.config['elr_gas_gap_length']

        # Distance between liquid level and uniform -> line field crossover point
        du = dg - self.config['anode_field_domination_distance']

        # Distance between liquid level and anode wire
        dw = dg - self.config['anode_wire_radius']

        # How many photons are produced in the uniform part?
        n_uniform = np.random.binomial(n, 1 / (1 + (dg - du) / du *
                                               math.log((dg - du) / (dg - dw))))

        # Sample the luminescence times in the uniform part
        pos_uniform = np.random.uniform(0, du, n_uniform)

        # Sample the luminescence positions in the non-uniform part
        FTilde = np.random.uniform(0, 1, n - n_uniform)
        pos_non_uniform = dg - (dg - du) ** (1 - FTilde) * (dg - dw) ** FTilde

        # Convert to luminescence times
        result = np.concatenate((
            pos_uniform,
            pos_non_uniform
            # To take electron speedup near anode into account, replace line above with:
            # (- du ** 2 + 2 * dg * pos_non_uniform - pos_non_uniform**2) / (2 * (dg - du))
            # NB: does not take electron bending towards anode into account, so probably worse!
        ))
        result *= 1/(self.config['gas_drift_velocity_slope'] * self.config['reduced_e_in_gas'])

        return result

    def pmt_pulse_current(self, gain, offset=0):
        # Rounds offset to nearest pmt_pulse_time_rounding so we can exploit caching
        return gain * pmt_pulse_current_raw(
            self.config['pmt_pulse_time_rounding'] * round(offset / self.config['pmt_pulse_time_rounding']),
            self.config['sample_duration'],
            self.config['samples_before_pulse_center'],
            self.config['samples_after_pulse_center'],
            self.config['pmt_rise_time'],
            self.config['pmt_fall_time'],
        )

    def distribute_photons(self, n_photons, x, y, z):
        """Distribute n_photons over the TPC PMTs, with LCE appropriate to (x, y, z)
        :return: numpy array of length == sim.config['n_channels'] with photon count per channel
        """
        if z == - self.config['gate_to_anode_distance']:
            # Use the S2 pattern information
            if not self.s2_patterns:
                return self.randomize_photons_over_channels(n_photons, self.config['channels_in_detector']['tpc'])

            # How many photons to the top array?
            n_top = np.random.binomial(n=n_photons, p=self.config['s2_mean_area_fraction_top'])

            # Distribute a fraction of the top photons randomly, if the user asked for it
            # This enables robustness testing of the position reconstruction
            p_random = self.config.get('randomize_fraction_of_s2_top_array_photons', 0)
            if p_random:
                n_random = np.random.binomial(n=n_photons, p=p_random)
                hitp = self.distribute_photons_by_pattern(n_top - n_random, self.s2_patterns, (x, y))
                hitp += self.randomize_photons_over_channels(n_random, channels=self.config['channels_top'])
            else:
                hitp = self.distribute_photons_by_pattern(n_top, self.s2_patterns, (x, y))

            # The bottom photons are distributed randomly
            hitp += self.randomize_photons_over_channels(n_photons - n_top, channels=self.config['channels_bottom'])
            return hitp

        else:
            # Use the S1 pattern information
            if not self.s1_patterns:
                return self.randomize_photons_over_channels(n_photons, self.config['channels_in_detector']['tpc'])
            return self.distribute_photons_by_pattern(n_photons, self.s1_patterns, (x, y, z))

    def distribute_photons_by_pattern(self, n_photons, pattern_fitter, coordinate_tuple):
        # TODO: assumes channels drawn from top, or from all channels (i.e. first index 0!!!)
        # Note a CoordinateOutOfRange exception can be raised if points outside the TPC radius are asked
        # We don't catch it here: users shouldn't ask for simulations of impossible things :-)
        lces = pattern_fitter.expected_pattern(coordinate_tuple)
        return self.randomize_photons_over_channels(n_photons,
                                                    channels=range(len(lces)),
                                                    relative_lce_per_channel=lces)

    def randomize_photons_over_channels(self, n_photons, channels=None, relative_lce_per_channel=None):
        """Distribute photon_timings over channels according to relative_lce_per_channel

        :param n_photons: number of photons to distribute
        :param channels: list of channel numbers that can receive photons. This will still be filtered
         to include only channels in self.config['channels_for_photons'].
        :param relative_lce_per_channel: list of relative lce per channel. Should all be >= 0.
                                         If omitted, will distribute photons uniformly over channels.
                                         Does not have to be normalized to sum to 1.
        :return: array of length sim.config['n_channels'] with photon counts per channel
        """
        if n_photons == 0:
            return np.zeros(self.config['n_channels'], dtype=np.int64)

        # Include only channels that can receive photons
        if channels is None:
            channels = np.array(self.config['channels_for_photons'])
        else:
            channels = np.array(channels)
            sel = np.in1d(channels, self.config['channels_for_photons'])
            channels = channels[sel]
            if relative_lce_per_channel is not None:
                relative_lce_per_channel = relative_lce_per_channel[sel]

        # Ensure relative LCEs are valid, and sum to one (among the remaining channels):
        if relative_lce_per_channel is not None:
            relative_lce_per_channel = np.clip(relative_lce_per_channel, 0, 1)
            relative_lce_per_channel /= np.sum(relative_lce_per_channel)

        # Generate a channel index for every photon
        channel_index_for_p = np.random.choice(channels, size=n_photons, p=relative_lce_per_channel)

        # Count number of photons in each channel
        # Note the histogram range must include n_channels, even though n_channels-1 is the maximum value
        # This is because of how numpy handles values on bin edges
        hitp, _ = np.histogram(channel_index_for_p,
                               bins=self.config['n_channels'], range=(0, self.config['n_channels']))

        if not len(hitp) == self.config['n_channels']:
            raise RuntimeError("You found a simulator bug!\n"
                               "Hitpattern has wrong length "
                               "(%d, should be %d)" % (len(hitp), len(channels)))
        if not np.sum(hitp) == n_photons:
            raise RuntimeError("You found a simulator bug!\n"
                               "Hitpattern has wrong number of photons "
                               "(%d, should be %d)" % (np.sum(hitp), n_photons))
        return hitp


##
# Photon pulse generation
##

# I pulled this out of the Simulator class: caching using Memoize gave me trouble on methods due to the self argument
# There's still a method pmt_pulse_current, but it just calls pmt_pulse_current_raw defined below

@Memoize
def pmt_pulse_current_raw(offset, dt, samples_before, samples_after, tr, tf):
    return np.diff(exp_pulse(
        np.linspace(
            - offset - samples_before * dt,
            - offset + samples_after * dt,
            1 + samples_before + samples_after),
        units.electron_charge,
        tr,
        tf
    )) / dt


@np.vectorize
def exp_pulse(t, q, tr, tf):
    """Integrated current (i.e. charge) of a single-pe PMT pulse centered at t=0
    Assumes an exponential rise and fall waveform model
    :param t:   Time to integrate up to
    :param q:   Total charge in the pulse
    :param tr:  Rise time
    :param tf:  Fall time
    :return: Float, charge deposited up to t
    """
    c = 0.45512  # 1/(ln(10)-ln(10/9))
    if t < 0:
        return q / (tr + tf) * (tr * math.exp(t / (c * tr)))
    else:
        return q / (tr + tf) * (tr + tf * (1 - math.exp(-t / (c * tf))))


##
# PMT gain sampling
##

@Memoize
def _truncated_gauss(my_mean, my_std, left_boundary, right_boundary):
    """NB: the mean & std are only used to fix the boundaries, this is still a standardized normal otherwise!"""
    return stats.truncnorm(
        (left_boundary - my_mean) / my_std,
        (right_boundary - my_mean) / my_std)


def truncated_gauss_rvs(my_mean, my_std, left_boundary, right_boundary, n_rvs):
    """Get Gauss with specified mean and std, truncated to boundaries
    See http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.truncnorm.html
    """
    return _truncated_gauss(my_mean, my_std, left_boundary, right_boundary).rvs(n_rvs) * my_std + my_mean

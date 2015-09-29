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
from pax.utils import Memoize


log = logging.getLogger('SimulationCore')


class Simulator(object):

    def __init__(self, config_to_init):
        self.config = config_to_init

        # Should we repeat events?
        if 'event_repetitions' not in self.config:
            self.config['event_repetitions'] = 1

        # Primary excimer fraction from Nest Version 098
        # See G4S1Light.cc line 298
        density = self.config['liquid_density'] / (units.g / units.cm ** 3)
        excfrac = 0.4 - 0.11131 * density - 0.0026651 * density ** 2    # primary / secondary excimers
        excfrac = 1 / (1 + excfrac)                                     # primary / all excimers
        # primary / all excimers that produce a photon:
        excfrac /= 1 - (1 - excfrac) * (1 - self.config['s1_ER_recombination_fraction'])
        self.config['s1_ER_primary_excimer_fraction'] = excfrac
        log.debug('Inferred s1_ER_primary_excimer_fraction %s' % excfrac)

        # Recombination time from NEST 2014
        # 3.5 seems fishy, they fit an exponential to data, but in the code they use a non-exponential distribution...
        efield = (self.config['drift_field'] / (units.V / units.cm))
        self.config['s1_ER_recombination_time'] = 3.5 / 0.18 * (1 / 20 + 0.41) * math.exp(-0.009 * efield)
        log.debug('Inferred s1_ER_recombination_time %s ns' % self.config['s1_ER_recombination_time'])

        # Calculate particle number density in the gas (ideal gas law)
        number_density_gas = self.config['pressure'] / (units.boltzmannConstant * self.config['temperature'])

        # electric field in the gas
        # Formula from xenon:xenon100:analysis:jacob:s2gain_v2
        e_in_gas = self.config['lxe_dielectric_constant'] * self.config['anode_voltage'] / (
            self.config['lxe_dielectric_constant'] * self.config['elr_gas_gap_length'] +
            (self.config['gate_to_anode_distance'] - self.config['elr_gas_gap_length'])
        )

        # Reduced electric field in the gas
        self.config['reduced_e_in_gas'] = e_in_gas / number_density_gas
        log.debug("Inferred a reduced electric field of %s Td in the gas" % (
            self.config['reduced_e_in_gas'] / units.Td))

        # Which channels stand to receive any photons?
        # TODO: In XENON100, channel 0 will receive photons unless magically_avoid_dead_pmts=True
        # To prevent this, subtract 0 from channel_for_photons. But don't do that for XENON1T!!
        channels_for_photons = self.config['channels_in_detector']['tpc']
        if self.config.get('magically_avoid_dead_pmts', False):
            channels_for_photons = [ch for ch in channels_for_photons if self.config['gains'][ch] > 0]
        if self.config.get('magically_avoid_s1_excluded_pmts', False) and \
           'channels_excluded_for_s1' in self.config:
            channels_for_photons = [ch for ch in channels_for_photons
                                    if ch not in self.config['channels_excluded_for_s1']]
        self.config['channels_for_photons'] = channels_for_photons

        # Determine sensible length of a pmt pulse to simulate
        dt = self.config['sample_duration']
        self.config['samples_before_pulse_center'] = math.ceil(
            self.config['pulse_width_cutoff'] * self.config['pmt_rise_time'] / dt
        )
        self.config['samples_after_pulse_center'] = math.ceil(
            self.config['pulse_width_cutoff'] * self.config['pmt_fall_time'] / dt
        )
        log.debug('Simulating %s samples before and %s samples after PMT pulse centers.' % (
            self.config['samples_before_pulse_center'], self.config['samples_after_pulse_center']))

        # Load real noise data from file, if requested
        if self.config['real_noise_file']:
            self.noise_data = np.load(utils.data_file_name(self.config['real_noise_file']))['arr_0']
            # The silly XENON100 PMT offset again: it's relevant for indexing the array of noise data
            # (which is one row per channel)
            self.channel_offset = 1 if self.config['pmt_0_is_fake'] else 0

        # Init s2 pattern maps
        log.debug("Initializing s2 patterns...")
        if 's2_patterns_file' in self.config:
            self.s2_patterns = PatternFitter(utils.data_file_name(self.config['s2_patterns_file']),
                                             zoom_factor=self.config.get('s2_patterns_zoom_factor', 1),
                                             adjust_to_qe=self.config['quantum_efficiencies'][
                                                 self.config['channels_top']])
        else:
            self.s2_patterns = None

        # Init s1 pattern maps
        log.debug("Initializing s1 patterns...")
        if 's1_patterns_file' in self.config:
            self.s1_patterns = PatternFitter(utils.data_file_name(self.config['s1_patterns_file']),
                                             zoom_factor=self.config.get('s1_patterns_zoom_factor', 1))
        else:
            self.s1_patterns = None

    def s2_electrons(self, electrons_generated=None, z=0., t=0.):
        """Return a list of electron arrival times in the ELR region caused by an S2 process.

            electrons             -   total # of drift electrons generated at the interaction site
            t                     -   Time at which the original energy deposition occurred.
            z                     -   Depth below the GATE mesh where the interaction occurs.
        As usual, all units in the same system used by pax (if you specify raw values: ns, cm)
        """

        if z < 0:
            log.warning("Unphysical depth: %s cm below gate. Not generating S2." % z)
            return []
        log.debug("Creating an s2 from %s electrons..." % electrons_generated)

        # Average drift time, taking faster drift velocity after gate into account
        drift_time_mean = z / self.config['drift_velocity_liquid'] + \
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

    def s1_photons(self, n_photons, recoil_type, t=0.):
        """
        Returns a list of photon production times caused by an S1 process.

        """
        # Apply detection efficiency
        log.debug("Creating an s1 from %s photons..." % n_photons)
        n_photons = np.random.binomial(n=n_photons, p=self.config['s1_detection_efficiency'])
        log.debug("    %s photons are detected." % n_photons)
        if n_photons == 0:
            return np.array([])

        if recoil_type == 'ER':

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

        elif recoil_type == 'NR':

            # Neglible recombination time, same singlet/triplet ratio for primary & secondary excimers
            # Hence, we don't care about primary & secondary excimers at all:
            timings = self.singlet_triplet_delays(
                np.zeros(n_photons),
                t1=self.config['singlet_lifetime_liquid'],
                t3=self.config['triplet_lifetime_liquid'],
                singlet_ratio=self.config['s1_NR_singlet_fraction']
            )

        else:
            raise ValueError('Recoil type must be ER or NR, not %s' % type)

        return timings + t * np.ones(len(timings))

    def s2_scintillation(self, electron_arrival_times):
        """
        Given a list of electron arrival times, returns photon production times
        """

        # How many photons does each electron make?
        # TODO: xy correction!
        photons_produced = np.random.poisson(
            self.config['s2_secondary_sc_gain_density'] * self.config['elr_gas_gap_length'],
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
            t1=self.config['singlet_lifetime_gas'],
            t3=self.config['triplet_lifetime_gas'],
            singlet_ratio=self.config['singlet_fraction_gas']
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

    def make_hitpattern(self, photon_times, x=0, y=0, z=0):
        return SimulatedHitpattern(simulator=self, photon_timings=photon_times, x=x, y=y, z=z)

    def to_pax_event(self, hitpattern):
        """Simulate PMT response to a hitpattern of photons
        Returns None if you pass a hitlist without any hits
        returns start_time (in units, ie ns), pmt waveform matrix
        """
        if not isinstance(hitpattern, SimulatedHitpattern):
            raise ValueError("to_pax_event takes an instance of SimulatedHitpattern, you gave a %s." % type(hitpattern))

        # Create pax event
        start_time = int(time.time() * units.s)
        event = datastructure.Event(
            n_channels=self.config['n_channels'],
            start_time=start_time,
            stop_time=start_time + int(hitpattern.max + 2 * self.config['event_padding']),
            sample_duration=self.config['sample_duration'],
        )

        # Ensure the event length is even (else it cannot be written to XED)
        if event.length() % 2 != 0:
            event.stop_time += self.config['sample_duration']

        log.debug("Now performing hitpattern to waveform conversion for %s photons" % hitpattern.n_photons)
        # TODO: Account for random initial digitizer state  wrt interaction?
        # Where?

        # Convenience variables
        dt = self.config['sample_duration']
        dV = self.config['digitizer_voltage_range'] / 2 ** (self.config['digitizer_bits'])  # noqa

        # Build waveform channel by channel
        for channel, photon_detection_times in hitpattern.arrival_times_per_channel.items():
            # If the channel is dead, we don't do anything.
            if self.config['gains'][channel] == 0:
                continue

            photon_detection_times = np.array(photon_detection_times)

            log.debug("Simulating %d photons in channel %d (gain=%s, gain_sigma=%s)" % (
                len(photon_detection_times), channel,
                self.config['gains'][channel], self.config['gain_sigmas'][channel]))

            if self.config['pmt_0_is_fake'] and channel == 0:
                continue

            #  Add padding, sort (eh.. or were we already sorted? and is sorting necessary at all??)
            all_pmt_pulse_centers = np.sort(photon_detection_times + self.config['event_padding'])

            if self.config['cheap_zle']:
                # No photons in this channel -- don't bother simulating noise
                if len(photon_detection_times) == 0:
                    continue
                # Cluster into pulses for cheap ZLE
                pmt_pulse_center_clusters = utils.cluster_by_diff(all_pmt_pulse_centers, 2 * self.config['zle_padding'])
            else:
                # All in one cluster...
                pmt_pulse_center_clusters = [all_pmt_pulse_centers]

            for pmt_pulse_centers in pmt_pulse_center_clusters:

                # Build the waveform pulse by pulse (bin by bin was slow, hope this
                # is faster)

                # Compute offset & center index for each pe-pulse
                # 'index' refers to the (hypothetical) event waveform, as usual
                pmt_pulse_centers = np.array(pmt_pulse_centers, dtype=np.int)
                offsets = pmt_pulse_centers % dt
                center_index = (pmt_pulse_centers - offsets) / dt   # Absolute index in waveform of pe-pulse center
                center_index = center_index.astype(np.int)

                if self.config['cheap_zle']:
                    # For cheap ZLE, define some padding around the cluster
                    start_index = np.min(center_index) - int(self.config['zle_padding'] / dt)
                    end_index = max(center_index) + int(self.config['zle_padding'] / dt)
                else:
                    # Simulate an event-long waveform in this channel
                    # Remember start padding has already been added to times, so just one padding in end_index
                    start_index = 0
                    end_index = event.length() - 1
                pulse_length = end_index - start_index + 1

                current_wave = np.zeros(pulse_length)

                if len(center_index) > self.config['use_simplified_simulator_from']:

                    # TODO: Is this actually faster still? Should check!

                    # Start with a delta function single photon pulse, then convolve with one actual single-photon pulse
                    # This effectively assumes photons always arrive at the start of a digitizer t-bin,
                    # and also
                    # but is much faster

                    # Division by dt necessary for charge -> current
                    unique, counts = np.unique(center_index - start_index, return_counts=True)
                    unique = unique.astype(np.int)
                    current_wave[unique] = counts * self.config['gains'][channel] * units.electron_charge / dt

                    # Previous, slow implementation
                    # pulse_counts = Counter(center_index)
                    # print(pulse_counts)
                    # current_wave2 = np.array([pulse_counts[n] for n in range(n_samples)]) \
                    #                * self.config['gains'][channel] * units.electron_charge / dt

                    # Calculate a normalized pmt pulse, then convolve with it
                    normalized_pulse = self.pmt_pulse_current(gain=1)
                    normalized_pulse /= np.sum(normalized_pulse)
                    current_wave = np.convolve(current_wave, normalized_pulse, mode='same')

                elif len(center_index) > 0:

                    # Use a Gaussian truncated to positive values for the SPE gain distribution
                    gains = truncated_gauss_rvs(
                        my_mean=self.config['gains'][channel],
                        my_std=self.config['gain_sigmas'][channel],
                        left_boundary=0,
                        right_boundary=float('inf'),
                        n_rvs=len(pmt_pulse_centers))

                    # Do the full, slower simulation for each single-photon pulse
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

                        # Debugging stuff
                        if len(generated_pulse) != righter_index - left_index:
                            raise RuntimeError(
                                "Generated pulse is %s samples long, can't be inserted between %s and %s" % (
                                    len(generated_pulse), left_index, righter_index))

                        if left_index < 0:
                            raise RuntimeError("Invalid left index %s: can't be negative!" % left_index)

                        if righter_index >= len(current_wave):
                            raise RuntimeError("Invalid right index %s: can't "
                                               "be longer than length of wave "
                                               "(%s)!" % (righter_index,
                                                          len(current_wave)))

                        current_wave[left_index: righter_index] += generated_pulse

                # Did you order some Gaussian current noise with that?
                if self.config['gauss_noise_sigma']:
                    # / dt is for charge -> current conversion, as in pmt_pulse_current
                    noise_sigma_current = self.config['gauss_noise_sigma'] * self.config['gains'][channel] / dt,
                    current_wave += np.random.normal(0,
                                                     noise_sigma_current,
                                                     len(current_wave))

                # Convert from PMT current to ADC counts
                adc_wave = current_wave
                adc_wave *= self.config['pmt_circuit_load_resistor']    # Now in voltage
                adc_wave *= self.config['external_amplification']       # Now in voltage after amplifier
                adc_wave /= dV                                          # Now in float ADC counts above baseline
                adc_wave = np.trunc(adc_wave)                           # Now in integer ADC counts "" ""
                # Could round instead of trunk... who cares?

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
        return event

    def distribute_s2_photons(self, n_photons, x, y):
        if not self.s2_patterns:
            return self.randomize_photons_over_channels(n_photons, self.config['channels_top'])

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
        hitp += self.randomize_photons_over_channels(n_photons - n_top,
                                                     channels=self.config['channels_bottom'])
        return hitp

    def distribute_s1_photons(self, n_photons, x, y, z):
        if not self.s1_patterns:
            return self.randomize_photons_over_channels(n_photons, self.config['channels_in_detector']['tpc'])
        # The z-dimension of the S1 pattern map is in drift time
        # TODO: compensate for S2 width & drift velocity increase after gate (both ~us effects though, not important)
        drift_time = z / self.config['drift_velocity_liquid']
        return self.distribute_photons_by_pattern(n_photons, self.s1_patterns, (x, y, drift_time))


    def distribute_photons_by_pattern(self, n_photons, pattern_fitter, coordinate_tuple):
        # TODO: assumes channels drawn from top, or from all channels (i.e. first index 0!!!)
        lces = pattern_fitter.expected_pattern(coordinate_tuple)
        if np.sum(lces) == 0:
            raise ValueError("LCEs at position %s are all zero, cannot be normalized!" % (coordinate_tuple,))
        return self.randomize_photons_over_channels(n_photons,
                                                    channels=range(len(lces)),
                                                    relative_lce_per_channel=lces)

    def randomize_photons_over_channels(self, n_photons, channels=None, relative_lce_per_channel=None):
        """Distribute photon_timings over channels according to relative_lce_per_channel

        :param n_photons: number of photons to distribute
        :param channels: list of channel numbers that can receive photons. This will still be filtered
         to include only channels in self.config['channels_for_photons'].
        :param relative_lce_per_channel: list of relative lce per channel. Should be >= 0 and sum to 1.
                                         If omitted, will distribute photons uniformly over channels.
        :return: array of length sim.config['n_channels'] with photon counts per channel
        """
        if n_photons == 0:
            return np.zeros(self.config['n_channels'])

        # Include only channels that can receive photons
        if channels is None:
            channels = np.array(self.config['channels_for_photons'])
        else:
            channels = np.array(channels)
            sel = np.in1d(channels, self.config['channels_for_photons'])
            channels = channels[sel]
            if relative_lce_per_channel is not None:
                relative_lce_per_channel = relative_lce_per_channel[sel]

        # Ensure relative LCEs are valid to one:
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
# Hitpattern simulation
##

class SimulatedHitpattern(object):

    def __init__(self, simulator, photon_timings, x=0, y=0, z=0):
        self.config = simulator.config

        # Correct for PMT TTS
        photon_timings += np.random.normal(
            self.config['pmt_transit_time_mean'],
            self.config['pmt_transit_time_spread'],
            len(photon_timings)
        )

        # Add the minimum and maximum times, and number of times
        # hitlist_to_waveforms would have to go through weird flattening stuff to determine these
        self.min = min(photon_timings)
        self.max = max(photon_timings)
        self.n_photons = len(photon_timings)

        if not len(photon_timings):
            raise ValueError('Need at least 1 photon timing to produce a valid hitpattern')

        # Shuffle all timings in the array, so channel 1 doesn't always get the first photon
        # Don't rely on randomize_photons_over_channels to do this, we'll be splitting top v bottom here
        # and want that split to be random too.
        np.random.shuffle(photon_timings)

        if z == - self.config['gate_to_anode_distance']:
            # Generated at anode: use S2 LCE data
            hitp = simulator.distribute_s2_photons(self.n_photons, x, y)
        else:
            hitp = simulator.distribute_s1_photons(self.n_photons, x, y, z)

        # Split photon times over channels
        self.arrival_times_per_channel = dict(zip(range(simulator.config['n_channels']),
                                                  np.split(photon_timings, np.cumsum(hitp))))

    def __add__(self, other):
        # Don't reuse __init__, we don't want another TTS correction..
        # print("add called self=%s, other=%s" % (type(self), type(other)))
        self.min = min(self.min, other.min)
        self.max = max(self.max, other.max)
        self.n_photons = self.n_photons + other.n_photons
        contributing_channels = set(self.arrival_times_per_channel.keys()) | set(other.arrival_times_per_channel.keys())
        self.arrival_times_per_channel = {
            ch: np.concatenate((
                self.arrival_times_per_channel.get(ch,  np.array([])),
                other.arrival_times_per_channel.get(ch, np.array([]))
            ))
            for ch in contributing_channels
        }
        return self

    def __radd__(self, other):
        # print("radd called self=%s, other=%s" % (type(self), type(other)))
        if other is 0:
            # Apparently sum() starts trying to add stuff to 0...
            return self
        self.__add__(other)


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

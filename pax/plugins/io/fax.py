import numpy as np
import math
import random
import time
import csv
try:
    import cPickle as pickle
except:
    import pickle
from pax import plugin, units, datastructure
from collections import Counter

# This is probably in some standard library...
def flatten(l):
    return [item for sublist in l for item in sublist]

def exp_pulse_raw(t, q, tr, tf):
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
exp_pulse = np.vectorize(exp_pulse_raw, excluded={1, 2, 3})

class FaX(plugin.InputPlugin):

    def startup(self):

        # Open the instructions file & truth file
        self.instructions_file = open(self.config['instruction_filename'], 'r')
        self.instructions = csv.DictReader(self.instructions_file)
        self.truth_file = open(self.config['truth_filename'], 'wb')

        # Should we repeat events?
        if not 'event_repetitions' in self.config:
            self.config['event_repetitions'] = 1

        # Determine sensible length of a pmt pulse to simulate
        self.dt = self.config['digitizer_t_resolution']
        self.samples_before_pulse_center = math.ceil(
            self.config['pulse_width_cutoff'] * self.config['pmt_rise_time'] / self.dt)
        self.samples_after_pulse_center = math.ceil(
            self.config['pulse_width_cutoff'] * self.config['pmt_fall_time'] / self.dt)

        # Padding on eiher side of event
        if 'event_padding' not in self.config:
            self.config['event_padding'] = 0

        # Padding before & after each pulse/peak/photon-cluster/whatever-you-call-it
        if not 'pad_after' in self.config:
            self.config['pad_after'] = 30 * self.dt + self.samples_after_pulse_center
        if not 'pad_before' in self.config:
            self.config['pad_before'] = (
                # 10 + Baseline bins
                50 * self.dt
                # Protection against early pre-peak rise
                + self.samples_after_pulse_center
                # Protection against pulses arriving earlier than expected due
                # to tail of TTS distribution
                + 10 * self.config['pmt_transit_time_spread']
                - self.config['pmt_transit_time_mean']
            )

        # Temp hack: need 0 in channels so we can use lists... hmmzzz
        self.channels = list({0} | self.config['pmts_top'] | self.config['pmts_bottom'])


    def shutdown(self):
        self.instructions_file.close()

    def write_truth(self, stuff):
        pickle.dump(stuff, self.truth_file)

    def s1_photons(self, photons, t=0, recombination_time=None, singlet_fraction=None, primary_excimer_fraction=None):
        """
        Returns a list of photon production times caused by an S1 process.

        """

        # Fill in optional arguments
        if recombination_time is None:
            recombination_time = self.config['s1_default_recombination_time']
        if singlet_fraction is None:
            singlet_fraction = self.config['s1_default_singlet_fraction']
        if primary_excimer_fraction is None:
            primary_excimer_fraction = self.config[
                's1_default_primary_excimer_fraction']
        # Apply detection efficiency
        self.log.debug("Creating an s1 from %s photons..." % photons)
        n_photons = np.random.binomial(
            n=photons, p=self.config['s1_detection_efficiency'])
        self.log.debug("    %s photons are detected." % n_photons)
        if n_photons == 0:
            return np.array([])
        # How many of these are primary excimers? Others arise through
        # recombination.
        n_primaries = np.random.binomial(
            n=n_photons, p=primary_excimer_fraction)
        # Handle recombination delays
        photon_times = t + np.concatenate([
            np.zeros(n_primaries),
            recombination_time *
            (1 / np.random.uniform(0, 1, n_photons - n_primaries) - 1)
        ])
        # Account for singlet/triplet decay times
        timings = self.singlet_triplet_delays(
            photon_times, t1=self.config['singlet_lifetime_liquid'], t3=self.config[
                'triplet_lifetime_liquid'],
            singlet_ratio=singlet_fraction
        )
        return timings

    def s2_electrons(self, electrons_generated=None, z=0., t=0.):
        """Return a list of electron arrival times in the ELR region caused by an S2 process.

            electrons             -   total # of drift electrons generated at the interaction site
            t                     -   Time at which the original energy deposition occurred.
            z                     -   Depth below the GATE mesh where the interaction occurs.
        As usual, all units in the same system used by pax (if you specify raw values: ns, cm)
        """

        if z < 0:
            self.log.warning("Unphysical depth: %s cm below gate. Not generating S2." % z)
            return []
        self.log.debug("Creating an s2 from %s electrons..." % electrons_generated)

        # Average drift time, taking faster drift velocity after gate into account
        drift_time_mean = z / self.config['drift_velocity_liquid'] + \
            (self.config['gate_to_anode_distance'] - self.config['elr_gas_gap_length']) \
            / self.config['drift_velocity_liquid_above_gate']

        # Diffusion model from Sorensen 2011
        drift_time_stdev = math.sqrt(
            2 * self.config['diffusion_constant_liquid'] *
            drift_time_mean / (self.config['drift_velocity_liquid']) ** 2
        )

        # Absorb electrons during the drift
        electrons_seen = np.random.binomial(
            n= electrons_generated,
            p= self.config['electron_extraction_yield']
               * math.exp(- drift_time_mean / self.config['electron_lifetime_liquid'])
        )
        self.log.debug("    %s electrons survive the drift." % electrons_generated)

        # Calculate electron arrival times in the ELR region
        e_arrival_times = t + np.random.exponential(self.config['electron_trapping_time'], electrons_seen)
        if drift_time_stdev:
            e_arrival_times += np.random.normal(drift_time_mean, drift_time_stdev, electrons_seen)
        return e_arrival_times

    def s2_scintillation(self, electron_arrival_times):
        """
        Given a list of electron arrival times, returns photon production times
        """
        # How many photons does each electron make?
        # TODO: xy correction!
        photons_produced = np.random.poisson(
            self.config['s2_secondary_sc_yield_density'] *
            self.config['elr_gas_gap_length'],
            len(electron_arrival_times)
        )
        total_photons = np.sum(photons_produced)
        self.log.debug(
            "    %s scintillation photons will be detected." % total_photons)
        if total_photons == 0:
            return np.array([])
        # Find the photon production times
        # Assume luminescence probability ~ electric field
        s2_pe_times = np.concatenate([
            t0 +
            self.get_luminescence_positions(
                photons_produced[i]) / self.config['drift_velocity_gas']
            for i, t0 in enumerate(electron_arrival_times)
        ])
        # Account for singlet/triplet excimer decay times
        return self.singlet_triplet_delays(
            s2_pe_times,
            t1=self.config['singlet_lifetime_gas'],
            t3=self.config['triplet_lifetime_gas'],
            singlet_ratio=self.config['singlet_fraction_gas']
        )

    @staticmethod
    def singlet_triplet_delays(times, t1, t3, singlet_ratio):
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

    def get_luminescence_positions(self, n):
        """Sample luminescence positions in the ELR, using a mixed wire-dominated / uniform field"""
        x = np.random.uniform(0, 1, n)
        l = self.config['elr_gas_gap_length']
        wire_par = self.config['wire_field_parameter']
        rm = self.config['anode_mesh_pitch'] * wire_par
        rw = self.config['anode_wire_radius']
        if wire_par == 0:
            return x * l
        totalArea = l + rm * (math.log(rm / rw) - 1)
        relA_wd_region = rm * math.log(rm / rw) / totalArea
        # AARRGHH! Should vectorize...
        return np.array([
            (l - np.exp(xi * totalArea / rm) * rw)
            if xi < relA_wd_region
            else l - (xi * totalArea + rm * (1 - math.log(rm / rw)))
            for xi in x
        ])

    def photons_to_hitlist(self, photon_timings, x=0., y=0., z=0.):
        """Compute photon arrival time list ('hitlist') from photon production times

        :param photon_timings: list of times at which photons are produced at this position
        :param x: x-coordinate of photon production site
        :param y: y-coordinate of photon production site
        :param z: z-coordinate of photon production site
        :return: numpy array, indexed by pmt number, of numpy arrays of photon arrival times
        """
        # TODO: Use light collection map to divide photons
        # TODO: if positions unspecified, pick a random position (useful for
        # poisson noise)
        # So channel 1 doesn't always get the first photon...
        random.shuffle(photon_timings)
        # Determine how many photons each pmt gets
        # TEMP - Uniformly distribute photons over all PMTs!
        # TEMP - hack to prevent photons getting into the ghost channel 0
        channels_for_photons = list(set(self.channels) - set([0]))
        if 'magically_avoid_dead_pmts' in self.config and self.config['magically_avoid_dead_pmts']:
            channels_for_photons = [
                ch for ch in channels_for_photons if self.config['gains'][ch] > 0]
        if 'magically_avoid_s1_excluded_pmts'in self.config and self.config['magically_avoid_dead_pmts']:
            channels_for_photons = [
                ch for ch in channels_for_photons if not ch in self.config['pmts_excluded_for_s1']]
        hit_counts = Counter([
            random.choice(channels_for_photons)
            for _ in photon_timings
        ])
        # Make the hitlist, a numpy array, so we can add it elementwise so we
        # can add them
        hitlist = []
        already_used = 0
        for ch in self.channels:
            hitlist.append(
                sorted(photon_timings[already_used:already_used + hit_counts[ch]]))
            already_used += hit_counts[ch]
        self.log.debug("    %s hits in hitlist" %
                       sum([len(x) for x in hitlist]))
        # TODO: factor in propagation time
        return np.array(hitlist)

    def pmt_pulse_current(self, gain, offset=0):
        return np.diff(exp_pulse(
            np.linspace(
                - offset - self.samples_before_pulse_center * self.dt,
                - offset + self.samples_after_pulse_center * self.dt,
                1 + self.samples_after_pulse_center +
                self.samples_before_pulse_center
            ),
            gain * units.electron_charge,
            self.config['pmt_rise_time'], self.config['pmt_fall_time']
        )) / self.dt

    def hitlist_to_waveforms(self, hitlist):
        """Simulate PMT response to incoming photons
        Returns None if you pass a hitlist without any hits
        """
        # TODO: Account for random initial digitizer state  wrt interaction?
        # Where?

        # Convenience variables
        dt = self.dt
        dV = self.config['digitizer_voltage_range'] / \
            2 ** (self.config['digitizer_bits'])
        pad_before = self.config['pad_before']
        pad_after = self.config['pad_after']

        # Compute waveform start, length, end
        all_photons = flatten(hitlist)
        if not all_photons:
            return None
        start_time = min(all_photons) - pad_before
        n_samples = math.ceil(
            (max(all_photons) + pad_after - start_time) / dt) + 2 + self.samples_after_pulse_center

        # Build waveform channel by channel
        pmt_waveforms = np.zeros((len(hitlist), n_samples), dtype=np.int16)
        for channel, photon_detection_times in enumerate(hitlist):
            if len(photon_detection_times) == 0:
                continue  # No photons in this channel

            # Correct for PMT transit time, subtract start_time, and (re-)sort
            pmt_pulse_centers = np.sort(
                photon_detection_times - start_time + np.random.normal(
                    self.config['pmt_transit_time_mean'],
                    self.config['pmt_transit_time_spread'],
                    len(photon_detection_times)
                )
            )

            # Build the waveform pulse by pulse (bin by bin was slow, hope this
            # is faster)

            # Compute offset & center index for each pulse
            offsets = pmt_pulse_centers % dt
            center_index = (pmt_pulse_centers - offsets) / dt
            if len(all_photons) > self.config['use_simplified_simulator_from']:
                # Start with a delta function single photon pulse, then convolve with one actual single-photon pulse
                # This effectively assumes photons always arrive at the start of a digitizer t-bin, but is much faster
                pulse_counts = Counter(center_index)
                current_wave = np.array([pulse_counts[n] for n in range(
                    n_samples)]) * self.config['gains'][channel] * units.electron_charge / dt
                # Calculate a normalized pmt pulse, for use in convolution later (only
                # for large peaks)
                normalized_pulse = self.pmt_pulse_current(gain=1)
                normalized_pulse /= np.sum(normalized_pulse)
                current_wave = np.convolve(current_wave, normalized_pulse, mode='same')
            else:
                # Do the full, slower simulation for each single-photon pulse
                current_wave = np.zeros(n_samples)
                for i, t0 in enumerate(pmt_pulse_centers):
                    # Add some current for this photon pulse
                    # Compute the integrated pmt pulse at various samples, then
                    # do their diffs/dt
                    current_wave[
                        # +1 due to np.diff in pmt_pulse_current
                        center_index[i] - self.samples_before_pulse_center + 1 :
                        center_index[i] + 1 + self.samples_after_pulse_center
                    ] += self.pmt_pulse_current(
                        # Really a Poisson (although mean is so high it is very
                        # close to a Gauss)
                        gain=np.random.poisson(self.config['gains'][channel]),
                        offset=offsets[i]
                    )

            # Add white noise current - only to channels that have seen a
            # photon, only around a peak
            current_wave += np.random.normal(
                0, self.config['white_noise_sigma'], len(current_wave))

            # Convert current to digitizer count (should I trunc, ceil or floor?) and store
            # Don't baseline correct, clip or flip down here, we do that at the
            # very end when all signals are combined
            temp = np.trunc(
                self.config['pmt_circuit_load_resistor'] * self.config['external_amplification'] / dV *
                current_wave
            )
            pmt_waveforms[channel] = temp.astype(np.int16)

        return start_time, pmt_waveforms

    def s2(self, electrons, t=0., z=0., x=0., y=0.):
        electron_times = self.s2_electrons(electrons_generated=electrons, t=t, z=z)
        photon_times = self.s2_scintillation(electron_times)
        self.truth_peaks.append({
            'signal_type': 's2',
            'x': x, 'y': y, 'z': z,
            't_interaction': t,
            'photons': len(photon_times),
            't_photons': np.mean(photon_times),
            'electrons': len(electron_times),
            'e_cloud_sigma': np.std(electron_times),
        })
        return self.hitlist_to_waveforms(
            self.photons_to_hitlist(photon_times),
        )

    def s1(self, photons, t=0, recombination_time=None, singlet_fraction=None):
        """
        :param photons: total # of photons generated in the S1
        :param t: Time at which the interaction occurs, i.e. offset for arrival times. Defaults to s1_default_recombination_time
        :param recombination_time: Fraction of recombining eximers that decay as singlets. Defaults to s1_default_eximer_fraction
        :param singlet_fraction: Recombination time (\tau_r in Nest papers).
        :return: start_time, pmt_waveforms
        """
        photon_times = self.s1_photons(photons, t, recombination_time, singlet_fraction)
        self.truth_peaks.append({
            'peak_type': 's1',
            #'x': x, 'y': y, 'z': z,
            't_interaction': t,
            'photons': len(photon_times),
            't_photons': np.mean(photon_times),
        })
        return self.hitlist_to_waveforms(
            self.photons_to_hitlist(photon_times),
        )

    def get_events(self):

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
                     2 * self.config['event_padding'] / self.dt +
                     max([s[0] - start_time_offset for s in signals]) / self.dt +
                     max([s[1].shape[1] for s in signals])
                )
                pmt_waveforms = self.config['digitizer_baseline'] *\
                                np.ones((len(self.channels), event_length), dtype=np.int16)
                for s in signals:
                    start_index = int(
                        (s[0] - start_time_offset + self.config['event_padding']) / self.dt
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
                event.stop_time = event.start_time + int(event_length * self.dt)
                event.sample_duration = self.dt
                # Make a single occurrence for the entire event... yeah, this
                # should be changed
                event.occurrences = {
                    ch: [(0, pmt_waveforms[ch])]
                    for ch in self.channels
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

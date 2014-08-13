import numpy as np
import math, random
from pax import plugin, units, datastructure
from collections import Counter

#This is probably in some standard library...
def flatten(l): return [item for sublist in l for item in sublist]

# Integrated PMT pulses
def delta_pulse_raw(t, q, tr, tf):
    """Delta function pulse: all charge in single bin"""
    return t>0 and q or 0
delta_pulse = np.vectorize(delta_pulse_raw, excluded=[1,2,3])
def exp_pulse_raw(t, q, tr, tf):
    """Exponential pulse: exponential rise and fall time"""
    c = 0.45512 #1/(ln(10)-ln(10/9))
    if t<0:
        return q/(tr+tf) * (tr*math.exp(t/(c*tr)))
    else:
        return q/(tr+tf) * (tr + tf*(1-math.exp(-t/(c*tf))))
exp_pulse = np.vectorize(exp_pulse_raw, excluded=[1,2,3])


class FaX(plugin.InputPlugin):

    def startup(self):
        self.dt = self.config['digitizer_t_resolution']
        # Determine sensible length of a pmt pulse to simulate
        self.samples_before_pulse_center = math.ceil(self.config['pulse_width_cutoff']*self.config['pmt_rise_time']/self.dt)
        self.samples_after_pulse_center  = math.ceil(self.config['pulse_width_cutoff']*self.config['pmt_fall_time']/self.dt)
        # Padding to add before & after a peak
        if not 'pad_after' in self.config:
            self.config['pad_after'] = 10 * self.dt + self.samples_after_pulse_center
        if not 'pad_before' in self.config:
            self.config['pad_before'] = (
                # 10 + Baseline bins
                50 * self.dt
                # Protection against early pre-peak rise
                + self.samples_after_pulse_center
                # Protection against pulses arriving earlier than expected due to tail of TTS distribution
                + 10*self.config['pmt_transit_time_spread'] - self.config['pmt_transit_time_mean']
            )
        # Temp hack: need 0 in so we can use lists
        self.channels = list(set([0]) | self.config['pmts_top'] | self.config['pmts_bottom'])
        # Calculate a normalized pmt pulse, for use in convolution later
        self.normalized_pulse = self.pmt_pulse_current(gain=1)
        self.normalized_pulse /= np.sum(self.normalized_pulse)

    def s2_electrons(self, electrons_generated=None, z=10.*units.cm, t=0.):
        """Return a list of electron arrival times in the ELR region caused by an S2 process.

            electrons             -   total # of drift electrons generated at the interaction site
            t                     -   Time at which the original energy deposition occurred.
            z                     -   Depth (in the liquid below the ELR) where the interaction occurs.
        As usual, all units in the same system used by pax (ns, cm)
        """
        # Diffusion model from Sorensen 2011
        drift_time_mean  = z/self.config['drift_velocity_liquid']
        drift_time_stdev = math.sqrt(
            2 * self.config['diffusion_constant_liquid']* drift_time_mean / (self.config['drift_velocity_liquid'])**2
        )
        # Absorb electrons during the drift
        electrons_seen = np.random.binomial(
            n=electrons_generated,
            p=self.config['electron_extraction_yield']*math.exp(-drift_time_mean/self.config['electron_lifetime_liquid'])
        )
        #Calculate electron arrival times in the ELR region
        return t + \
               np.random.normal(drift_time_mean, drift_time_stdev, electrons_seen) +\
               np.random.exponential(self.config['electron_trapping_time'], electrons_seen)

    def s2_scintillation(self, electron_arrival_times):
        """
        Given a list of electron arrival times, returns photon production times
        """
        # How many photons does each electron make?
        # TODO: xy correction!
        photons_produced = np.random.poisson(
            self.config['s2_secondary_sc_yield_density']*self.config['elr_length'],
            len(electron_arrival_times)
        )
        # Find the photon production times
        # Assume luminescence probability ~ electric field
        s2_pe_times =np.concatenate([
            t0 + self.get_luminescence_positions(photons_produced[i]) / self.config['drift_velocity_gas']
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
            np.random.exponential(t3, len(times)-n_singlets)
        ])

    def get_luminescence_positions(self, n):
        """Sample luminescence positions in the ELR, using a mixed wire-dominated / uniform field"""
        x = np.random.uniform(0, 1, n)
        l = self.config['elr_length']
        wire_par = self.config['wire_field_parameter']
        rm = self.config['anode_mesh_pitch'] * wire_par
        rw = self.config['anode_wire_radius']
        if wire_par == 0:
            return x * l
        totalArea = l + rm*(math.log(rm/rw)-1)
        relA_wd_region =  rm * math.log(rm/rw)/totalArea
        # AARRGHH! Should vectorize...
        return np.array([
            (l - np.exp(xi*totalArea/rm)*rw)
            if xi < relA_wd_region
            else l - (xi*totalArea+rm*(1-math.log(rm/rw)))
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
        #TODO: Use light collection map to divide photons
        #TODO: if positions unspecified, pick a random position (useful for poisson noise)
        random.shuffle(photon_timings) #So channel 1 doesn't always get the first photon...
        # Determine how many photons each pmt gets
        # TEMP - Uniformly distribute photons over all PMTs!
        # TEMP - hack to prevent photons getting into the ghost channel 0
        channels_for_photons = list(set(self.channels) - set([0]))
        hit_counts =  Counter([
            random.choice(channels_for_photons)
            for _ in photon_timings
        ])
        #Make the hitlist, a numpy array, so we can add it elementwise so we can add them
        hitlist = []
        already_used = 0
        for ch in self.channels:
            hitlist.append( sorted(photon_timings[already_used:already_used+hit_counts[ch]]) )
            already_used += hit_counts[ch]
        #TODO: factor in propagation time
        return np.array(hitlist)

    def pmt_pulse_current(self, gain, offset=0):
        return np.diff(exp_pulse(
            np.linspace(
                - offset - self.samples_before_pulse_center*self.dt,
                - offset + self.samples_after_pulse_center*self.dt,
                1 + self.samples_after_pulse_center + self.samples_before_pulse_center
            ),
            gain * units.electron_charge,
            self.config['pmt_rise_time'], self.config['pmt_fall_time']
        ))/self.dt

    def hitlist_to_waveforms(self, hitmap):
        """Simulate PMT response to incoming photons"""
        #TODO: Account for random initial digitizer state  wrt interaction? Where?

        # Convenience variables
        dt = self.dt
        dV = self.config['digitizer_voltage_range'] / 2**(self.config['digitizer_bits'])
        pad_before = self.config['pad_before']
        pad_after  = self.config['pad_after']
        pmt_integrated_pulse = exp_pulse


        # Compute waveform start, length, end
        all_photons = flatten(hitmap)
        start_time = min(all_photons)-pad_before
        n_samples = math.ceil((max(all_photons) + pad_after - start_time)/dt) + 2 + self.samples_after_pulse_center
        end_time = start_time + (n_samples - 1) * dt

        # Build waveform channel by channel
        pmt_waveforms = np.zeros((len(hitmap), n_samples), dtype=np.uint16)
        for channel, photon_detection_times in enumerate(hitmap):
            if len(photon_detection_times) == 0: continue   #No photons in this channel

            # Correct for PMT transit time, subtract start_time, and (re-)sort
            pmt_pulse_centers = np.sort(
                photon_detection_times - start_time + np.random.normal(
                    self.config['pmt_transit_time_mean'],
                    self.config['pmt_transit_time_spread'],
                    len(photon_detection_times)
                )
            )

            # Build the waveform pulse by pulse (bin by bin was slow, hope this is faster)

            # Compute offset & center index for each pulse
            offsets = pmt_pulse_centers % dt
            center_index = (pmt_pulse_centers - offsets) / dt
            if len(all_photons) > self.config['use_simplified_simulator_from']:
                # Assume a delta function single photon pulse, then convolve with the actual single-photon pulse
                # This effectively assumes photons always arrive at the start of a digitizer t-bin, but is much faster
                pulse_counts = Counter(center_index)
                current_wave = np.array([pulse_counts[n] for n in range(n_samples)]) * self.config['gains'][channel] * units.electron_charge / dt
                current_wave = np.convolve(current_wave, self.normalized_pulse, mode='same')
            else:
                # Do the full, slower simulation for each single-photon pulse
                current_wave = np.zeros(n_samples)
                for i, t0 in enumerate(pmt_pulse_centers):
                    # Add some current for this photon pulse
                    # Compute the integrated pmt pulse at various samples, then do their diffs/dt
                    current_wave[
                           center_index[i] - self.samples_before_pulse_center + 1    # +1 due to np.diff in pmt_pulse_current
                         : center_index[i] + 1 +self.samples_after_pulse_center
                    ] = self.pmt_pulse_current(gain=self.config['gains'][channel], offset=offsets[i])

            # Convert current to digitizer count (should I trunc, ceil or floor?), clip, and store
            temp = np.trunc(
                self.config['digitizer_baseline'] -
                self.config['pmt_circuit_load_resistor'] * self.config['external_amplification'] / dV *
                current_wave
            )
            pmt_waveforms[channel] = np.clip(temp.astype(np.uint16), 0, 2**(self.config['digitizer_bits']))

        return pmt_waveforms

    def s2(self, electrons, t=0., z=0.):
        return self.hitlist_to_waveforms(
            self.photons_to_hitlist(
                self.s2_scintillation(
                    self.s2_electrons(electrons_generated=electrons, t=t, z=z)
                )
            ),
        )

    def add_noise(self, waveform):
        pass

    def get_events(self):

        self.log.debug("Starting simulation...")
        pmt_waveforms = self.s2(3, 0, 0.01)
        self.log.debug("Done!")

        # Create and yield a pax event
        event = datastructure.Event()
        event.event_number = 0  # TODO
        event.event_start = 0
        event.event_stop = 0 + self.dt * pmt_waveforms.shape[1]
        event.sample_duration = self.dt
        assert event.length() == pmt_waveforms.shape[1]
        event.occurrences = {
            ch : [(event.event_start, pmt_waveforms[ch])]
            for ch in range(len(pmt_waveforms))
        }
        yield event
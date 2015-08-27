"""Data structure for pax

This is meant to be a fixed data structure that people can use to access
physically meaningful variables.  For example, S2.

NOTE: This class is stable within major releases.  Do not change any variable
names of functionality between major releases.  You may add variables in minor
releases.  Patch releases cannot modify this.
"""

import inspect

import numpy as np

import math
from pax import units, utils

# DO NOT use Model instead of StrictModel:
# It improves performance, but kills serialization (numpy int types will appear in class etc)
# TODO: For Hit class, we may want Model for performance?
#       Look where the numpy int types get in, force them to python ints.
from pax.data_model import StrictModel, ListField


INT_NAN = -99999    # Do not change without talking to me. -Tunnell 12/3/2015 ... and me. -Jelle 05/08/2015


class ReconstructedPosition(StrictModel):
    """Reconstructed position

    Each reconstruction algorithm creates one of these.
    """
    x = float('nan')  #: x position (cm)
    y = float('nan')  #: y position (cm)

    #: goodness-of-fit parameter generated with PosRecChiSquareGamma
    goodness_of_fit = float('nan')
    # : number of degrees of freedom calculated with PosRecChiSquareGamma
    ndf = float('nan')

    #: Name of algorithm used for computation
    algorithm = 'none'

    # : Errors - currently not used
    # error_matrix = np.array([], dtype=np.float64)

    # For convenience: cylindrical coordinates
    # Must be properties so InterpolatingDetectorMap can transparently use
    # cylindrical coordinates
    @property
    def r(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    #: phi position, i.e. angle wrt the x=0 axis in the xy plane (radians)
    @property
    def phi(self):
        return math.atan2(self.y, self.x)


class Hit(StrictModel):
    """A hit results from, within individual channel, fluctation above baseline.

    These are be clustered into ordinary peaks later. This is commonly
    called a 'hit' in particle physics detectors.  Very generally, a hit is
    made every time that the data recorded for one channel flucates above
    baseline.
    """
    #: Channel in which this peak was found
    channel = 0

    #: Index in the event at which this peak has its maximum.
    index_of_maximum = 0

    #: Time (since start of event in ns) of hit's center of gravity
    center = 0.0

    left = 0                 #: Index of left bound (inclusive) of peak.
    right = 0                #: Index of right bound (INCLUSIVE!!) of peak

    @property
    def length(self):
        return self.right - self.left + 1

    area = 0.0                  #: Area of the peak in photoelectrons

    #: Height of highest point in peak (in pe/bin)
    height = 0.0

    #: Noise sigma in pe/bin of pulse in which peak was found.
    #: Note: in Pulse the same number is stored in ADC-counts
    noise_sigma = 0.0

    #: Index of pulse (in event.pulses) in which peak was found
    found_in_pulse = 0

    #: Set to True if rejected by suspicious channel algorithm
    is_rejected = False

    #: Number of samples with ADC saturation in this hit
    n_saturated = 0


class Peak(StrictModel):
    """Peak

    A peak will be, e.g., S1 or S2.
    """
    #: Type of peak (e.g., 's1', 's2', ...)
    type = 'unknown'

    #: Detector in which the peak was found, e.g. tpc or veto
    detector = 'none'

    ##
    #  Hit, area, and saturation data
    ##

    #: Peaks in individual channels that make up this peak
    hits = ListField(Hit)

    #: Array of areas in each PMT.
    area_per_channel = np.array([], dtype='float64')

    #: Area of the pulse in photoelectrons. Includes only contributing pmts in the right detector.
    #: For XDP matching rightmost sample is not included in area integral.
    area = 0.0

    #: Fraction of area in the top array
    area_fraction_top = 0.0

    #: Number of hits in the peak, per channel
    hits_per_channel = np.array([], dtype=np.int16)

    #: Total channels which contribute to the peak
    n_contributing_channels = 0

    #: Total number of hits in the peak
    n_hits = 0

    #: Fraction of hits in the top array
    hits_fraction_top = 0.0

    #: Number of samples with ADC saturation in this peak, per channel
    n_saturated_per_channel = np.array([], dtype=np.int16)

    #: Total number of samples with ADC saturation threshold in all channels in this peak
    n_saturated_samples = 0

    #: Total number of channels in the peakwhich have at least one saturated hit
    n_saturated_channels = 0

    @property
    def does_channel_contribute(self):
        return self.area_per_channel > 0

    @property
    def contributing_channels(self):
        return np.where(self.does_channel_contribute)[0]

    ##
    # Time distribution information
    ##

    left = 0                 #: Index of left bound (inclusive) in event.
    right = 0                #: Index of right bound (INCLUSIVE) in event.

    #: Weighted (by hit area) mean of hit times (since event start)
    hit_time_mean = 0.0

    #: Weighted (by hit area) std of hit times
    hit_time_std = 0.0

    #: Central ange of peak (hit-only) sum waveform which includes fraction of area.
    #: e.g. range_50p_area = distance (in time) between point of 25% area and 75% area (interpolated between samples)
    range_20p_area = 0.0
    range_50p_area = 0.0
    range_90p_area = 0.0

    ##
    # Spatial pattern information
    ##

    #: List of reconstructed positions (instances of :class:`pax.datastructure.ReconstructedPosition`)
    reconstructed_positions = ListField(ReconstructedPosition)

    def get_reconstructed_position_from_algorithm(self, algorithm):
        """Return reconstructed position found by algorithm, or None if the peak doesn't have one"""
        for rp in self.reconstructed_positions:
            if rp.algorithm == algorithm:
                return rp
        return None

    #: Weighted root mean square deviation of top hitpattern (cm)
    top_hitpattern_spread = 0.0

    #: Weighted root mean square deviation of bottom hitpattern (cm)
    bottom_hitpattern_spread = 0.0

    ##
    # Signal / noise info
    ##

    #: Weighted (by area) mean hit amplitude / noise level in that hit's channel
    mean_amplitude_to_noise = 0.0

    ##
    # Sum-waveform properties
    ##

    #: The peak's sum waveform in pe/bin
    #: The peak's center of gravity is always in the center of the array.
    sum_waveform = np.array([], dtype=np.float32)

    #: For tpc peaks, the peak's sum waveform in the top array only. Aligned with the sum waveform.
    sum_waveform_top = np.array([], dtype=np.float32)

    #: Index in the event's sum waveform at which this peak has its maximum.
    index_of_maximum = 0

    #: Time at which the peak's sum waveform has its center of gravity.
    center_time = 0.0

    #: Height of sum waveform (in pe/bin)
    height = 0.0

    def compute_properties_from_hits(self, config):
        """Set basic peak properties from the peak's hits.
        Can't be in a plugin because the hit content of the peaks changes during clustering / noise rejection,
        but these algorithms need access to these properties... so you want to do this several times.
        We always assume the hits are sorted from left to right.
        """
        last_top_ch = np.max(np.array(config['channels_top']))
        self.area_per_channel = np.zeros(config['n_channels'], dtype='float64')
        self.hits_per_channel = np.zeros(config['n_channels'], dtype=np.int16)
        self.n_saturated_per_channel = np.zeros(config['n_channels'], dtype=np.int16)
        if len(self.hits) == 0:
            raise ValueError("Can't compute properties of an empty peak!")

        # Compute basic properties of peak, needed later in the plugin
        # For speed it would be better to compute as much as possible later..
        self.left = self.hits[0].left
        self.right = self.hits[0].right
        hit_areas = []
        hit_times = []
        for hit in self.hits:
            self.area_per_channel[hit.channel] += hit.area
            self.left = min(self.left, hit.left)
            self.right = max(self.right, hit.right)
            self.hits_per_channel[hit.channel] += 1
            self.n_saturated_per_channel[hit.channel] += hit.n_saturated
            # Add the hit height / noise sigma to mean amplitude to noise
            # We weigh by hit area, so multipy by it now, and will divide by self.area later
            self.mean_amplitude_to_noise += hit.area * hit.height / hit.noise_sigma
            hit_areas.append(hit.area)
            hit_times.append(hit.center)

        self.area = np.sum(self.area_per_channel)
        self.n_saturated_samples = np.sum(self.n_saturated_per_channel)
        self.n_saturated_channels = len(np.where(self.n_saturated_per_channel)[0])
        self.n_contributing_channels = len(self.contributing_channels)
        self.mean_amplitude_to_noise /= self.area

        # Compute top fraction
        self.area_fraction_top = np.sum(self.area_per_channel[:last_top_ch + 1]) / self.area
        self.hits_fraction_top = np.sum(self.hits_per_channel[:last_top_ch + 1]) / self.area

        # Compute timing quantities
        self.hit_time_mean, self.hit_time_std = utils.weighted_mean_variance(hit_times, hit_areas)
        self.hit_time_std **= 0.5  # Convert variance to std

        if self.n_contributing_channels == 0:
            raise RuntimeError("Every peak should have at least one contributing channel... what's going on?")
        elif self.n_contributing_channels == 1:
            self.type = 'lone_hit'


class SumWaveform(StrictModel):
    """Class used to store sum (filtered or not) waveform information.
    """

    #: Name of the filter used (or 'none')
    name_of_filter = 'none'
    #: Name of this sum waveform
    name = 'none'
    #: Name of the detector this waveform belongs to (e.g. tpc or veto)
    detector = 'none'

    #: Array of PMT numbers included in this waveform
    channel_list = np.array([], dtype=np.uint16)

    #: Array of samples, units of pe/bin.
    samples = np.array([], dtype=np.float32)

    def is_filtered(self):
        if self.name_of_filter != 'none':
            return True
        else:
            return False


class Pulse(StrictModel):
    """A DAQ pulse

    A DAQ pulse can also be thought of as a pulse in a PMT.  Remember that this is
    inverted.
    """

    #: Start time of this pulse: samples
    #:
    #: Units are samples. This nonnegative number starts at zero and is an integer because
    #: it's an index.
    left = INT_NAN

    #: Stoptime of this pulse within event
    #:
    #: Units are samples and this time is inclusive of last sample.  This nonnegative number
    #: starts at zero and is an integer because it's an index.
    right = INT_NAN

    #: Channel number the pulse belongs to
    channel = INT_NAN

    #: Raw wave data (numpy array of int16, ADC counts)
    raw_data = np.array([], np.int16)

    #: Baseline in ADC counts relative to reference baseline -- but float!
    baseline = float('nan')

    #: Maximum amplitude reached in the pulse (in ADC counts above baseline)
    maximum = float('nan')

    #: Minimum amplitude (in ADC counts above baseline, so should be negative)
    minimum = float('nan')

    #: Noise sigma for this pulse (in ADC counts - but float!)
    noise_sigma = float('nan')

    @property
    def length(self):
        return self.right - self.left + 1

    def __init__(self, **kwargs):
        """Initialize an pulse
        You must specify at least:
         - left (first index)
        And one of
         - raw_data (numpy array of samples)
         - right (last index)
        """
        super().__init__(**kwargs)

        if self.channel == INT_NAN:
            raise ValueError("Must specify channel to init Pulse")

        # Determine right from raw_data if needed
        # Don't want right as a property, we want it to be saved...
        if self.right == INT_NAN:
            if not len(self.raw_data):
                raise ValueError('Must have right or raw_data to init Pulse')
            self.right = self.left + len(self.raw_data) - 1


class Interaction(StrictModel):
    """An interaction in the TPC, reconstructed from a pair of S1 and S2 peaks.
    """
    #: The S1 peak of the interaction
    s1 = Peak()

    #: The S2 peak of the interaction
    s2 = Peak()

    ##
    # Position information
    ##

    #: The reconstructed position of the interaction
    x = float('nan')  #: x position (cm)
    y = float('nan')  #: y position (cm)

    #: goodness-of-fit parameter of s2 hitpattern to x,y position reconstructed by PosRecChiSquareGamma
    xy_posrec_goodness_of_fit = float('nan')

    #: number of degrees of freedom calculated with PosRecChiSquareGamma
    xy_posrec_ndf = float('nan')

    #: Algorithm used for xy position reconstructed
    xy_posrec_algorithm = 'none'

    #: drift time (ns) between S1 and S2
    drift_time = float('nan')

    #: z position (cm), calculated from drift time
    z = float('nan')

    #: r position (cm)
    @property
    def r(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    #: phi position, i.e. angle wrt the x=0 axis in the xy plane (radians)
    @property
    def phi(self):
        return math.atan2(self.y, self.x)

    def set_position(self, recpos):
        """Sets the x, y position of the interaction
        based on a :class:`pax.datastructure.ReconstructedPosition` object"""
        self.x = recpos.x
        self.y = recpos.y
        self.xy_posrec_algorithm = recpos.algorithm
        self.xy_posrec_ndf = recpos.ndf
        self.xy_posrec_goodness_of_fit = recpos.goodness_of_fit

    ##
    # Interaction properties
    ##

    #: Multiplicative correction to s1 area based on position (due to LCE variations)
    s1_area_correction = 1.0

    @property
    def corrected_s1_area(self):
        return self.s1.area * self.s1_area_correction

    #: Multiplicative correction to s2 area based on position (due to electron lifetime and LCE variations)
    s2_area_correction = 1.0

    @property
    def corrected_s2_area(self):
        return self.s2.area * self.s2_area_correction

    #: log10(corrected S2 area / corrected S1 area). Used for recoil type discrimination.
    @property
    def log_cs2_cs1(self):
        return np.log10(self.corrected_s2_area / self.corrected_s1_area)

    # #: Estimated interaction energy in keV (ee? nr?)
    # energy = float('nan')
    #
    # #: Estimated error on interaction energy in keV (ee? nr?)
    # energy_error = float('nan')

    ##
    # Likelihoods
    ##

    #: Likelihood of s2 width based on z position
    s2_width_likelihood = float('nan')

    #: Likelihood of s1 asymmetry based on z position
    s1_asymmetry_likelihood = float('nan')

    #: Likelihood of s1 hitpattern spread based on z position
    s1_hitpattern_spread_likelihood = float('nan')


class Event(StrictModel):
    """Event class

    Stores high level information about the triggered event.
    """
    dataset_name = 'Unknown'  # The name of the dataset this event belongs to

    # A nonnegative integer that uniquely identifies the event within the
    # dataset.
    event_number = 0

    #: Number of channels in the event
    #: Has to be the same as n_channels in config, provided here for deserialization ease
    n_channels = INT_NAN

    #: Integer start time of the event in nanoseconds
    #:
    #: Time that the first sample starts. This is a 64-bit number that follows the
    #: UNIX clock. Or rather, it starts from January 1, 1970.  This must be an integer
    #: because floats have rounding that result in imprecise times.  You could
    #: think of this as the time of the earliest sample.
    start_time = 0

    #: Integer stop time of the event in nanoseconds
    #
    #: This stop time includes the last recorded sample.  Therefore, it's the right
    #: edge of the last sample.  This is a 64-bit integer for the reasons explained
    #: in 'start_time'.
    stop_time = 0

    #: Time duration of a sample (in pax units, i.e. ns)
    #: For V1724 digitizers (e.g. XENON), this is 10 nanoseconds always.
    #: This is also in config, but we need it here too, to convert between event duration and length in samples
    #: Must be an int for same reason as start_time and stop_time
    #: DO NOT set to 10 ns as default, otherwise no way to check if it was given to constructor!
    sample_duration = 0

    #: A list of :class:`pax.datastructure.Interaction` objects.
    interactions = ListField(Interaction)

    #: A list of :class:`pax.datastructure.Peak` objects.
    peaks = ListField(Peak)

    #: Temporary list of hits -- will be shipped off to peaks later
    all_hits = ListField(Hit)

    #: A list :class:`pax.datastructure.SumWaveform` objects.
    sum_waveforms = ListField(SumWaveform)

    #: A list of all pulses in the event (containing instances of the Pulse class)
    #: An pulse holds a stream of samples in one channel, as provided by the digitizer.
    pulses = ListField(Pulse)

    #: Number of noise pulses (pulses without any hits found) per channel
    noise_pulses_in = np.array([], dtype=np.int)

    #: Was channel flagged as suspicious?
    is_channel_suspicious = np.array([], dtype=np.bool)

    #: Number of hits rejected in the suspicious channel algorithm
    n_hits_rejected = np.array([], dtype=np.int)

    def __init__(self, n_channels, start_time, partial=False, **kwargs):

        # Start time is mandatory, so it is not in kwargs
        kwargs['start_time'] = start_time
        kwargs['n_channels'] = n_channels

        # Model's init must be called first, else we can't store attributes
        # This will store all of the kwargs as attrs
        # We don't pass length, it's not an attribute that can be set
        super().__init__(**{k: v for k, v in kwargs.items() if k != 'length'})

        # Cheat to init stop_time from length and duration
        if 'length' in kwargs and self.sample_duration and not self.stop_time:
            self.stop_time = int(self.start_time + kwargs['length'] * self.sample_duration)

        if not self.stop_time or not self.sample_duration:
            raise ValueError("Cannot initialize an event with an unknown length: " +
                             "pass sample_duration and either stop_time or length")

        if self.duration() <= 0:
            raise ValueError("Negative event duration")

        # Initialize numpy arrays -- need to have n_channels and self.length
        self.noise_pulses_in = np.zeros(n_channels, dtype=np.int)
        self.n_hits_rejected = np.zeros(n_channels, dtype=np.int)
        self.is_channel_suspicious = np.zeros(n_channels, dtype=np.bool)

    @classmethod
    def empty_event(cls):
        """Returns an empty example event: for testing purposes only!!
        """
        return Event(n_channels=1, start_time=10, length=1, sample_duration=int(10 * units.ns))

    def duration(self):
        """Duration of event window in units of ns
        """
        return self.stop_time - self.start_time

    def get_sum_waveform_names(self):
        """Get list of the names of sum waveform objects
        Deprecated -- for Xerawdp matching only
        """
        return [sw.name for sw in self.sum_waveforms]

    def get_sum_waveform(self, name):
        """Get sum waveform object by name
        Deprecated -- for Xerawdp matching only
        """
        for sw in self.sum_waveforms:
            if sw.name == name:
                return sw

        raise RuntimeError("SumWaveform %s not found" % name)

    def length(self):
        """Number of samples in the event
        """
        return int(self.duration() / self.sample_duration)

    def s1s(self, detector='tpc', sort_key='area', reverse=True):  # noqa
        """List of S1 (scintillation) signals in this event

        Returns a list of :class:`pax.datastructure.Peak` objects
          whose type is 's1', and
          who are in the detector specified by the 'detector' argument (unless detector='all')
        The returned list is sorted DESCENDING (i.e. reversed!) by the key sort_key (default area)
        unless you pass reverse=False, then it is ascending.
        """
        return self.get_peaks_by_type('s1', sort_key=sort_key, reverse=reverse, detector=detector)

    def S1s(self, *args, **kwargs):
        return self.s1s(*args, **kwargs)

    def s2s(self, detector='tpc', sort_key='area', reverse=True):  # noqa
        """List of S2 (ionization) signals in this event

        Returns a list of :class:`pax.datastructure.Peak` objects
          whose type is 's2', and
          who are in the detector specified by the 'detector' argument (unless detector='all')
        The returned list is sorted DESCENDING (i.e. reversed!) by the key sort_key (default area)
        unless you pass reverse=False, then it is ascending.
        """
        return self.get_peaks_by_type(desired_type='s2', sort_key=sort_key, reverse=reverse, detector=detector)

    def S2s(self, *args, **kwargs):
        return self.s2s(*args, **kwargs)

    def get_peaks_by_type(self, desired_type='all', detector='tpc', sort_key='area', reverse=True):
        """Helper function for retrieving only certain types of peaks
        Returns a list of :class:`pax.datastructure.Peak` objects
          whose type is desired_type, and
          who are in the detector specified by the 'detector' argument (unless detector='all')
        The returned list is sorted DESCENDING (i.e. reversed!) by the key sort_key (default area)
        unless you pass reverse=False, then it is ascending (normal sort order).
        """
        # Extract only peaks of a certain type
        peaks = []
        for peak in self.peaks:
            if detector != 'all':
                if peak.detector != detector:
                    continue
            if desired_type != 'all' and peak.type.lower() != desired_type:
                continue
            peaks.append(peak)

        # Sort the peaks by your sort key
        peaks = sorted(peaks,
                       key=lambda x: getattr(x, sort_key),
                       reverse=reverse)

        return peaks


def _explain(class_name):
    x = inspect.getmembers(class_name,
                           predicate=inspect.isdatadescriptor)

    for a, b in x:
        if a.startswith('_'):
            continue
        print(a, b.__doc__)


if __name__ == '__main__':
    _explain(Peak)
    _explain(Event)

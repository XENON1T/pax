"""Data structure for pax

This is meant to be a fixed data structure that people can use to access
physically meaningful variables.  For example, S2.

NOTE: This class is stable within major releases.  Do not change any variable
names of functionality between major releases.  You may add variables in minor
releases.  Patch releases cannot modify this.
"""
import operator
from collections import namedtuple
import numpy as np
import six

from pax import units
from pax.data_model import Model, StrictModel, ListField
if six.PY3:
    long = int


INT_NAN = -99999    # Do not change without talking to me. -Tunnell 12/3/2015 ... and me. -Jelle 05/08/2015


class ConfidenceTuple(StrictModel):
    """Confidence tuple

    Stores the information of a confidence level of a reconstructed position
    """
    level = float('nan')
    x0 = float('nan')
    y0 = float('nan')
    dx = float('nan')
    dy = float('nan')

    at_edge = False

    @property
    def failed(self):
        return np.isnan(self.x0) or np.isnan(self.y0) or np.isnan(self.dx) or np.isnan(self.dy)


class ReconstructedPosition(StrictModel):
    """Reconstructed position

    Each reconstruction algorithm creates one of these.
    """
    x = float('nan')  #: x position (cm)
    y = float('nan')  #: y position (cm)

    #: For 3d-position reconstruction algorithms, the z-position (cm)
    #: This is NOT related to drift time, which is an interaction-level quantity!
    z = float('nan')

    #: Goodness-of-fit of hitpattern to position (provided by PosRecTopPatternFit)
    #: For PosRecThreedPatternFit, the 3d position goodness-of-fit.
    goodness_of_fit = float('nan')

    #: Number of degrees of freedom used in goodness-of-fit calculation
    ndf = float('nan')

    #: Name of algorithm which provided this position
    algorithm = 'none'

    #: Confidence_levels
    # error_matrix = np.array([], dtype=np.float64)
    confidence_tuples = ListField(ConfidenceTuple)

    # For convenience: cylindrical coordinates
    # Must be properties so InterpolatingDetectorMap can transparently use
    # cylindrical coordinates
    @property
    def r(self):
        """Radial position"""
        return np.sqrt(self.x ** 2 + self.y ** 2)

    #: phi position, i.e. angle wrt the x=0 axis in the xy plane (radians)
    @property
    def phi(self):
        """Angular position (radians, origin at positive x-axis)"""
        return np.arctan2(self.y, self.x)


class Hit(StrictModel):
    """A significant upwards fluctuation in a channel,
    usually indicative of one or more detected photo-electrons.
    These are clustered into groups called peaks later.

    Inside pax this Hit class is rarely used, for performance reasons;
    instead we build a numpy dtype from this declaration, and use it in arrays of hits.
    """
    #: Channel in which this hit was found
    channel = 0

    #: Index/sample in the event at which this hit has its maximum.
    index_of_maximum = 0

    #: Time (since start of event in ns) of hit's center of gravity
    center = 0.0

    #: Weighted sum of absolute deviation (in ns) of hit waveform from hit center
    sum_absolute_deviation = 0.0

    left = 0                 #: Index/sample of left bound (inclusive) of hit in event.
    right = 0                #: Index/sample of right bound (INCLUSIVE!!) of hit in event.

    left_central = 0         #: Index/sample of the left bound of the central (above treshold) part of the hit in event
    right_central = 0        #: Index/sample of the right bound of the central (above treshold) part of the hit in event

    @property
    def length(self):
        """Length of the hit (in samples)"""
        return self.right - self.left + 1

    area = 0.0               #: Area of the hit in photoelectrons

    #: Height of highest point in hit (in pe/sample)
    height = 0.0

    #: Noise sigma in pe/sample of the pulse in which the hit was found.
    #: Note: in Pulse the same number is stored in ADC-counts.
    noise_sigma = 0.0

    #: Index of pulse (in event.pulses) in which hit was found
    found_in_pulse = 0

    #: Set to True if rejected by suspicious channel algorithm.
    #: This means the hit should be disregarded by clustering algorithms.
    is_rejected = False

    #: Number of samples in this hit where the ADC saturates
    n_saturated = 0


class TriggerSignal(StrictModel):
    """A simplified peak class which is produced by the trigger
    Like Hit, this class not actually used. So default here are meaningless (except for type spec),
    np.zeros just sets all to zero.

    All times below are in ns since the start of the run only while we are in the trigger,
    but in ns since the start of the event as soon as the event is built. The conversion is done in
    MongoDB.ReadUntriggeredFiller.
    """

    #: "Type" of the signal.
    #:   0: unknown TPC
    #:   1: TPC S1 candidates
    #:   2: TPC S2 candidates
    #:   106: muon veto
    type = 0

    #: Did this signal cause a trigger?
    trigger = False

    #: Time at which the signal starts.
    left_time = 0

    #: Time at which the signal ends
    right_time = 0

    #: Number of pulses contributing to this signal
    n_pulses = 0

    #: Number of channels contributing at least 1 pulse to this signal
    n_contributing_channels = 0

    #: Mean pulse time start time
    time_mean = 0.0

    #: Root mean square deviation of pulse start times
    time_rms = float('nan')

    #: Total area in the signal (gain-weighted sum of integrals found by Kodiaq pulse integration)
    area = float('nan')

    # x = float('nan')
    # y = float('nan')


class Peak(StrictModel):
    """A group of nearby hits across one or more channels.
    Peaks will be classified as e.g. s1, s2, lone_hit, unknown, coincidence
    """
    #: Type of peak (e.g., 's1', 's2', ...):
    #: NB 'lone_hit' incicates one or more hits in a single channel. Use lone_hit_channel to retrieve that channel.
    type = 'unknown'

    #: Detector in which the peak was found, e.g. tpc or veto
    detector = 'none'

    ##
    #  Hit, area, and saturation data
    ##

    #: The hits that make up this peak. To save space, we usually only store the hits for s1s in the root file.
    #: Do not rely on the order of hits in this field!!
    #: For the root output, this gets converted back to a list of Hit classes (then to a vector of c++ Hit objects)
    hits = np.array([], dtype=Hit.get_dtype())

    #: Total areas of all hits per PMT (pe).
    area_per_channel = np.array([], dtype='float64')

    #: Total area of all hits across all PMTs (pes).
    #: In XerawdpImitation mode, rightmost sample is not included in area integral.
    area = 0.0

    #: Fraction of area in the top PMTs
    area_fraction_top = 0.0

    #: Multiplicative correction on S2 due to LCE variations
    s2_spatial_correction = 1.0

    #: Multiplicative correction on S2 top due to LCE variations
    s2_top_spatial_correction = 1.0

    #: Multiplicative correction on S2 bottom due to LCE variations
    s2_bottom_spatial_correction = 1.0

    #: Multiplicative correction on S2 due to saturation
    s2_saturation_correction = 1.0

    #: Number of hits in the peak, per channel (that is, it's an array with index = channel number)
    hits_per_channel = np.array([], dtype=np.int16)

    #: Number of channels which contribute to the peak
    n_contributing_channels = 0

    #: Number of channels in the top array contributing to the peak
    n_contributing_channels_top = 0

    #: Total number of hits in the peak
    n_hits = 0

    #: Fraction of hits in the top array
    hits_fraction_top = 0.0

    #: Number of samples with ADC saturation in this peak, per channel
    n_saturated_per_channel = np.array([], dtype=np.int16)

    @property
    def is_channel_saturated(self):
        """Boolean array of n_channels which indicates if there was ADC saturation in any hit
        in that channel during the peak"""
        return self.n_saturated_per_channel > 0

    @property
    def saturated_channels(self):
        """List of channels which contribute hits with saturated channels in this peak"""
        return np.where(self.n_saturated_per_channel > 0)[0]

    #: Total number of samples with ADC saturation threshold in all channels in this peak
    n_saturated_samples = 0

    #: Total number of channels in the peakw hich have at least one saturated hit
    n_saturated_channels = 0

    #: If the peak is a lone_hit: the channel the hit is / hits are in
    lone_hit_channel = INT_NAN

    # Area of the largest hit in the peak
    largest_hit_area = float('nan')

    # Channel of the largest hit in the peak
    largest_hit_channel = INT_NAN

    @property
    def does_channel_contribute(self):
        """Boolean array of n_channels which tells you if the channel contributes any hit"""
        return self.area_per_channel > 0

    @property
    def contributing_channels(self):
        """List of channels which contribute one or more hits to this peak"""
        return np.where(self.does_channel_contribute)[0]

    #: Number of channels that have a hit maximum within a short (configurable) window around the peak's sum
    #: waveform maximum.
    tight_coincidence = INT_NAN

    ##
    # Time distribution information
    ##

    left = 0                 #: Index/sample of left bound (inclusive) in event.
    right = 0                #: Index/sample of right bound (INCLUSIVE) in event.

    #: Weighted (by hit area) mean of hit times (since event start) [ns]
    hit_time_mean = 0.0

    #: Weighted (by hit area) std of hit times [ns]
    hit_time_std = 0.0

    #: Central range of peak (hit-only) sum waveform which includes a given decile (0-10) of area [ns].
    #: e.g. range_area_decile[5] = range of 50% area = distance (in time) between point
    #: of 25% area and 75% area (with boundary samples added fractionally).
    #: First element (0) is always zero, last element (10) is the full range of the peak.
    range_area_decile = np.zeros(11, dtype=np.float)

    #: Time (ns) from the area decile point to the area midpoint.
    #: If you want to know the time until some other point (say the sum waveform maximum),
    #: just add the difference between that point and the area midpoint.
    area_decile_from_midpoint = np.zeros(11, dtype=np.float)

    @property
    def range_50p_area(self):
        return self.range_area_decile[5]

    @property
    def range_90p_area(self):
        return self.range_area_decile[9]

    @property
    def full_range(self):
        return self.range_area_decile[10]

    #: Time at which the peak reaches 50% of its area (with the central sample considered fractionally)
    area_midpoint = 0.0

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

    def get_position_from_preferred_algorithm(self, algorithm_list):
        """Return reconstructed position by the first algorithm in list,
        unless it doesn't exist or is a nan position, then moves on to further algorithms."""
        for algo in algorithm_list:
            rp = self.get_reconstructed_position_from_algorithm(algo)
            if rp is not None and not np.isnan(rp.x):
                return rp
        else:
            raise ValueError("Could not find any position from the chosen algorithms: %s" % algorithm_list)

    #: Weighted-average distance of top array hits from weighted mean hitpattern center on top array (cm)
    top_hitpattern_spread = float('nan')

    #: Weighted-average distance of bottom array hits from weighted mean hitpattern center on bottom array (cm)
    bottom_hitpattern_spread = float('nan')

    ##
    # Signal / noise info
    ##

    #: Weighted (by area) mean hit amplitude / noise level in that hit's channel
    mean_amplitude_to_noise = 0.0

    #: Number of pulses without hits in the event overlapping (in time; at least partially) with this peak.
    #: Includes channels from other detectors (since veto and tpc cables could influence each other)
    n_noise_pulses = 0

    ##
    # Sum-waveform properties
    ##

    #: Cut-out of the peak's sum waveform in pe/bin
    #: The peak's center of gravity is always in the center of the array.
    sum_waveform = np.array([], dtype=np.float32)

    #: For tpc peaks, the peak's sum waveform in the top array only. Aligned with the sum waveform.
    sum_waveform_top = np.array([], dtype=np.float32)

    #: Index/sample in the event's sum waveform at which this peak has its maximum.
    index_of_maximum = 0

    #: Time since start of the event at which the peak's sum waveform has its center of gravity [ns].
    center_time = 0.0

    #: Height of sum waveform (in pe/bin)
    height = 0.0

    ##
    # Clustering record
    ##

    #: Best goodness of split observed inside the peak
    interior_split_goodness = float('nan')

    #: Area fraction of the smallest of the two halves considered in the best split inside the peak
    #: (i.e. the one corresponding to interior_split_goodness)
    interior_split_fraction = float('nan')

    #: Goodness of split of last split that was used to construct this peak (if split did occur).
    birthing_split_goodness = float('nan')

    #: Area of this peak / area of parent peak it was split from (if split did occur)
    birthing_split_fraction = float('nan')


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


class Pulse(Model):
    """A region of raw digitizer data.
    For DAQs with zero-length encoding or self-triggering, there will be several of these per channel per event.

    Remember that PMT signals show as downward fluctuations in raw digitizer data.
    In processed data output, only pulses contributing hits to S1s are stored (except in LED mode), and the
    raw_data field is never stored.
    """

    #: Start index/sample of this pulse (inclusive)
    #: This refers to a hypothetical array containing the event waveform information.
    #: For example, 0 is the first sample that could exist in the event, 1 the second, etc.
    left = INT_NAN

    #: Stop index/sample of this pulse (INCLUSIVE). For example, this is 1 for a 2-sample event.
    right = INT_NAN

    #: Channel number the pulse belongs to.
    channel = INT_NAN

    #: Raw wave data (numpy array of int16, raw ADC counts).
    #: This is just what you get from the ADC, not corrected or modified in any way
    raw_data = np.array([], np.int16)

    #: Baseline, in ADC counts relative to reference baseline -- but float!
    #: This is computed on the first few samples at the start of the pulse.
    baseline = float('nan')

    #: Baseline at end of the pulse (relative to reference) - baseline at start of pulse (relative to reference)
    #: E.g. if this is positive, the baseline increased in signal-like direction = decreased in raw ADC.
    #: Depending on the PMTs used and baselining length, this may not reflect an actual baseline increase but just
    #: a long tail of a photoelectron pulse
    baseline_increase = float('nan')

    #: Maximum amplitude reached in the pulse (in ADC counts above pulse baseline)
    maximum = float('nan')

    #: Minimum amplitude (in ADC counts above pulse baseline, so should be negative)
    minimum = float('nan')

    #: Noise sigma for this pulse (in ADC counts - but float!)
    noise_sigma = float('nan')

    #: Number of hits found in this pulse
    n_hits_found = 0

    # Hitfinder threshold used in ADC counts
    hitfinder_threshold = 0

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
        Model.__init__(self, **kwargs)

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
    #: Index (in event.peaks) of the s1 peak of this interaction
    s1 = INT_NAN

    #: Index (in event.peaks) of the s2 peak of this interaction
    s2 = INT_NAN

    ##
    # Position information
    ##

    #: The reconstructed position of the interaction, (r,z) corrected
    x = float('nan')  #: x position of the interaction (cm)
    y = float('nan')  #: y position of the interaction (cm)

    #: goodness-of-fit parameter of s2 hitpattern to x,y position reconstructed by PosRecTopPatternFit
    xy_posrec_goodness_of_fit = float('nan')

    #: number of degrees of freedom calculated with PosRecTopPatternFit
    xy_posrec_ndf = float('nan')

    #: Algorithm used to reconstruct xy position
    xy_posrec_algorithm = 'none'

    #: Drift time (ns) between s1 and s2
    drift_time = float('nan')

    #: z position (cm) of the interaction,
    #: This starts from - (drift time - drift time of gate) * drift velocity, then applies the (r,z) correction
    z = float('nan')

    #: r position (cm), (r,z) corrected
    @property
    def r(self):
        return np.sqrt(self.x ** 2 + self.y ** 2)

    #: phi position, i.e. angle wrt the x=0 axis in the xy plane (radians).
    @property
    def phi(self):
        return np.arctan2(self.y, self.x)

    #: R correction that has been added to r. Subtract it from interaction.r to recover the uncorrected r position
    r_correction = 0.0

    #: Z correction that has been added to z. Subtract it from interaction.z to recover the uncorrected z position
    z_correction = 0.0

    ##
    # Interaction properties
    ##

    #: Multiplicative correction on S1 due to LCE variations
    s1_spatial_correction = 1.0

    #: Multiplicative correction on S1 due to saturation
    s1_saturation_correction = 1.0

    #: Multiplicative correction on S2 due to electron lifetime
    s2_lifetime_correction = 1.0

    #: Final multiplicative correction to s1 area based on position (due to LCE variations and saturation)
    s1_area_correction = 1.0

    @property
    def corrected_s1_area(self):
        return self.s1.area * self.s1_area_correction

    #: Final multiplicative correction to s2 area based on position
    # (due to electron lifetime, LCE variations and saturation)
    s2_area_correction = 1.0

    @property
    def corrected_s2_area(self):
        return self.s2.area * self.s2_area_correction

    #: log10(corrected S2 area / corrected S1 area). Used for recoil type discrimination.
    @property
    def log_cs2_cs1(self):
        return np.log10(self.corrected_s2_area / self.corrected_s1_area)

    ##
    # Likelihoods
    ##

    #: Goodness of fit of S1 pattern to interaction's (x, y, drift_time)
    s1_pattern_fit = float('nan')

    #: Goodness of fit of S1 pattern to interaction's (x, y, drift_time),
    #: computed using number of hits/channel instead of area/channel
    s1_pattern_fit_hits = float('nan')


class Event(StrictModel):
    """Object holding high-level information about a triggered event,
    and list of objects (such as Peak, Hit and Pulse) containing lower-level information.
    """
    #: The name of the dataset this event belongs to
    dataset_name = 'Unknown'

    #: A nonnegative integer that uniquely identifies the event within the dataset.
    event_number = 0

    #: Internal number used for multiprocessing, no physical meaning.
    block_id = -1

    #: Total number of channels in the event (whether or not they see anything).
    #: Has to be the same as n_channels in config, provided here for deserialization ease.
    n_channels = INT_NAN

    #: Integer start time of the event in nanoseconds since January 1, 1970.
    #: This is the time that the first sample starts.
    #: NB: don't do floating-point arithmetic on 64-bit integers such as these,
    #: floats have rounding that result in loss of precision.
    start_time = long(0)

    #: Integer stop time of the event in nanoseconds since January 1, 1970.
    #: This is the time that the last sample ends.
    #: NB: don't do floating-point arithmetic on 64-bit integers such as these,
    #: floats have rounding that result in loss of precision.
    stop_time = long(0)

    #: Time duration of a sample (in ns).
    #: For V1724 digitizers (e.g. XENON), this is 10 nanoseconds.
    #: This is also in config, but we need it here too, to convert between event duration and length in samples
    #: Must be an int for same reason as start_time and stop_time
    #: DO NOT set to 10 ns as default, otherwise no way to check if it was given to constructor!
    sample_duration = 0

    #: A list of :class:`pax.datastructure.Interaction` objects.
    interactions = ListField(Interaction)

    #: A list of :class:`pax.datastructure.Peak` objects.
    peaks = ListField(Peak)

    #: Array of trigger signals contained in the event
    trigger_signals = np.array([], dtype=TriggerSignal.get_dtype())

    #: Array of all hits found in event
    #: These will get grouped into peaks during clustering. New hits will be added when peaks are split.
    #: NEVER rely upon the order of hits in this field! It depends on lunar phase and ambient pressure.
    #: This is usually emptied before output (but not in LED mode)
    all_hits = np.array([], dtype=Hit.get_dtype())

    #: A list :class:`pax.datastructure.SumWaveform` objects.
    sum_waveforms = ListField(SumWaveform)

    #: A list of :class:`pax.datastructure.Interaction` objects.
    #: A pulse holds a stream of samples in one channel provided by the digitizer.
    #: To save space, only the pulses contributing hits to S1s are kept in the output (but not in LED mode)
    #: The order of this field cannot be changed after the hitfinder, since hits have a found_in_pulse field
    #: referring to the index of a pulse in this field.
    pulses = ListField(Pulse)

    #: Number of pulses per channel
    n_pulses_per_channel = np.array([], dtype=np.int16)

    #: Total number of pulses
    n_pulses = 0

    #: Number of noise pulses (pulses without any hits found) per channel
    noise_pulses_in = np.array([], dtype=np.int16)

    #: Number of lone hits per channel BEFORE suspicious channel hit rejection.
    #: lone_hit is a peak type (sorry, confusing...) indicating just one contributing channel.
    #: Use this to check / calibrate the suspicious channel hit rejection.
    lone_hits_per_channel_before = np.array([], dtype=np.int16)

    #: Number of lone hits per channel AFTER suspicious channel hit rejection.
    #: lone_hit is a peak type (sorry, confusing...) indicating just one contributing channel
    lone_hits_per_channel = np.array([], dtype=np.int16)

    #: Was channel flagged as suspicious?
    is_channel_suspicious = np.array([], dtype=np.bool)

    #: Number of hits rejected per channel in the suspicious channel algorithm
    n_hits_rejected = np.array([], dtype=np.int16)

    def __init__(self, n_channels, start_time, **kwargs):

        # Start time is mandatory, so it is not in kwargs
        kwargs['start_time'] = start_time
        kwargs['n_channels'] = n_channels

        # Model's init must be called first, else we can't store attributes
        # This will store all of the kwargs as attrs
        # We don't pass length, it's not an attribute that can be set
        StrictModel.__init__(self, **{k: v for k, v in kwargs.items() if k != 'length'})

        # Cheat to init stop_time from length and duration
        if 'length' in kwargs and self.sample_duration and not self.stop_time:
            self.stop_time = int(self.start_time + kwargs['length'] * self.sample_duration)

        if not self.stop_time or not self.sample_duration:
            raise ValueError("Cannot initialize an event with an unknown length: " +
                             "pass sample_duration and either stop_time or length")

        if self.duration() <= 0:
            raise ValueError("Nonpositive event duration %s!" % self.duration())

        # Initialize numpy arrays -- need to have n_channels and self.length
        self.n_pulses_per_channel = np.zeros(n_channels, dtype=np.int16)
        self.noise_pulses_in = np.zeros(n_channels, dtype=np.int16)
        self.n_hits_rejected = np.zeros(n_channels, dtype=np.int16)
        self.is_channel_suspicious = np.zeros(n_channels, dtype=np.bool)
        self.lone_hits_per_channel_before = np.zeros(n_channels, dtype=np.int16)
        self.lone_hits_per_channel = np.zeros(n_channels, dtype=np.int16)

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

    def s1s(self, detector='tpc', sort_key=('tight_coincidence', 'area'), reverse=True):  # noqa
        """List of S1 (scintillation) signals in this event
        In the ROOT class output, this returns a list of integer indices in event.peaks
        Inside pax, returns a list of :class:`pax.datastructure.Peak` objects
          whose type is 's1', and
          who are in the detector specified by the 'detector' argument (unless detector='all')
        The returned list is sorted DESCENDING (i.e. reversed!) by the key sort_key (default area)
        unless you pass reverse=False, then it is ascending.
        """
        return self.get_peaks_by_type('s1', sort_key=sort_key, reverse=reverse, detector=detector)

    def S1s(self, *args, **kwargs):
        """See s1s"""
        return self.s1s(*args, **kwargs)

    def s2s(self, detector='tpc', sort_key='area', reverse=True):  # noqa
        """List of S2 (ionization) signals in this event
        In the ROOT class output, this returns a list of integer indices in event.peaks.
        Inside pax, returns a list of :class:`pax.datastructure.Peak` objects
          whose type is 's2', and
          who are in the detector specified by the 'detector' argument (unless detector='all')
        The returned list is sorted DESCENDING (i.e. reversed!) by the key sort_key (default area)
        unless you pass reverse=False, then it is ascending.
        """
        return self.get_peaks_by_type(desired_type='s2', sort_key=sort_key, reverse=reverse, detector=detector)

    def S2s(self, *args, **kwargs):
        """See s2s"""
        return self.s2s(*args, **kwargs)

    @property
    def main_s1(self):
        """Return the S1 of the primary interaction, or if that does not exist, the largest S1 in the tpc.
        Returns None if neither exist"""
        if self.interactions:
            return self.peaks[self.interactions[0].s1]
        else:
            try:
                return self.s1s()[0]
            except IndexError:
                return None

    @property
    def main_s2(self):
        """Return the S2 of the primary interaction, or if that does not exist, the largest S2 in the tpc.
        Returns None if neither exist"""
        if self.interactions:
            return self.peaks[self.interactions[0].s2]
        else:
            try:
                return self.s2s()[0]
            except IndexError:
                return None

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
        if isinstance(sort_key, (str, bytes)):
            sort_key = [sort_key]
        peaks = sorted(peaks,
                       key=operator.attrgetter(*sort_key),
                       reverse=reverse)

        return peaks


# An event proxy object which can hold arbitrary data
# but still has an event_number attribute
# The decoders for and WriteZipped & Readzipped knows what to do with this,
# other code will be fooled into treating it as a normal event
# (except the explicit event class checks in ProcessPlugin of course,
#  these have to be disabled by as needed using do_output_check and do_input_check)
EventProxy = namedtuple('EventProxy', ['data', 'event_number', 'block_id'])


def make_event_proxy(event, data, block_id=None):
    if block_id is None:
        block_id = event.block_id
    return EventProxy(data=data, event_number=event.event_number, block_id=block_id)

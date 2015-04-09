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
from pax import units

# To turn off type-checking, replace StrictModel with Model
# This will improve performance, but use at your own risk
from pax.data_model import StrictModel, Model


INT_NAN = -99999  # Do not change without talking to me. -Tunnell 12/3/2015


class ReconstructedPosition(StrictModel):
    """Reconstructed position

    Each reconstruction algorithm creates one of these.
    """
    x = 0.0  #: x position (cm)
    y = 0.0  #: y position (cm)

    #: goodness-of-fit parameter generated with PosRecChiSquareGamma
    goodness_of_fit = 0.0
    # : number of degrees of freedom calculated with PosRecChiSquareGamma
    ndf = 0.0

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

    @property
    def phi(self):
        return math.atan2(self.y, self.x)


class ChannelPeak(Model):
    """Peaks found in individual channels

    These are be clustered into ordinary peaks later. This is commonly
    called a 'hit' in particle physics detectors.
    """
    channel = 0              #: Channel in which this peak was found
    #: Index in the event at which this peak has its maximum.
    index_of_maximum = 0

    left = 0                 #: Index of left bound (inclusive) of peak.
    right = 0                #: Index of right bound (INCLUSIVE!!) of peak

    @property
    def length(self):
        return self.right - self.left + 1

    area = 0.0                  #: Area of the peak in photoelectrons
    #: Height of highest point in peak (in pe/bin) in unfiltered waveform
    height = 0.0
    #: Noise sigma in pe/bin of pulse in which peak was found.
    noise_sigma = 0.0
    # note: in Pulse the same number is stored in ADC-counts

    #: Index of pulse (in event.occurrences) in which peak was found
    found_in_pulse = 0

    #: Set to True if rejected by suspicious channel algorithm
    is_rejected = False


class Peak(StrictModel):
    """Peak

    A peak will be, e.g., S1 or S2.
    """

    #: Peaks in individual channels that make up this peak
    channel_peaks = (ChannelPeak,)

    type = 'unknown'        #: Type of peak (e.g., 's1', 's2', ...)
    detector = 'none'       #: e.g. tpc or veto

    left = 0                 #: Index of left bound (inclusive) in event.
    right = 0                #: Index of right bound (INCLUSIVE!!) in event.
    # For XDP matching rightmost sample is not in integral, so you could say
    # it is exclusive then.

    #: Array of areas in each PMT.
    area_per_channel = np.array([], dtype='float64')

    #: Area of the pulse in photoelectrons.
    #:
    #: Includes only contributing pmts (see later) in the right detector
    area = 0.0

    #: Fraction of area in the top array
    area_fraction_top = 0.0

    #: Does a PMT see 'something significant'? (thresholds configurable)
    does_channel_contribute = np.array([], dtype=np.bool)

    @property
    def contributing_channels(self):
        return np.where(self.does_channel_contribute)[0]

    @property
    def number_of_contributing_channels(self):
        """ Number of PMTS which see something significant (depends on settings) """
        return len(self.contributing_channels)

    does_channel_have_noise = np.array([], dtype=np.bool)

    @property
    def noise_channels(self):
        return np.where(self.does_channel_have_noise)[0]

    @property
    def number_of_noise_channels(self):
        """ Number of channels which have noise during this peak """
        return len(self.noise_channels)

    #: Returns a list of reconstructed positions
    #:
    #: Returns an :class:`pax.datastructure.ReconstructedPosition` class.
    reconstructed_positions = (ReconstructedPosition,)

    #: Weighted root mean square deviation of top hitpattern (cm)
    top_hitpattern_spread = 0.0

    #: Weighted root mean square deviation of bottom hitpattern (cm)
    bottom_hitpattern_spread = 0.0

    #: Median absolute deviation of photon arrival times (in ns)
    # Deprecate this?
    median_absolute_deviation = 0.0

    #: Weighted (by area) mean and rms of hit maxima (in ns; mean is relative to event start)
    hit_time_mean = 0.0
    hit_time_std = 0.0

    ##
    # Deprecated sum-waveform stuff
    ##
    #: Index in the event's sum waveform at which this peak has its maximum.
    index_of_maximum = 0
    #: Height of highest point in peak (in pe/bin)
    height = 0.0


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
    samples = np.array([], dtype=np.float64)

    def is_filtered(self):
        if self.name_of_filter != 'none':
            return True
        else:
            return False


class Occurrence(StrictModel):
    """A DAQ occurrence

    A DAQ occurrence can also be thought of as a pulse in a PMT.
    """

    #: Starttime of this occurence within event
    #:
    #: Units are samples.  This nonnegative number starts at zero and is an integere because
    #: it's an index.
    left = INT_NAN

    #: Stoptime of this occurence within event
    #:
    #: Units are samples and this time is inclusive of last sample.  This nonnegative number
    #: starts at zero and is an integere because it's an index.
    right = INT_NAN

    #: Channel the occurrence belongs to (integer)
    channel = INT_NAN

    #: Maximum amplitude (in ADC counts; float) in unfiltered waveform
    #: Will remain nan if channel's gain is 0
    #: baseline_correction, if any, has been substracted
    # TODO: may not be equal to actual occurrence height, baseline correction
    # is computed on filtered wv. :-(
    height = float('nan')

    #: Noise sigma for this occurrence (in ADC counts)
    #: Computed in the filtered channel waveform
    #: Will remain nan unless occurrence is processed by smallpeakfinder
    noise_sigma = float('nan')

    #: Baseline (in ADC counts, but float!) relative to configured reference baseline
    #: Will remain nan if occurrence is not processed by hitfinder
    baseline = float('nan')

    #: Raw wave data (in ADC counts, NOT pe/bin!; numpy array of int16)
    raw_data = np.array([], np.int16)

    @property
    def length(self):
        return self.right - self.left + 1

    def __init__(self, **kwargs):
        """Initialize an occurrence
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


class Event(StrictModel):
    """Event class

    Stores high level information about the triggered event.
    """
    dataset_name = 'Unknown'  # The name of the dataset this event belongs to
    # A nonnegative integer that uniquely identifies the event within the
    # dataset.
    event_number = 0

    #: Integer start time of the event in nanoseconds
    #:
    #: Time that the first sample starts. This is a 64-bit number that follows the
    #: UNIX clock. Or rather, it starts from January 1, 1970.  This must be an integer
    #: because floats have rounding that result in imprecise times.
    start_time = 0

    #: Integer stop time of the event in nanoseconds
    #
    #: This stop time includes the last recorded sample.  Therefore, it's the right
    #: edge of the last sample.  This is a 64-bit integer for the reasons explained
    #: in 'start_time'.
    stop_time = 0

    #: Time duration of a sample (in pax units, i.e. ns)
    #: This is also in config, but we need it here too, to convert between event duration and length in samples
    #: Must be an int for same reason as start_time and stop_time
    #: DO NOT set to 10 ns as default, otherwise no way to check if it was given to constructor!
    sample_duration = 0

    user_float_0 = 0.0  # : Unused float (useful for developing)
    user_float_1 = 0.0  # : Unused float (useful for developing)
    user_float_2 = 0.0  # : Unused float (useful for developing)
    user_float_3 = 0.0  # : Unused float (useful for developing)
    user_float_4 = 0.0  # : Unused float (useful for developing)

    #: List of peaks
    #:
    #: Returns a list of :class:`pax.datastructure.Peak` classes.
    peaks = (Peak,)

    #: Temporary list of channel peaks -- will be shipped off to peaks later
    all_channel_peaks = (ChannelPeak,)

    #: Returns a list of sum waveforms
    #:
    #: Returns an :class:`pax.datastructure.SumWaveform` class.
    sum_waveforms = (SumWaveform,)

    #: A python list of all occurrences in the event (containing instances of the Occurrence class)
    #: An occurrence holds a stream of samples in one channel, as provided by the digitizer.
    occurrences = (Occurrence,)

    #: Number of noise pulses (pulses without any hits found) per channel
    noise_pulses_in = np.array([], dtype=np.int)

    #: Was channel flagged as suspicious?
    is_channel_suspicious = np.array([], dtype=np.bool)

    #: Number of hits rejected in the suspicious channel algorithm
    n_hits_rejected = np.array([], dtype=np.int)

    def __init__(self, n_channels, start_time, partial=False, **kwargs):

        # Start time is mandatory, so it is not in kwargs
        kwargs['start_time'] = start_time

        # Model's init must be called first, else we can't store attributes
        # This will store all of the kwargs as attrs
        # We don't pass length, it's not an attribute that can be set
        super().__init__(**{k: v for k, v in kwargs.items() if k != 'length'})

        # Cheat to init stop_time from length and duration
        if 'length' in kwargs and self.sample_duration and not self.stop_time:
            self.stop_time = int(
                self.start_time + kwargs['length'] * self.sample_duration)

        if not self.stop_time or not self.sample_duration:
            raise ValueError("Cannot initialize an event with an unknown length: " +
                             "pass sample_duration and either stop_time or length")

        if not partial:
            # Initialize numpy arrays -- need to have n_channels and self.length
            # TODO: don't initialize these if is already in kwargs
            # TODO: better yet, make an alternate init or something?
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
        """Get list of the names of waveforms
        """
        return [sw.name for sw in self.sum_waveforms]

    def get_sum_waveform(self, name):
        """Get waveform for name
        """
        for sw in self.sum_waveforms:
            if sw.name == name:
                return sw

        raise RuntimeError("SumWaveform %s not found" % name)

    def length(self):
        """Number of samples for the sum waveform
        """
        return int(self.duration() / self.sample_duration)

    def S1s(self, detector='tpc', sort_key='area', reverse=True):  # noqa
        """List of S1 (scintillation) signals

        Returns an :class:`pax.datastructure.Peak` class.
        """
        return self._get_peaks_by_type('s1', sort_key, reverse, detector)

    def S2s(self, detector='tpc', sort_key='area', reverse=True):  # noqa
        """List of S2 (ionization) signals

        Returns an :class:`pax.datastructure.Peak` class.
        """
        return self._get_peaks_by_type('s2', sort_key, reverse, detector)

    def _get_peaks_by_type(self, desired_type, sort_key, reverse, detector='tpc'):
        """Helper function for retrieving only certain types of peaks

        You shouldn't be using this directly.
        """
        # Extract only peaks of a certain type
        peaks = []
        for peak in self.peaks:
            if peak.detector is not 'all':
                if peak.detector != detector:
                    continue
            if peak.type.lower() != desired_type:
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

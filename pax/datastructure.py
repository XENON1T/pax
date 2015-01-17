"""Data structure for pax

This is meant to be a fixed data structure that people can use to access
physically meaningful variables.  For example, S2.

NOTE: This class is stable within major releases.  Do not change any variable
names of functionality between major releases.  You may add variables in minor
releases.  Patch releases cannot modify this.
"""

import inspect
import math

import numpy as np

from pax import units
# To turn off type-checking for all models, replace the line below with
# from pax.data_model import Model
# This will improve performance a bit (+ ~10% running time), but use at your own risk
from pax.data_model import StrictModel as Model


class ReconstructedPosition(Model):

    """Reconstructed position

    Each reconstruction algorithm creates one of these.
    """
    x = 0.0  #: x position (cm)
    y = 0.0  #: y position (cm)
    z = 0.0  #: z position (cm)

    goodness_of_fit = 0.0  #: goodness-of-fit parameter generated with PosRecChiSquareGamma
    ndf = 0.0  # : number of degrees of freedom calculated with PosRecChiSquareGamma

    #: For this reconstructed peak, index of maximum value within sum waveform.
    index_of_maximum = 0

    #: Name of algorithm used for computation
    algorithm = 'none'

    # : Errors - currently not used
    # error_matrix = np.array([], dtype=np.float64)

    # For convenience: cylindrical coordinates
    # Must be properties so InterpolatingDetectorMap can transparently use cylindrical coordinates
    @property
    def r(self):
        return math.sqrt(self.x ** 2 + self.y ** 2)

    @property
    def phi(self):
        return math.atan2(self.y, self.x)


class ChannelPeak(Model):

    """Peaks found in individual channels
    These are be clustered into ordinary peaks later
    """
    channel = 0              #: Channel in which this peak was found
    index_of_maximum = 0     #: Index in the event at which this peak has its maximum.

    left = 0                 #: Index of left bound (inclusive) of peak.
    right = 0                #: Index of right bound (INCLUSIVE!!) of peak

    @property
    def length(self):
        return self.right - self.left + 1

    area = 0.0                   #: Area of the peak in photoelectrons
    height = 0.0                 #: Height of highest point in peak (in pe/bin)
    noise_sigma = 0.0            #: StDev of the noise in the occurrence (in pe/bin) where we found this peak


class Peak(Model):

    """Peak object"""

    ##
    #   Fields present in all peaks
    ##

    left = 0                 #: Index of left bound (inclusive) in event.
    right = 0                #: Index of right bound (INCLUSIVE!!) in event.
    # For XDP matching rightmost sample is not in integral, so you could say it is exclusive then.

    #: Area of the pulse in photoelectrons.
    #:
    #: Includes only contributing pmts (see later) in the right detector
    area = 0.0

    type = 'unknown'        #: Type of peak (e.g., 's1', 's2', ...)
    subtype = 'unknown'     #: Subtype of peak
    detector = 'none'  #: e.g. tpc or veto

    #: Does a PMT see 'something significant'? (thresholds configurable)
    does_channel_contribute = np.array([], dtype=np.bool)

    @property
    def contributing_channels(self):
        return np.where(self.does_channel_contribute)[0]

    @property
    def number_of_contributing_channels(self):
        """ Number of PMTS which see something significant (depends on settings) """
        return len(self.contributing_channels)

    #: Array of areas in each PMT.
    area_per_channel = np.array([], dtype='float64')

    #: Returns a list of reconstructed positions
    #:
    #: Returns an :class:`pax.datastructure.ReconstructedPosition` class.
    reconstructed_positions = (ReconstructedPosition,)

    ##
    #   Fields present in sum-waveform peaks
    ##

    index_of_maximum = 0           #: Index in the event's sum waveform at which this peak has its maximum.
    index_of_filtered_maximum = 0  #: same, but maximum in filtered (for S2) sum waveform

    height = 0.0                 #: Height of highest point in peak (in pe/bin)
    height_filtered = 0.0        #: Height of highest point in filtered waveform of peak (in pe/bin)

    central_area = 0.0           #: Area in the central part of the peak (used for classification)

    # Note these are floats -- the widths get interpolated
    full_width_half_max = 0.0               #: Full width at half maximum in samples
    full_width_tenth_max = 0.0              #: Full width at tenth of maximum in samples
    full_width_half_max_filtered = 0.0      #: Full width at half of maximum in samples, in filtered waveform
    full_width_tenth_max_filtered = 0.0     #: Full width at tenth of maximum in samples, in filtered waveform

    #: Array of squared signal entropies in each PMT.
    entropy_per_channel = np.array([], dtype='float64')

    ##
    #   Fields present in peaks from single-channel peakfinding
    ##

    #: Peaks in individual channels that make up this peak
    channel_peaks = (ChannelPeak,)

    does_channel_have_noise = np.array([], dtype=np.bool)

    @property
    def noise_channels(self):
        return np.where(self.does_channel_have_noise)[0]

    @property
    def number_of_noise_channels(self):
        """ Number of channels which have noise during this peak """
        return len(self.noise_channels)

    #: Variables indicating width of peak
    mean_absolute_deviation = 0.0
    # standard_deviation = 0.0
    # half_area_range = 0.0
    # tenth_area_range = 0.0


class SumWaveform(Model):

    """Class used to store sum (filtered or not) waveform information.
    """

    #: Name of the filter used (or 'none')
    name_of_filter = 'none'
    #: Name of this sum waveform
    name = 'none'
    detector = 'none'  #: Name of the detector this waveform belongs to (e.g. tpc or veto)

    #: Array of PMT numbers included in this waveform
    channel_list = np.array([], dtype=np.uint16)

    #: Array of samples, units of pe/bin.
    samples = np.array([], dtype=np.float64)

    def is_filtered(self):
        if self.name_of_filter != 'none':
            return True
        else:
            return False


class Occurrence(Model):

    """A DAQ occurrence
    """

    #: First index (inclusive; integer)
    left = 0

    #: Last index (inclusive; integer)
    right = 0

    #: Channel the occurrence belongs to (integer)
    channel = 0

    #: Maximum amplitude (in pe/bin; float)
    #: Will remain nan if channel's gain is 0
    #: baseline_correction, if any, has been substracted
    # TODO: may not be equal to actual occurrence height,
    # baseline correction is computed on 2-sample filtered wv. :-(
    height = float('nan')

    #: Noise sigma for this occurrence
    #: Will remain nan unless occurrence is processed by smallpeakfinder
    noise_sigma = float('nan')

    #: Baseline (in ADC counts, but float!)
    #: Will remain nan if channel's gain is 0
    digitizer_baseline_used = float('nan')

    #: Baseline correction computed by FindSmallpeaks (in pe/bin)
    #: Will remain nan if channel is not processed by FindSmallPeaks
    baseline_correction = float('nan')

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

        # Determine right from raw_data if needed
        # Don't want right as a property, we want it to be saved...
        if self.right == 0:
            if not len(self.raw_data):
                raise ValueError('Must have right or raw_data to init Occurrence')
            self.right = self.left + len(self.raw_data) - 1


class Event(Model):

    """Event class
    """
    dataset_name = 'Unknown'  # The name of the dataset this event belongs to
    event_number = 0    # A nonnegative integer that uniquely identifies the event within the dataset.

    #: Start time of the event (time at which the first sample STARTS)
    #:
    #: This is a 64-bit number in units of ns that follows the UNIX clock.
    #: Or rather, it starts from January 1, 1970.
    #: Must be an int!! large floats don't support precision arithmetic
    start_time = 0

    #: Stop time of the event (time at which the last sample ENDS).
    #:
    #: This is a 64-bit number in units of ns that follows the UNIX clock.
    #: Or rather, it starts from January 1, 1970.
    stop_time = 0

    #: Time duration of a sample (in pax units, i.e. ns)
    #: This is also in config, but we need it here too, to convert between event duration and length in samples
    #: Must be an int for same reason as start_time and stop_time
    sample_duration = int(10 * units.ns)

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

    #: A 2D array of all the PMT waveforms, units of pe/bin.
    #:
    #: The first index is the PMT number (starting from zero), and the second
    #: index is the sample number.  This must be a numpy array.  To access the
    #: waveform for a specific PMT such as PMT 10, you can do::
    #:
    #:     event.channel_waveforms[10]
    #:
    #: which returns a 1D array of samples.
    #:
    #: The data type is a float32 since these numbers are already baseline
    #: and gain corrected.
    channel_waveforms = np.array([], dtype=np.float64)  # : Array of samples.

    #: A python list of all occurrences in the event (containing instances of the Occurrence class)
    #: An occurrence holds a stream of samples in one channel, as provided by the digitizer.
    occurrences = (Occurrence,)

    #: List of channels which showed an increased dark rate
    #: Declared as basefield as we want to store a list (it will get appended to constantly)
    is_channel_bad = np.array([], dtype=np.bool)

    def __init__(self, n_channels, start_time, **kwargs):

        # Start time is mandatory, so it is not in kwargs
        kwargs['start_time'] = start_time

        # Model's init must be called first, else we can't store attributes
        # This will store all of the kwargs as attrs
        # We don't pass length, it's not an attribute that can be set
        super().__init__(**{k: v for k, v in kwargs.items() if k != 'length'})

        # Cheat to init stop_time from length and duration
        if 'length' in kwargs and self.sample_duration and not self.stop_time:
            self.stop_time = int(self.start_time + kwargs['length'] * self.sample_duration)

        if not self.length:
            raise ValueError("Cannot initialize an event with an unknown length: " +
                             "pass either stop_time or length and sample_duration")

        # Initialize numpy arrays -- need to have n_channels and self.length
        # This is the main reason for having Event.__init__
        self.channel_waveforms = np.zeros((n_channels, self.length()))
        self.is_channel_bad = np.zeros(n_channels, dtype=np.bool)

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

    def S1s(self, detector='tpc', sort_key='area', reverse=True):
        """List of S1 (scintillation) signals

        Returns an :class:`pax.datastructure.Peak` class.
        """
        return self._get_peaks_by_type('s1', sort_key, reverse, detector)

    def S2s(self, detector='tpc', sort_key='area', reverse=True):
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

    def get_occurrences_between(self, left, right, strict=False):
        """Returns all occurrences that overlap with [left, right]
        If strict=True, only returns occurrences that are not outside [left, right]
        """
        if strict:
            return [oc for oc in self.occurrences if oc.left >= left and oc.right <= right]
        else:
            return [oc for oc in self.occurrences if oc.left <= right and oc.right >= left]


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

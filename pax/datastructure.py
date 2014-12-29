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
from pax.micromodels.models import Model
from pax.micromodels import fields as f
from pax.micromodels.fields import IntegerField, FloatField, StringField


class ReconstructedPosition(Model):

    """Reconstructed position

    Each reconstruction algorithm creates one of these.
    """
    x = FloatField()  #: x position (cm)
    y = FloatField()  #: y position (cm)
    z = FloatField()  #: z position (cm)

    goodness_of_fit = FloatField()  #: goodness-of-fit parameter generated with PosRecChiSquareGamma
    ndf = FloatField() #: number of degrees of freedom calculated with PosRecChiSquareGamma

    #: For this reconstructed peak, index of maximum value within sum waveform.
    index_of_maximum = IntegerField()

    #: Name of algorithm used for computation
    algorithm = StringField(default='none')

    # : Errors - currently not used
    # error_matrix = f.NumpyArrayField(dtype=np.float64)

    # For convenience: cylindrical coordinates
    # Must be properties so InterpolatingDetectorMap can transparently use cylindrical coordinates
    @property
    def r(self):
        return math.sqrt(self.x**2 + self.y**2)

    @property
    def phi(self):
        return math.atan2(self.y, self.x)


class ChannelPeak(Model):

    """Peaks found in individual channels
    These are be clustered into ordinary peaks later
    """
    channel = IntegerField()              #: Channel in which this peak was found
    index_of_maximum = IntegerField()     #: Index in the event at which this peak has its maximum.

    left = IntegerField()                 #: Index of left bound (inclusive) of peak.
    right = IntegerField()                #: Index of right bound of peak

    area = FloatField()                   #: Area of the peak in photoelectrons
    height = FloatField()                 #: Height of highest point in peak (in pe/bin)
    noise_sigma = FloatField()            #: StDev of the noise in the occurrence (in pe/bin) where we found this peak


class Peak(Model):

    """Peak object"""

    ##
    #   Fields present in all peaks
    ##

    left = IntegerField()                       #: Index of left bound (inclusive) in event.
    right = IntegerField() #: Index of right bound (for Xdp matching: exclusive; otherwise: inclusive) in event.

    area = FloatField()                   #: Area of the pulse in photoelectrons. Only
                                          #: Includes only contributing pmts (see later) in the right detector

    type = StringField(default='unknown')   #: Type of peak (e.g., 's1', 's2', ...)
    detector = StringField(default='none')  #: e.g. tpc or veto

    #: Does a PMT see 'something significant'? (thresholds configurable)
    does_channel_contribute = f.NumpyArrayField(dtype=np.bool)

    @property
    def contributing_pmts(self):
        return np.where(self.does_channel_contribute)[0]

    @property
    def number_of_contributing_channels(self):
        """ Number of PMTS which see something significant (depends on settings) """
        return len(self.contributing_pmts)

    # Alias for backwards compatibility
    @property
    def coincidence_level(self):
        """ Number of PMTS which see something significant (depends on settings) """
        return self.number_of_contributing_channels

    #: Array of areas in each PMT.
    area_per_pmt = f.NumpyArrayField(dtype='float64')

    #: Returns a list of reconstructed positions
    #:
    #: Returns an :class:`pax.datastructure.ReconstructedPosition` class.
    reconstructed_positions = f.ModelCollectionField(default=[],
                                                     wrapped_class=ReconstructedPosition)


    ##
    #   Fields present in sum-waveform peaks
    ##

    index_of_maximum = IntegerField()           #: Index in the event's sum waveform at which this peak has its maximum.
    index_of_filtered_maximum = IntegerField()  #: same, but maximum in filtered (for S2) sum waveform

    height = FloatField()                 #: Height of highest point in peak (in pe/bin)
    height_filtered = FloatField()        #: Height of highest point in filtered waveform of peak (in pe/bin)

    central_area = FloatField()           #: Area in the central part of the peak (used for classification)

    full_width_half_max = FloatField()             #: Full width at half maximum in samples
    full_width_tenth_max = FloatField()            #: Full width at tenth of maximum in samples
    full_width_half_max_filtered = FloatField()    #: Full width at half of maximum in samples, in filtered waveform
    full_width_tenth_max_filtered = FloatField()   #: Full width at tenth of maximum in samples, in filtered waveform

    #: Array of squared signal entropies in each PMT.
    entropy_per_pmt = f.NumpyArrayField(dtype='float64')


    ##
    #   Fields present in peaks from single-channel peakfinding
    ##

    #: Peaks in individual channels that make up this peak
    channel_peaks = f.ModelCollectionField(default=[],
                                           wrapped_class=ChannelPeak)

    does_channel_have_noise = f.NumpyArrayField(dtype=np.bool)

    @property
    def number_of_noise_channels(self):
        """ Number of PMTS which see something significant (depends on settings) """
        return len(self.contributing_pmts)


    #: Variables indicating width of peak
    mean_absolute_deviation = FloatField()
    # standard_deviation = FloatField()
    # half_area_range = FloatField()
    # tenth_area_range = FloatField()


class Waveform(Model):

    """Class used to store sum (filtered or not) waveform information.
    """

    #: Name of the filter used (or 'none')
    name_of_filter = StringField(default='none')
    name = StringField(default='none')  #: e.g. top
    detector = StringField(default='none')  #: e.g. tpc or veto

    #: Array of PMT numbers included in this waveform
    pmt_list = f.NumpyArrayField(dtype=np.uint16)

    #: Array of samples, units of pe/bin.
    samples = f.NumpyArrayField(dtype=np.float64)

    def is_filtered(self):
        if self.name_of_filter != 'none':
            return True
        else:
            return False


class Event(Model):

    """Event class
    """

    dataset_name = StringField(default='Unknown')  # The name of the dataset this event belongs to
    event_number = IntegerField()    # A nonnegative integer that uniquely identifies the event within the dataset.

    #: Time duration of a sample in units of ns
    sample_duration = IntegerField(default=10*units.ns)

    #: Start time of the event.
    #:
    #: This is a 64-bit number in units of ns that follows the UNIX clock.
    #: Or rather, it starts from January 1, 1970.
    start_time = IntegerField()

    #: Stop time of the event.
    #:
    #: This is a 64-bit number in units of ns that follows the UNIX clock.
    #: Or rather, it starts from January 1, 1970.
    stop_time = IntegerField()

    user_float_0 = FloatField()  # : Unused float (useful for developing)
    user_float_1 = FloatField()  # : Unused float (useful for developing)
    user_float_2 = FloatField()  # : Unused float (useful for developing)
    user_float_3 = FloatField()  # : Unused float (useful for developing)
    user_float_4 = FloatField()  # : Unused float (useful for developing)

    #: List of peaks
    #:
    #: Returns a list of :class:`pax.datastructure.Peak` classes.
    peaks = f.ModelCollectionField(default=[], wrapped_class=Peak)
    #: Temporary list of channel peaks -- will be shipped off to peaks later
    all_channel_peaks = f.ModelCollectionField(default=[], wrapped_class=ChannelPeak)

    #: Returns a list of sum waveforms
    #:
    #: Returns an :class:`pax.datastructure.SumWaveform` class.
    waveforms = f.ModelCollectionField(default=[], wrapped_class=Waveform)

    #: A 2D array of all the PMT waveforms, units of pe/bin.
    #:
    #: The first index is the PMT number (starting from zero), and the second
    #: index is the sample number.  This must be a numpy array.  To access the
    #: waveform for a specific PMT such as PMT 10, you can do::
    #:
    #:     event.pmt_waveforms[10]
    #:
    #: which returns a 1D array of samples.
    #:
    #: The data type is a float32 since these numbers are already baseline
    #: and gain corrected.
    pmt_waveforms = f.NumpyArrayField(dtype=np.float64)  # : Array of samples.

    #: Occurrences
    #:
    #: An occurrence holds a stream of samples in one channel, as provided by the digitizer.
    #: The format of this field is a dict of lists of 2-tuples:
    #: {
    #:      channel_int : [ (start_index_int, samples_ndarray), (...,...), ...],
    #:      channel_int : ...,
    #:      ...
    #: }
    #: The first element of each 2-tuple is the index within the event (i.e.
    #: units of sample, e.g., 10 ns), the second a numpy array 16-bit signed integers,
    #: which represent the ADC counts.
    #:
    #: (This may get moved into the Input plugin base class, see issue #32)
    occurrences = f.BaseField()

    #: Used to be an interval tree (from PyIntervalTree) for occurrence lookup.
    #: Now just an 'IntervalTree' mock-up defined in dsputils, see there for usage
    #: TODO: document this properly once stable
    #: After BuildWaveforms, you should only use this, not event.occurrences
    occurrences_interval_tree = f.BaseField()

    #: List of channels which showed an increased dark rate
    #: Declared as basefield as we want to store a list (it will get appended to constantly)
    is_channel_bad = f.NumpyArrayField(dtype=np.bool)

    def event_duration(self):
        """Duration of event window in units of ns
        """
        return self.stop_time - self.start_time

    def get_waveform_names(self):
        """Get list of the names of waveforms
        """
        return [sw.name for sw in self.waveforms]

    def get_waveform(self, name):
        """Get waveform for name
        """
        for sw in self.waveforms:
            if sw.name == name:
                return sw

        raise RuntimeError("Waveform %s not found" % name)

    def length(self):
        """Number of samples for the sum waveform
        """
        return int(self.event_duration() / self.sample_duration)

    # TODO: should this return tpc peaks only? also veto? configurable for separate/both?
    def S1s(self, sort_key='area', reverse=True):
        """List of S1 (scintillation) signals

        Returns an :class:`pax.datastructure.Peak` class.
        """
        return self._get_peaks_by_type('s1', sort_key, reverse)

    def S2s(self, sort_key='area', reverse=True):
        """List of S2 (ionization) signals

        Returns an :class:`pax.datastructure.Peak` class.
        """
        return self._get_peaks_by_type('s2', sort_key, reverse=reverse)

    def _get_peaks_by_type(self, desired_type, sort_key, reverse):
        """Helper function for retrieving only certain types of peaks

        You shouldn't be using this directly.
        """
        # Extract only peaks of a certain type
        peaks = []
        for peak in self.peaks:
            if peak.type.lower() == desired_type:
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

"""Data structure for pax

This is meant to be a fixed data structure that people can use to access
physically meaningful variables.  For example, S2.

NOTE: This class is stable within major releases.  Do not change any variable
names of functionality between major releases.  You may add variables in minor
releases.  Patch releases cannot modify this.
"""

import inspect

import numpy as np

from pax.micromodels.models import Model
from pax.micromodels import fields as f
from pax.micromodels.fields import IntegerField, FloatField, StringField


class Peak(Model):

    """Peak object

    """
    area = FloatField()  #: Area of the pulse in photoelectrons
    index_of_maximum = IntegerField()  #: Index of maximum value within sum waveform.
    height = IntegerField()  #: Height of highest point in peak (in pe/bin)
    left = IntegerField()  #: Index of left bound (inclusive) in sum waveform.
    right = IntegerField()  #: Index of right bound (exclusive) in sum waveform.
    type = StringField(default='s1')  #: Type of peak (e.g., 'S1' or 'S2')

    #: Array of PMT numbers included in this peaks.
    #:
    #: This is added by PeakPostProcessing.MakeHitList
    pmt_list = f.NumpyArrayField(dtype=np.uint16)


class Waveform(Model):

    """Class used to store sum (filtered or not) waveform information.
    """

    name_of_filter = StringField(default='none')  #: Name of the filter used (or 'none')
    name = StringField(default='none')  #: e.g. top

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

    event_number = IntegerField()
    """An integer number that uniquely identifies the event within the dataset.

    Always positive."""

    #: Time duration of a sample in units of ns
    sample_duration = IntegerField(default=10)

    #: Start time of the event.
    #:
    #: This is a 64-bit number in units of ns that follows the UNIX clock.
    #: Or rather, it starts from January 1, 1970.
    event_start = IntegerField()

    #: Stop time of the event.
    #:
    #: This is a 64-bit number in units of ns that follows the UNIX clock.
    #: Or rather, it starts from January 1, 1970.
    event_stop = IntegerField()

    user_float_0 = FloatField()  # : Unused float (useful for developing)
    user_float_1 = FloatField()  # : Unused float (useful for developing)
    user_float_2 = FloatField()  # : Unused float (useful for developing)
    user_float_3 = FloatField()  # : Unused float (useful for developing)
    user_float_4 = FloatField()  # : Unused float (useful for developing)

    #: List of peaks
    #:
    #: Returns a list of :class:`pax.datastructure.Peak` classes.
    peaks = f.ModelCollectionField(default=[], wrapped_class=Peak)

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
    #: Each one of these is like a 'pulse' from one channel.  This field is a
    #: dictionary where each key is an integer channel number.  Each value for
    #: the dictionary is a list, where each element of the list is a 2-tuple.
    #: The first element of the 2-tuple is the index within the event (i.e.
    #: units of sample, e.g., 10 ns) where this 'occurence'/pulse begins. The
    #: second element is a numpy array 16-bit signed integers, which represent
    #: the ADC counts.
    #:
    # : (This may get moved into the Input plugin baseclass, see issue #32)
    occurrences = f.BaseField()

    def event_duration(self):
        """Duration of event window in units of ns
        """
        return self.event_stop - self.event_start

    def get_waveform(self, name):
        """Get waveform for name
        """
        for sw in self.waveforms:
            if sw.name == name:
                return sw

        raise RuntimeError("Waveform not found")

    def length(self):
        """Number of samples for the sum waveform
        """
        return int(self.event_duration() / self.sample_duration)

    def S1s(self, sort_key='area'):
        """List of S1 (scintillation) signals

        Returns an :class:`pax.datastructure.Peak` class.
        """
        return self._get_peaks_by_type('s1', sort_key)

    def S2s(self, sort_key='area'):
        """List of S2 (ionization) signals

        Returns an :class:`pax.datastructure.Peak` class.
        """
        return self._get_peaks_by_type('s2', sort_key)

    def _get_peaks_by_type(self, desired_type, sort_key):
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
                       key=lambda x: getattr(x, sort_key))

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

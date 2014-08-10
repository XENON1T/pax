"""Data structure for pax

This is meant to be a fixed data structure that people can use to access
physically meaningful variables.  For example, S2.

NOTE: This class is stable within major releases.  Do not change any variable
names of functionality between major releases.  You may add variables in minor
releases.  Patch releases cannot modify this.
"""

import logging
import inspect

import numpy as np

from pax.models import Model
from pax import fields as f
from pax.fields import IntegerField, FloatField, ModelCollectionField, StringField

class Peak(Model):
    """Peak object

    Used for either S1 or S2.  Please do not add an "is_S1" or "is_S2" field because this
    can lead to discrepancies if an 'is_S1' is in the S2 list.  Then we need to check
    for that.  Maybe we can do something smart with a method?  See Issue #30.
    """
    area = FloatField()  #: Area of the pulse in photoelectrons
    index_of_maximum = IntegerField() #: Index of maximum value within sum waveform.
    height = IntegerField() #: Highest point in peak in units of ADC counts
    left = IntegerField() #: Index of left bound (inclusive) in sum waveform.
    right = IntegerField()  #: Index of right bound (exclusive) in sum waveform.


class Waveform(Model):
    """Class used to store sum (filtered or not) waveform information.
    """
    is_filtered = f.BooleanField()
    name_of_filter = StringField(default="none") #: Name of the filter used (or None)
    name = StringField(default="none") #: e.g. top

    #: Array of PMT numbers included in this waveform
    pmt_list = f.NumpyArrayField(dtype=np.float64)

    samples = f.NumpyArrayField(dtype=np.float64) #: Array of samples.

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

    user_float_0 = FloatField() #: Unused float (useful for developing)
    user_float_1 = FloatField() #: Unused float (useful for developing)
    user_float_2 = FloatField() #: Unused float (useful for developing)
    user_float_3 = FloatField() #: Unused float (useful for developing)
    user_float_4 = FloatField() #: Unused float (useful for developing)


    #: List of S1 (scintillation) signals
    #
    #: Returns an :class:`pax.datastructure.Peak` class.
    S1s = ModelCollectionField(default=[], wrapped_class=Peak)

    #: List of S2 (ionization) signals
    #
    #: Returns an :class:`pax.datastructure.Peak` class.
    S2s = ModelCollectionField(default=[], wrapped_class=Peak)

    #: Returns a list of sum waveforms
    #:
    #: Returns an :class:`pax.datastructure.SumWaveform` class.
    waveforms = ModelCollectionField(default=[], wrapped_class=Waveform)

    #: A 2D array of all the PMT waveforms.
    #:
    #: The first index is the PMT number (starting from zero), and the second
    #: index is the sample number.  This must be a numpy array.  To access the
    #: waveform for a specific PMT such as PMT 10, you can do::
    #:
    #:     event.pmt_waveforms[10]
    #:
    #:which returns a 1D array of samples.
    pmt_waveforms = f.NumpyArrayField(dtype=np.float64) #: Array of samples.

    #: Occurences
    #:
    #: Each one of these is like a 'pulse' from one channel.
    occurrences = f.BaseField()



    def event_duration(self):
        """Duration of event window in units of ns
        """
        return self.event_stop - self.event_start

    def length(self):
        """Number of samples for the sum waveform
        """
        return int(self.event_duration()/self.sample_duration)

    def get_waveform(self, name):
        """Get waveform for name
        """
        for sw in self.waveforms:
            if sw.name == name:
                return sw

        raise RuntimeError("Waveform not found")



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

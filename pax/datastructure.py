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

import pax.fields


#metaclass=OrderedClass

# class StorageObject(object):
#     class __metaclass__(type):
#         '''Creates the metaclass for Model. The main function of this metaclass
#         is to move all of fields into the _fields variable on the class.
#
#         '''
#         def __init__(cls, name, bases, attrs):
#             print('hi')
#             cls._clsfields = {}
#             for key, value in attrs.iteritems():
#                 if isinstance(value, pax.fields.BaseField):
#                     cls._clsfields[key] = value
#                     delattr(cls, key)
#
#         def __setattr__(self, key, value):
#             if key in self._fields:
#                 field = self._fields[key]
#                 field.populate(value)
#                 field._related_obj = self
#                 super(StorageObject, self).__setattr__(key, field.to_python())
#             raise AttributeError("%s does not exist" % key)
#
#         def __getattr__(self, key):
#             print('getattr')
#             if key in self._fields:
#                 field = self._fields[key]
#                 return field.to_python()
#             raise AttributeError("%s does not exist" % key)
#
#         @property
#         def _fields(self):
#             return dict(self._clsfields, **self._extra)
#
# class Event2(metaclass=StorageObject):
#     test = pax.fields.FloatField()

class BaseStorageObject(object):
    def __init__(self):
        self.log = logging.getLogger('Event')

        self._internal_values = {}



    def _fetch_variable(original_function):
        """Decorator
        """

        # This is the variable being accessed
        var = original_function.__name__

        def new_function(self):
            if var not in self._internal_values:
                self.log.debug(self._internal_values)
                self.log.fatal("Tried to access %s, but not set yet. Returning None." % var)
                raise RuntimeError("Variable %s not set" % var)
            return self._internal_values[var]

        new_function.__doc__ = original_function.__doc__
        return new_function

    def _set_variable(original_function):
        """Decorator
        """

        # This is the variable being accessed
        var = original_function.__name__

        def new_function(self, value):
            if isinstance(value, int):
                self.log.debug("Converting to int")
                value = np.int64(value)

            try:
                original_function(self, value)
            except Exception as e:
                self.log.exception(e)
                raise e
            finally:
                self.log.debug("Setting %s with value %s" % (var,
                                                             str(value)))
                self._internal_values[var] = value

        new_function.__doc__ = original_function.__doc__
        return new_function


class Event(BaseStorageObject):

    """The event class.

    This class defines the data structure for pax.  Within it are all of the
    high-level quantities that pax computes.  For example, the size of the S2
    signals.  This class is passed between the pax plugins and used as an
    information container.

    .. note:: This class should not include much logic since otherwise it
                      makes it harder to serialize.  Nevertheless, some methods are
                      provided for convenience.
    """

    def __init__(self):
        BaseStorageObject.__init__(self)

        self._internal_values['waveforms'] = []
        self._internal_values['S1s'] = []
        self._internal_values['S2s'] = []


    @property
    @BaseStorageObject._fetch_variable
    def sample_duration(self):
        """Time duraation of a sample in units of ns
        """
        pass

    @sample_duration.setter
    @BaseStorageObject._set_variable
    def sample_duration(self, value):
        pass

    @property
    @BaseStorageObject._fetch_variable
    def event_number(self):
        """An integer event number that uniquely identifies the event.

        Always positive.
        """
        pass

    @event_number.setter
    @BaseStorageObject._set_variable
    def event_number(self, value):
        if not isinstance(value, (np.int64)):
            raise RuntimeError("Wrong type; must be int np.int64.")
        elif value < 0:
            raise RuntimeError("Must be positive")

    @property
    @BaseStorageObject._fetch_variable
    def event_window(self):
        """Two numbers for the start and stop time of the event.

        This is a 64-bit number in units of ns that follows the UNIX clock.
        Or rather, it starts from January 1, 1970."""
        pass

    @event_window.setter
    @BaseStorageObject._set_variable
    def event_window(self, value):
        if not isinstance(value, (tuple)):
            raise RuntimeError("Wrong type; must be tuple.")
        elif len(value) != 2:
            raise RuntimeError("Wrong size; must be event_duration 2.")

    @property
    @BaseStorageObject._fetch_variable
    def pmt_waveforms(self):
        """A 2D array of all the PMT waveforms.

        The first index is the PMT number (starting from zero), and the second
        index is the sample number.  This must be a numpy array.  To access the
        waveform for a specific PMT such as PMT 10, you can do::

                event.pmt_waveforms[10]

        which returns a 1D array of samples.
        """
        pass

    @pmt_waveforms.setter
    @BaseStorageObject._set_variable
    def pmt_waveforms(self, value):
        if not isinstance(value, np.ndarray):
            raise RuntimeError("Wrong type; must be numpy array.")
        elif value.ndim != 2:
            raise RuntimeError("Wrong size; must be dimension 2.")
        elif value.shape[0] > 500:
            self.log.warning("Found %d channels, which seems high." % value.shape[0])
        elif value.shape[1] > 100000:
            self.log.warning("Found %d samples, which seems high." % value.shape[1])

    @property
    @BaseStorageObject._fetch_variable
    def S2s(self):
        """List of S2 (ionization) signals

        Returns an :class:`pax.datastructure.Peak` class.
        """
        pass

    @S2s.setter
    @BaseStorageObject._set_variable
    def S2s(self, value):
        if not isinstance(value, (list, tuple)):
            raise RuntimeError("Wrong type; must be ntuple.")
        for item in value:
            if not isinstance(item, Peak):
                raise ValueError("Must pass Peak class")

    @property
    @BaseStorageObject._fetch_variable
    def S1s(self):
        """List of S1 (scintillation) signals

        Returns an :class:`pax.datastructure.Peak` class.
        """
        pass

    @S1s.setter
    @BaseStorageObject._set_variable
    def S1s(self, value):
        if not isinstance(value, (list, tuple)):
            raise RuntimeError("Wrong type; must be ntuple.")
        for item in value:
            if not isinstance(value, Peak):
                raise ValueError("Must pass Peak class")

    @property
    @BaseStorageObject._fetch_variable
    def waveforms(self):
        """Returns a list of sum waveforms

        Returns an :class:`pax.datastructure.SumWaveform` class.
        """
        pass

    @waveforms.setter
    @BaseStorageObject._set_variable
    def waveforms(self, value):
        if not isinstance(value, (list, tuple)):
            raise RuntimeError("Wrong type; must be ntuple.")
        for item in value:
            if not isinstance(value, SumWaveform):
                raise ValueError("Must pass SumWaveform")


    @property
    @BaseStorageObject._fetch_variable
    def occurrences(self):
        """A dictionary of DAQ occurrences

        A DAQ occurrence is essentially a 'pulse' for one channel.
        """
        pass

    @occurrences.setter
    #@BaseStorageObject._set_variable
    def occurrences(self, value):
        self._internal_values['occurrences'] = value#pass

    @property
    def user_float_0(self):
        """Unused float (useful for developing)
        """
        pass

    @property
    def user_float_1(self):
        """Unused float (useful for developing)
        """
        pass

    @property
    def user_float_3(self):
        """Unused float (useful for developing)
        """
        pass

    @property
    def user_array_0(self):
        """Unused array (useful for developing)
        """
        pass

    @property
    def user_array_1(self):
        """Unused array (useful for developing)
        """
        pass

    @property
    def total_area(self):
        """Sum area of waveform
        """
        return self.pmt_waveforms.sum()

    @property
    def total_area_veto(self):
        """Summed area of waveform only for veto PMTs"""
        return self.pmt_waveforms[self._pmt_groupings['veto']]

    def event_duration(self):
        """Duration of event window in units of ns
        """
        return self.event_window[1] - self.event_window[0]

    def length(self):
        """Number of samples for the sum waveform
        """
        return int(self.event_duration()/self.sample_duration)

    def event_start(self):
        """Start time of the event in units of ns
        """
        return self.event_window[0]

    def event_end(self):
        """End time of event in units of ns
        """
        return self.event_window[1]

    def filtered_waveform(self):
        """Top and bottom filtered waveform method for convenience.
        """
        pass

    def append_waveform(self,
                            name,
                            pmt_list,
                            samples,
                            name_of_filter=None,
                            is_filtered=False):
        """Append a sum waveform to the event

        This is useful for adding sum waveforms without having to create your
        own SumWaveform class.
        """
        sw = SumWaveform()
        sw.name = name
        sw.pmt_list = pmt_list
        sw.samples = samples
        sw.name_of_filter = name_of_filter
        sw.is_filtered = is_filtered

        self.waveforms.append(sw)

    def get_waveform(self, name):
        """Get waveform for name
        """
        for sw in self.waveforms:
            if sw.name == name:
                return sw

        raise RuntimeError("Waveform not found")



class Peak(BaseStorageObject):

    """Class for S1 and S2 peak information
    """

    def __init__(self,
                 area,
                 index_of_maximum,
                 height,
                 left,
                 right):
        BaseStorageObject.__init__(self)
        self.area = area
        self.index_of_maximum = index_of_maximum
        self.height = height
        self.left = left
        self.right = right

    @property
    @BaseStorageObject._fetch_variable
    def area(self):
        """Area of the pulse in photoelectrons
        """
        pass

    @area.setter
    @BaseStorageObject._set_variable
    def area(self, value): pass

    @property
    @BaseStorageObject._fetch_variable
    def index_of_maximum(self):
        """position of each S1 peak"""
        pass

    @index_of_maximum.setter
    @BaseStorageObject._set_variable
    def index_of_maximum(self, value): pass

    @property
    @BaseStorageObject._fetch_variable
    def height(self):
        """Highest point in peak in units of ADC counts
        """
        pass

    @height.setter
    @BaseStorageObject._set_variable
    def height(self, value): pass

    @property
    @BaseStorageObject._fetch_variable
    def left(self):
        """Index of left bound (inclusive) in sum waveform.
        """
        pass

    @left.setter
    @BaseStorageObject._set_variable
    def left(self, value): pass

    @property
    @BaseStorageObject._fetch_variable
    def right(self):
        """Index of right bound (exclusive) in sum waveform.
        """
        pass

    @right.setter
    @BaseStorageObject._set_variable
    def right(self, value): pass



class SumWaveform(BaseStorageObject):
    """Class used to store sum (filtered or not) waveform information.
    """

    @property
    @BaseStorageObject._fetch_variable
    def is_filtered(self):
        """Boolean"""
        pass

    @is_filtered.setter
    @BaseStorageObject._set_variable
    def is_filtered(self, value): pass

    @property
    @BaseStorageObject._fetch_variable
    def name_of_filter(self):
        """Name of the filter used (or None)
        """
        pass

    @name_of_filter.setter
    @BaseStorageObject._set_variable
    def name_of_filter(self, value): pass

    @property
    @BaseStorageObject._fetch_variable
    def name(self):
        """e.g., top"""
        pass

    @name.setter
    @BaseStorageObject._set_variable
    def name(self, value): pass

    @property
    @BaseStorageObject._fetch_variable
    def pmt_list(self):
        """Array of PMT numbers included in this waveform"""
        pass

    @pmt_list.setter
    @BaseStorageObject._set_variable
    def pmt_list(self, value): pass

    @property
    @BaseStorageObject._fetch_variable
    def samples(self):
        """Array of samples.
        """
        # if not isinstance(value, (np.array)):
        #	raise RuntimeError("Wrong type; must be numpy array.")
        # elif value.ndim != 1:
        #		raise RuntimeError("Wrong size; must be dimension 1.")
        #	elif value.shape[0] > 100000:
        #	self.log.warning("Found %d samples, which seems high." % value.shape[1])
        pass

    @samples.setter
    @BaseStorageObject._set_variable
    def samples(self, value): pass


    #: Blah blah
    testtest = None


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

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
import collections


def _flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(_flatten(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Event(object):

    """The event class.

    This class defines the data structure for pax.  Within it are all of the
    high-level quantities that pax computes.  For example, the size of the S2
    signals.  This class is passed between the pax plugins and used as an
    information container.

    .. note:: This class should not include much logic since otherwise it
                      makes it harder to serialize.  Nevertheless, some methods are
                      provided for convienence.
    """

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

    # # Begining of properties ##

    @property
    @_fetch_variable
    def event_number(self):
        """An integer event number that uniquely identifies the event.

        Always positive.
        """
        pass

    @event_number.setter
    @_set_variable
    def event_number(self, value):
        if not isinstance(value, (int)):
            raise RuntimeError("Wrong type; must be int.")
        elif value < 0:
            raise RuntimeError("Must be positive")

    @property
    @_fetch_variable
    def event_window(self):
        """Two numbers for the start and stop time of the event.

        This is a 64-bit number in units of 10 ns that follows the UNIX clock.
        Or rather, it starts from January 1, 1970."""
        pass

    @event_window.setter
    @_set_variable
    def event_window(self, value):
        if not isinstance(value, (tuple)):
            raise RuntimeError("Wrong type; must be tuple.")
        elif len(value) != 2:
            raise RuntimeError("Wrong size; must be event_duration 2.")

    @property
    @_fetch_variable
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
    @_set_variable
    def pmt_waveforms(self, value):
        if not isinstance(value, (np.array)):
            raise RuntimeError("Wrong type; must be numpy array.")
        elif value.ndim != 2:
            raise RuntimeError("Wrong size; must be dimension 2.")
        elif value.shape[0] > 500:
            self.log.warning("Found %d channels, which seems high." % value.shape[0])
        elif value.shape[1] > 100000:
            self.log.warning("Found %d samples, which seems high." % value.shape[1])

    @property
    @_fetch_variable
    def S2s(self):
        """List of S2 (ionization) signals

        Returns an :class:`pax.datastructure.Peak` class.
        """
        pass

    @S2s.setter
    @_set_variable
    def S2s(self, value):
        if not isinstance(value, (list, tuple)):
            raise RuntimeError("Wrong type; must be ntuple.")
        for item in value:
            if not isinstance(value, Peak):
                raise ValueError("Must pass Peak class")

    @property
    @_fetch_variable
    def S1s(self):
        """List of S1 (scintillation) signals

        Returns an :class:`pax.datastructure.Peak` class.
        """
        pass

    @S1s.setter
    @_set_variable
    def S1s(self, value):
        if not isinstance(value, (list, tuple)):
            raise RuntimeError("Wrong type; must be ntuple.")
        for item in value:
            if not isinstance(value, Peak):
                raise ValueError("Must pass Peak class")

    @property
    @_fetch_variable
    def waveforms(self):
        """Returns a list of waveforms

        Returns an :class:`pax.datastructure.SumWaveform` class.
        """
        pass

    @waveforms.setter
    @_set_variable
    def waveforms(self, value):
        if not isinstance(value, (list, tuple)):
            raise RuntimeError("Wrong type; must be ntuple.")
        for item in value:
            if not isinstance(value, SumWaveform):
                raise ValueError("Must pass SumWaveform")


    @property
    @_fetch_variable
    def occurrences(self):
        """A dictionary of DAQ occurrences

        A DAQ occurrence is essentially a 'pulse' for one channel.
        """
        pass

    @occurrences.setter
    @_set_variable
    def occurrences(self, value):
        if not isinstance(value, dict):
            raise RuntimeError("Wrong type; must be dict.")
        for k, v in value.items():
            if not isinstance(k, int):
                raise RuntimeError("Wrong type; key be int.")
            if not isinstance(v, list):
                raise RuntimeError("Wrong type; value be list.")

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
        """Duration of event window in units of samples
        """
        return self.event_window[1] - self.event_window[0]

    def event_start(self):
        """Start time of the event in units of samples
        """
        return self.event_window[0]

    def event_end(self):
        """End time of event in units of samples
        """
        return self.event_window[1]

    def sum_waveform(self):
        """Top and bottom sum waveform method for convienence.
        """
        pass

    def filtered_waveform(self):
        """Top and bottom filtered waveform method for convienence.
        """
        pass

class Peak(object):

    """Class for S1 and S2 peak information
    """

    def __init__(self):
        self._area = 'blah'

    @property
    def area(self):
        """Area of the pulse in photoelectrons
        """
        pass

    @property
    def time_in_waveform(self):
        """position of each S1 peak"""
        pass

    @property
    def height(self):
        """Highest point in peak in units of ADC counts
        """
        pass

    @property
    def width_fwhm(self):
        """Width at full width half max

        Units are 10 ns.
        """
        pass

    @property
    def bounds(self):
        """Where the peak starts and ends in the sum waveform
"""
        return (self.peak_dict['left'],
                self.peak_dict['right'])


class SumWaveform(object):

    """Class used to store sum (filtered or not) waveform information.
    """
    @property
    def is_filtered(self):
        """Boolean"""
        pass

    @property
    def name_of_filter(self):
        """Name of the filter used (or None)
        """
        pass

    @property
    def short_name(self):
        """e.g., top"""
        pass

    @property
    def pmt_list(self):
        """Array of PMT numbers included in this waveform"""
        pass

    @property
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

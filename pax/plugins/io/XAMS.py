"""
This plug-in reads raw waveform data from the Old XAMS Lecroy scope binary & text files.

"""
import glob
import re
import struct

import numpy as np

import time
from pax import plugin, units
from pax.datastructure import Event, Occurrence


# TODO: probably doesn't work anymore

class DirWithWaveformFiles(plugin.InputPlugin):

    # Should be set by child class
    filename_regexp = None

    def startup(self):
        self.input_dirname = self.config['input_name']
        files = glob.glob(self.input_dirname + "/*.*")
        if len(files) == 0:
            self.log.fatal(
                "No files found in input directory %s!" % self.input_dirname)
        # Load all events... ugly code
        self.events = {}
        for file in files:
            m = re.search(self.filename_regexp + '$', file)
            if m is None:
                continue
            gd = m.groupdict()
            idx, channel = gd['event_index'], gd['channel']
            # Have we never seen a channel data file for this event?
            if idx not in self.events:
                self.events[idx] = {}
            # Store all regex fields, so we can find the file later
            self.events[idx][channel] = gd
        if len(self.events) == 0:
            self.log.fatal(
                "No valid event files found in input directory %s!" % self.input_dirname)
        # Maybe sort self.events by event index?
        # ????? self.get_next_event()
        self.first_event = min(self.events.keys())
        self.last_event = max(self.events.keys())
        self.log.debug('Found events %s to %s in input directory' %
                       (self.first_event, self.last_event))

    def get_events(self):
        for index in self.events:
            yield self.get_single_event(index)

    def get_single_event(self, index):

        if index not in self.events:
            raise RuntimeError("Event %s not found! File missing?" % index)

        # Get the event data
        self.current_event_channels = self.events[index]
        occurrences = {
            int(channel): [(0, np.array(self.get_channel_data(channel), dtype=np.int16))]
            for channel in self.events[index]
        }
        event_length = len(occurrences[list(occurrences.keys())[0]][0][1])

        now = time.time()
        event = Event(
            n_channels=self.config['n_channels'],
            sample_duration=self.config['sample_duration'],
            # XAMS files don't have a timestamp!
            # TODO: can we get this from e.g. file creation time?
            start_time=int(np.random.random() + now * units.s),
            length=event_length
        )

        # Convert to new occurrences format... meh
        for ch, occs in occurrences:
            for oc in occs:
                event.occurrences.append(Occurrence(
                    left=oc[0],
                    raw_data=oc[1],
                    channel=ch,
                ))

        event.event_number = int(index)

        return event

    def get_channel_filename(self, channel):
        temp = self.filename_regexp
        for key, value in self.current_event_channels[str(channel)].items():
            temp = re.sub('\(\?P\<%s\>[^\)]*\)' % key, value, temp)
        return self.input_dirname + '/' + temp

    # Should be overriden by child class
    def get_channel_data(self, channel):
        raise NotImplementedError()


class XAMSBinary(DirWithWaveformFiles):
    filename_regexp = 'C(?P<channel>\d+)_(?P<voltage>\d+)V_(?P<event_index>\d*).trc'

    def get_channel_data(self, channel):
        filename = self.get_channel_filename(channel)
        with open(filename, 'rb') as input:
            input.seek(167)
            # Set metadata (ideally we'd read this from some other file... but
            # oh well)
            self.dV = struct.unpack('<f', input.read(4))[0] * units.mV
            print(self.dV, units.mV, type(self.dV))
            if not self.dV == self.config['digitizer_voltage_range'] / 2 ** (self.config['digitizer_bits']):
                self.log.error("This file reports a digitizer dV of %s mV, your settings imply %s mv!!" % (
                    self.dV / units.mV,
                    self.config['digitizer_voltage_range'] / 2 ** (self.config['digitizer_bits']))
                )
            input.seek(187)
            self.dt = struct.unpack('<f', input.read(4))[0] * units.s
            if not self.dt == self.config['sample_duration']:
                self.log.error("This file reports a digitizer dt of %s ns,"
                               "your settings say %s ns!!" % (self.dt / units.ns,
                                                              self.config['sample_duration'] / units.ns))
            input.seek(357)
            return np.fromfile(input, dtype=np.int8)


class LecroyScopeTxt(DirWithWaveformFiles):
    filename_regexp = 'C(?P<channel>\d+)_(?P<voltage>\d+)V_(?P<event_index>\d*).dat'

    def __init__(self, dir):
        self.dir = dir
        self.dt = 1 * units.ns  # ns resolution.. todo: don't hardcode
        self.dV = 1 * units.V  # Values are analog

        super(LecroyScopeTxt, self).__init__()

    def get_channel_data(self, channel):
        filename = self.get_channel_filename(channel)
        with open(filename, 'rb') as input:
            return np.array([float(line.split()[1]) for line in input])

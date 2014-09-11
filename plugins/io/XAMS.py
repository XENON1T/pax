"""
This plug-in reads raw waveform data from the XAMS scope binary & text files.


"""
import glob, re, struct, time
import numpy as np

from pax import plugin, units
from pax.datastructure import Event


class DirWithWaveformFiles(plugin.InputPlugin):

    def startup(self):
        #self.config['input_dir'],
        # Assumes self.filename_regexp is set by child class
        files = glob.glob(self.config['input_dir'] + "/*.*")
        if len(files)==0:
            self.log.fatal("No files found in input directory %s!" % self.config['input_dir'])
        #Load all events... ugly code
        self.events = {}
        for file in files:
            m = re.search(self.filename_regexp + '$',file)
            if m is None: continue
            gd = m.groupdict()
            idx, channel = gd['event_index'], gd['channel']
            # Have we never seen a channel data file for this event?
            if not idx in self.events:
                self.events[idx] = {}
            # Store all regex fields, so we can find the file later
            self.events[idx][channel] = gd
        if len(self.events) == 0:
            self.log.fatal("No valid event files found in input directory %s!" % self.config['input_dir'])
        #Maybe sort self.events by event index?
        #????? self.get_next_event()
        self.first_event = min(self.events.keys())
        self.last_event = max(self.events.keys())

    def get_single_event(self, index):

        if not index in self.events:
            raise RuntimeError("Event %s not found! File missing?" % index)
        event = Event()

        #Get the event data
        self.current_event_channels = self.events[index]
        event.occurrences = {
            channel : (0, np.array(self.get_channel_data(channel), dtype=np.int16))
            for channel in self.events[index]
        }
        event_length = len(event.occurrences[event.occurrences.keys()[0]][1])

        # Set all the other stuff
        event.event_number = int(index)
        event.sample_duration = int(1 * units.ns)    # TODO: don't hardcode sample size...
        now = time.now()    # TODO: read file creation time?
        event.start_time = int(np.random.random() + now * units.s)
        # Remember stop_time is the stop time of the LAST sample!
        event.stop_time = event.start_time + int(event_length * event.sample_duration)

        return event

    def get_channel_filename(self, channel):
        temp = self.filename_regexp
        for key, value in self.current_event_channels[str(channel)].items():
            temp = re.sub('\(\?P\<%s\>[^\)]*\)'%key, value, temp)
        return self.config['input_dir'] +'/'+ temp



class XAMSBinary(DirWithWaveformFiles):
    filename_regexp = 'C(?P<channel>\d+)_(?P<voltage>\d+)V_(?P<event_index>\d*).trc'

    def get_channel_data(self, channel):
        filename = self.get_channel_filename(channel)
        #Open the file
        input = open(filename, 'rb')
        input.seek(167)
        #Set metadata (ideally we'd read this from some other file... but oh well)
        self.dV = struct.unpack('<f', input.read(4))[0] *units.V
        if not self.dV == self.config['digitizer_voltage_range']/2**(self.config['digitizer_bits']):
            raise RuntimeError("This file reports a digitizer dV of %s mV, your settings imply something else!!" % self.dV/units.mV)
        input.seek(187)
        self.dt = struct.unpack('<f', input.read(4))[0] *units.s
        if not self.dt == self.config['digitizer_time_resolution']:
            raise RuntimeError("This file reports a digitizer dt of %s ns, your settings say %s ns!!" % (self.dt/units.ns, self.config['digitizer_time_resolution']/units.ns))
        input.seek(357)
        data = np.fromfile(input, dtype=np.int8)
        input.close()   #I know about with... but do you know about python 2.3?
        return data



class LecroyScopeTxt(DirWithWaveformFiles):
    filename_regexp = 'C(?P<channel>\d+)_(?P<voltage>\d+)V_(?P<event_index>\d*).dat'

    def __init__(self, dir):
        self.dir = dir
        self.dt = 1*units.ns  #ns resolution.. todo: don't hardcode
        self.dV = 1*units.V   #Values are analog

        super(LecroyScopeTxt, self).__init__()

    def get_channel_data(self, channel):
        filename = self.get_channel_filename(channel)
        input = open(filename, 'r')
        data = np.array([float(line.split()[1]) for line in input])
        input.close()   #I know about with... but do you know about python 2.3?
        return data

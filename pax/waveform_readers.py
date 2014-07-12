"""
Waveform formats:
    - DirWithWaveforms: every channel in each event is one file 
        - LecroyScopeText   (for XAms)
        - LecroyScopeBinary (for XAms)
        - Fax stored waveforms?
    - XDIO
    - FaX (simulates a new s2 waveform every time)
    - Mongo
TODO:
    - More general FaX waveforms
    - ???
    
Behaviour:
    inputformat = YourFancyWaveformFormat(...)
    for event_id inputformat.get_next_event():
        for channel_id in inputformat.get_next_channel():
            waveform = inputformat.get_channel_data(channel)
            #Do your magic
    inputformat.close()
    
To add: functions for converting between formats, i.e. only need wave-writing functions, some driving code...
"""


import glob, collections, re, struct, math, io, bz2, random
from . import units
import numpy as np
try:
    from pymongo import MongoClient
    from bson.binary import Binary
except:
    print("You don't seem to have pymongo: the mongodb input format won't work")

class WaveformFormat(object):
    def get_next_event(self):   #Override this if you build event index gradually
        for (event_index, self.current_event_channels) in list(self.events.items()):
            yield event_index
            
    def get_next_channel(self):
        #Assuming channel is an int!
        for self.current_channel in self.current_event_channels:
            yield int(self.current_channel)
            
    def get_channel_data(self, channel):
        return self.current_event_channels[channel]    
        
    def close(self):
        pass
            
class DirWithWaveforms(WaveformFormat):
    def __init__(self):
        #self.dir and self.filename_regexp have to be already set by child class
        files = glob.glob(self.dir+"/*.*")
        if len(files)==0:
            raise Exception("DirWithWaveforms format: no files found in input directory %s!" % self.dir)
        #Load all events... ugly code
        self.events = collections.OrderedDict()
        for file in files:
            m = re.search(self.filename_regexp+'$',file)
            if m == None: continue
            gd = m.groupdict()
            if gd['event_index'] in self.events:
                self.events[gd['event_index']][gd['channel']] = gd
            else:
                self.events[gd['event_index']] = collections.OrderedDict()
                self.events[gd['event_index']][gd['channel']] = gd
        if len(self.events) == 0:
            raise Exception("No valid events found in input directory %s!" % self.dir)
        #Maybe sort self.events by event index?
        self.get_next_event()
        
    def get_channel_filename(self, channel):
        temp = self.filename_regexp
        for key, value in list(self.current_event_channels[str(channel)].items()):
            temp = re.sub('\(\?P\<%s\>[^\)]*\)'%key, value, temp)
        return self.dir +'/'+ temp
        
class LecroyScopeBinary(DirWithWaveforms):
    def __init__(self, dir):
        self.dir = dir
        self.filename_regexp = 'C(?P<channel>\d+)_(?P<voltage>\d+)V_(?P<event_index>\d*).trc'
        super(LecroyScopeBinary, self).__init__()
        
    def get_channel_data(self, channel):
        filename = self.get_channel_filename(channel)
        #Open the file
        input = open(filename, 'rb')
        input.seek(167)
        #Set metadata (ideally we'd read this from some other file... but oh well)
        self.dV = struct.unpack('<f', input.read(4))[0] *units.V
        input.seek(187)
        self.dt = struct.unpack('<f', input.read(4))[0] *units.s
        input.seek(357)
        data = np.fromfile(input, dtype=np.int8)
        input.close()   #I know about with... but do you know about python 2.3?
        return data
   
class LecroyScopeTxt(DirWithWaveforms):
    def __init__(self, dir):
        self.dir = dir
        self.dt = 1*units.ns  #ns resolution.. todo: don't hardcode
        self.dV = 1*units.V   #Values are analog
        self.filename_regexp = 'C(?P<channel>\d+)_(?P<voltage>\d+)V_(?P<event_index>\d*).dat'
        super(LecroyScopeTxt, self).__init__()
        
    def get_channel_data(self, channel):
        filename = self.get_channel_filename(channel)
        input = open(filename, 'r')
        data = np.array([float(line.split()[1]) for line in input])
        input.close()   #I know about with... but do you know about python 2.3?
        return data
        
class MongoDB(WaveformFormat):
    def __init__(self, settings, collection, server='145.102.135.218', port=27017, database='output'):
        self.settings = settings
        self.client = MongoClient(server, port)
        self.coll = self.client[database][collection]
        print("MongoDB Format: Reading from collection %s" % collection)
        self.cursor = self.coll.find()
        self.number_of_events = self.cursor.count()
        if self.number_of_events == 0: raise NameError("No events found... did you run the event builder?")
        #These are not in the Mongo format (yet?)... hope your settings are right!
        self.dt = settings.digitizer_t_resolution
        self.dV = settings.digitizer_V_resolution
        
    def get_next_event(self):
        for i in range(self.number_of_events):
            self.current_event_channels = {}
            event = next(self.cursor)
            print(event.keys())
            (event_start, event_end) = event['range']
            event_length = event_end - event_start
            for doc in event['docs']:
                channel = doc['channel']
                wave_start = doc['time'] - event_start
                if channel not in self.current_event_channels:
                    self.current_event_channels[channel] = self.settings.digitizer_baseline * np.ones(event_length, dtype=np.int16)
                waveform = np.fromstring(doc['data'],dtype=np.int16)
                self.current_event_channels[channel][wave_start:wave_start+len(waveform)] = waveform
            yield i
        
class FaX_S2s(WaveformFormat):

    def __init__(self, settings, number_of_events, depth_range=None, electrons_range=(0,1000)):
        self.number_of_events = number_of_events
        import fax
        self.fax = fax
        fax.load_settings(settings)
        if depth_range==None:
            self.depth_range = (0,30*units.cm)
        else:
            self.depth_range = depth_range
        self.electrons_range = electrons_range
        self.dt = fax.st.digitizer_t_resolution
        self.dV = fax.st.digitizer_V_resolution

    def get_next_event(self):
        for event_index in range(self.number_of_events):
            self.event_metadata = {
                'interaction_depth':    random.randint(*self.depth_range),
                'electrons_generated':  random.randint(*self.electrons_range)
            }
            electron_times = self.fax.s2_electrons(**self.event_metadata)
            if len(electron_times)==0: continue
            hitlists = self.fax.s2_scintillation(electron_times)
            fax_event = self.fax.make_event(hitlists)
            self.current_event_channels = fax_event['waveforms']
            self.event_metadata.update({
                'index'             :   event_index,
                'electrons_seen'    :   len(electron_times),
                'photons_generated' :   len(list(hitlists.values())),
                #Todo: load all metadata in fax_event automatically
                'start_time'        :   fax_event['start_time'],
                'end_time'          :   fax_event['end_time'],
            })
            yield event_index
    
        
class Xed(WaveformFormat):
    file_header = np.dtype([
        ("dataset_name",            "S64"),
        ("creation_time",           "<u4"),
        ("first_event_number",      "<u4"),
        ("events_in_file",          "<u4"),
        ("event_index_size",        "<u4")
    ])

    event_header = np.dtype([
        ("dataset_name",            "S64"),
        ("utc_time",                "<u4"),
        ("utc_time_usec",           "<u4"),
        ("event_number",            "<u4"),
        ("chunks",                  "<u4"),
        #This is where the 'chunk layer' starts... but there always seems to be one chunk per event
        #I'll always assume this is true and raise an exception otherwise
        ("type",                    "S4"),
        ("size",                    "<u4"),
        ("sample_precision",        "<i2"),
        ("flags",                   "<u2"), #indicating compression type.. I'll assume bzip2 always
        ("samples_in_event",        "<u4"),
        ("voltage_range",           "<f4"),
        ("sampling_frequency",      "<f4"),
        ("channels",                "<u4"),
    ])

    def __init__(self, filename):
        self.input = open(filename,'rb')
        file_metadata      =  np.fromfile(self.input, dtype=Xed.file_header, count=1)[0]
        #print "File data: " + str(file_metadata)
        assert file_metadata['events_in_file'] == file_metadata['event_index_size']
        self.event_positions = np.fromfile(self.input, dtype=np.dtype("<u4"), count=file_metadata['event_index_size'])
        
    #This spends a lot of time growing the numpy array. Maybe faster if we first allocate 4000000 zeroes.
    def get_next_event(self):
        input = self.input
        for event_position in self.event_positions:
            self.current_event_channels = {}
            if not input.tell() == event_position:
                raise Exception("Reading error: this event should be at %s, but we are at %s!" % (event_position, input.tell()))
            self.event_metadata = np.fromfile(input, dtype=Xed.event_header, count=1)[0]
            #print "Reading event %s, consisting of %s chunks" % (self.event_metadata['event_number'], self.event_metadata['chunks'])
            if self.event_metadata['chunks'] != 1:
                raise Exception("The day has come: event with %s chunks found!" % self.event_metadata['chunks'])
            #print "Event type %s, size %s, samples %s, channels %s" % (self.event_metadata['type'], self.event_metadata['size'], self.event_metadata['samples_in_chunk'], event_metadata['channels'],)

            mask_bytes = 4 * int(math.ceil(self.event_metadata['channels']/32))
            mask = self.bytestobits(
                np.fromfile(input, dtype=np.dtype('<S%s'% mask_bytes), count=1)[0]
            )   #Last bytes are on front or something? Maybe whole mask is a single little-endian field??
            channels_included = [i for i,m in enumerate(reversed(mask)) if m == 1]
            chunk_fake_file = io.StringIO(input.read(self.event_metadata['size']-28-mask_bytes))  #28 is the chunk header size. TODO: only decompress if needed
            for channel_id in channels_included:
                channel_waveform = np.array([],dtype="<i2")
                #Read channel size (in 4bit words), subtract header size, convert from 4-byte words to bytes
                channel_data_size =  4*(np.fromstring(chunk_fake_file.read(4), dtype='<u4')[0] - 1)
                #print "Data size for channel %s is %s bytes" % (channel_id, channel_data_size)
                channel_fake_file = io.StringIO(chunk_fake_file.read(channel_data_size))
                while 1:
                    control_word_string = channel_fake_file.read(4)
                    if not control_word_string: break
                    control_word = np.fromstring(control_word_string, dtype='<u4')[0]
                    if control_word<2**31:
                        #print "Next %s samples are 0" % (2*control_word)
                        channel_waveform = np.append(channel_waveform, np.zeros(2*control_word))
                        continue
                    else:
                        data_samples = 2*(control_word-(2**31))
                        #print "Now reading %s data samples" % data_samples
                        baseline = 16000 #TODO: determine this from wave, as XerawDP does?
                        samples_fromfile = np.fromstring(channel_fake_file.read(2*data_samples), dtype="<i2")
                        """
                        According to Guillaume's thesis, and the FADC manual, samples come in pairs, with later sample first!
                        This would mean we have to ungarble them (split into pairs, reverse the pairs, join again):
                            channel_waveform = np.append(channel_waveform, ungarble_samplepairs(samples_fromfile-baseline))
                        However, this makes the peak come out two-headed... Maybe they were already un-garbled by some previous program??
                        We won't do any ungarbling for now:
                        """
                        channel_waveform = np.append(channel_waveform, samples_fromfile-baseline)
                if len(channel_waveform) != 40000:
                    raise Exception("Channel %s waveform is..")
                self.current_event_channels[channel_id] = channel_waveform

            #Ok... reading is done, time to yield 
            self.dV = units.V * self.event_metadata['voltage_range']/2**14
            self.dt = units.s * 1/self.event_metadata['sampling_frequency']
            yield self.event_metadata['event_number']
        #If we get here, all events have been read
        self.input.close()
            
    def bytestobits(self,bytes_string):
        bits = []
        bytes = (ord(bs) for bs in bytes_string)
        for b in bytes:
            for i in range(8):
                bits.append( (b >> i) & 1 )
        return bits
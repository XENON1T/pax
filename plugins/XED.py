import numpy as np

from pax import plugin, units

class Xed(plugin.InputPlugin):
    is_baseline_corrected = True

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

    def __init__(self, config):
        self.input = open(filename,'rb')
        file_metadata      =  np.fromfile(self.input, dtype=Xed.file_header, count=1)[0]
        #print "File data: " + str(file_metadata)
        assert file_metadata['events_in_file'] == file_metadata['event_index_size']
        self.event_positions = np.fromfile(self.input, dtype=np.dtype("<u4"), count=file_metadata['event_index_size'])
        
        # Few steps of math for conversion factor
        self.conversion_factor = config[
            'digitizer_V_resolution'] * config['digitizer_t_resolution']
        self.conversion_factor /= config['gain']
        self.conversion_factor /= config['digitizer_resistor']
        self.conversion_factor /= config['digitizer_amplification']
        self.conversion_factor /= units.electron_charge
        
    #This spends a lot of time growing the numpy array. Maybe faster if we first allocate 40000 zeroes.
    def get_next_event(self):
        input = self.input
        for event_position in self.event_positions:
            self.current_event_channels = {}
            if not input.tell() == event_position:
                raise Exception, "Reading error: this event should be at %s, but we are at %s!" % (event_position, input.tell())
            self.event_metadata = np.fromfile(input, dtype=Xed.event_header, count=1)[0]
            #print "Reading event %s, consisting of %s chunks" % (self.event_metadata['event_number'], self.event_metadata['chunks'])
            if self.event_metadata['chunks'] != 1:
                raise Exception, "The day has come: event with %s chunks found!" % event_metadata['chunks']
            #print "Event type %s, size %s, samples %s, channels %s" % (self.event_metadata['type'], self.event_metadata['size'], self.event_metadata['samples_in_chunk'], event_metadata['channels'],)
            if self.event_metadata['type'] == 'zle0':
                """
                Read the arcane mask
                Hope this is ok with endianness and so on... no idea... what happens if it is wrong??
                TODO: Check with the actual bits in a hex editor..
                """
                mask_bytes = 4 * int(math.ceil(self.event_metadata['channels']/32))
                mask = self.bytestobits(
                    np.fromfile(input, dtype=np.dtype('<S%s'% mask_bytes), count=1)[0]
                )   #Last bytes are on front or something? Maybe whole mask is a single little-endian field??
                channels_included = [i for i,m in enumerate(reversed(mask)) if m == 1]
                chunk_fake_file = StringIO.StringIO(bz2.decompress(input.read(self.event_metadata['size']-28-mask_bytes)))  #28 is the chunk header size. TODO: only decompress if needed
                for channel_id in channels_included:
                    channel_waveform = np.array([],dtype="<i2")
                    #Read channel size (in 4bit words), subtract header size, convert from 4-byte words to bytes
                    channel_data_size =  4*(np.fromstring(chunk_fake_file.read(4), dtype='<u4')[0] - 1)
                    #print "Data size for channel %s is %s bytes" % (channel_id, channel_data_size)
                    channel_fake_file = StringIO.StringIO(chunk_fake_file.read(channel_data_size))
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
                            samples_fromfile = np.fromstring(channel_fake_file.read(2*data_samples), dtype="<i2")
                            baseline, _ = dsputils.baseline_mean_stdev(samples_fromfile)
                            """
                            According to Guillaume's thesis, and the FADC manual, samples come in pairs, with later sample first!
                            This would mean we have to ungarble them (split into pairs, reverse the pairs, join again): 
                                channel_waveform = np.append(channel_waveform, ungarble_samplepairs(samples_fromfile-baseline))
                            However, this makes the peak come out two-headed... Maybe they were already un-garbled by some previous program??
                            We won't do any ungarbling for now:
                            """
                            channel_waveform = np.append(channel_waveform, samples_fromfile-baseline)
                    if len(channel_waveform) != 40000:  #TODO: don't hardcode...
                        raise Exception, "Channel %s waveform is %s samples long, expected %s!" % (channel_id, event_metadata['samples_in_event'])
                    self.current_event_channels[channel_id] = channel_waveform
            else:
                raise Exception, "Still have to code grokking for sample type %s..." % event_metadata['type']
            #Ok... reading is done, time to yield 
            self.dV = units.V * self.event_metadata['voltage_range']/2**14
            self.dt = units.s * 1/self.event_metadata['sampling_frequency']
            
        #If we get here, all events have been read
        self.input.close()
            
    def bytestobits(self,bytes_string):
        bits = []
        bytes = (ord(bs) for bs in bytes_string)
        for b in bytes:
            for i in xrange(8):
                bits.append( (b >> i) & 1 )
        return bits




class MongoDBInput(plugin.InputPlugin):

    def __init__(self, config):
        plugin.InputPlugin.__init__(self, config)

        self.client = MongoClient(config['mongodb_address'])
        self.database = self.client[config['database']]
        self.collection = self.database[config['collection']]

        self.baseline = config['digitizer_baseline']

        # TODO (tunnell): Sort by event number
        self.cursor = self.collection.find()
        self.number_of_events = self.cursor.count()

        if self.number_of_events == 0:
            raise RuntimeError(
                "No events found... did you run the event builder?")

        # Few steps of math for conversion factor
        self.conversion_factor = config[
            'digitizer_V_resolution'] * config['digitizer_t_resolution']
        self.conversion_factor /= config['gain']
        self.conversion_factor /= config['digitizer_resistor']
        self.conversion_factor /= config['digitizer_amplification']
        self.conversion_factor /= units.electron_charge

    @staticmethod
    def baseline_mean_stdev(samples):
        """ returns (baseline, baseline_stdev) """
        baseline_sample = samples[:42]
        return (
            np.mean(sorted(baseline_sample)[int(0.4 * len(baseline_sample)):int(0.6 * len(baseline_sample))
                                            ]),  # Ensures peaks in baseline sample don't skew computed baseline
            np.std(baseline_sample)  # ... but do count towards baseline_stdev!
        )
        # Don't want to just take the median as V-resolution is finite

    def GetEvents(self):
        """Generator of events from Mongo

        What is returned is all of the channel waveforms
        """
        for doc_event in self.collection.find():
            current_event_channels = {}

            # Build channel waveforms by iterating over all occurences.  This
            # involves parsing MongoDB documents using WAX output format
            (event_start, event_end) = doc_event['range']
            event_length = event_end - event_start
            for doc_occurence in doc_event['docs']:
                channel = doc_occurence['channel']
                wave_start = doc_occurence['time'] - event_start
                if channel not in current_event_channels:
                    current_event_channels[
                        channel] = self.baseline * np.ones(event_length, dtype=np.int16)
                waveform = np.fromstring(doc_occurence['data'], dtype=np.int16)
                current_event_channels[channel][
                    wave_start:wave_start + len(waveform)] = waveform

            # 'event' is what we will return
            event = {}
            event['channel_waveforms'] = {}

            # Remove baselines
            for channel, data in current_event_channels.items():

                baseline, _ = self.baseline_mean_stdev(data)

                event['channel_waveforms'][channel] = -1 * \
                    (data - baseline) * self.conversion_factor

            yield event

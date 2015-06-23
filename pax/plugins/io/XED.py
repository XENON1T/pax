"""
These plugins read & write raw waveform data from a Xenon100 XED file.
The XED file format is documented in Guillaume Plante's PhD thesis.
This is code does not use the libxdio C-library though.

The read plugin supports:
 - sequential reading and random access (XED has an index, so this is easy)
 - one 'XED chunk' per event, it raises an exception if it sees more than one chunk.
   This seems to be fine for all the XENON100 data I've looked at.
 - zle0 and raw sample encoding;
 - bzip2 or uncompressed chunk data compression, not any other compression scheme.
 
The write plugin always writes zle0 XED files with bzip2 data compression.
"""


import bz2
import io
import time
from itertools import groupby
import math

import numpy as np

from pax import units
from pax.datastructure import Event, Pulse

from pax.FolderIO import InputFromFolder, WriteToFolder


xed_file_header = np.dtype([
    ("dataset_name", "S64"),
    ("creation_time", "<u4"),
    ("first_event_number", "<u4"),
    ("events_in_file", "<u4"),
    ("event_index_size", "<u4")
])

xed_event_header = np.dtype([
    ("dataset_name", "S64"),
    ("utc_time", "<u4"),
    ("utc_time_usec", "<u4"),
    ("event_number", "<u4"),
    ("chunks", "<u4"),
    # This is where the 'chunk layer' starts... but there always seems to be one chunk per event
    # I'll always assume this is true and raise an exception otherwise
    ("type", "S4"),
    ("size", "<u4"),
    ("sample_precision", "<i2"),
    ("flags", "<u2"),               # indicating compression type.. I'll assume bzip2 always
    ("samples_in_event", "<u4"),
    ("voltage_range", "<f4"),
    ("sampling_frequency", "<f4"),
    ("channels", "<u4"),
])


class ReadXED(InputFromFolder):

    file_extension = 'xed'

    def get_first_and_last_event_number(self, filename):
        """Return the first and last event number in file specified by filename"""
        with open(filename, 'rb') as xedfile:
            fmd = np.fromfile(xedfile, dtype=xed_file_header, count=1)[0]
            return (fmd['first_event_number'],
                    fmd['first_event_number'] + fmd['events_in_file'] - 1)

    def close(self):
        """Close the currently open file"""
        self.current_xedfile.close()

    def open(self, filename):
        """Opens an XED file so we can start reading events"""
        self.current_xedfile = open(filename, 'rb')

        # Read in the file metadata
        self.file_metadata = np.fromfile(self.current_xedfile,
                                         dtype=xed_file_header,
                                         count=1)[0]
        self.event_positions = np.fromfile(self.current_xedfile,
                                           dtype=np.dtype("<u4"),
                                           count=self.file_metadata['event_index_size'])

        # Handle for special case of last XED file
        # Index size is larger than the actual number of events written:
        # The writer didn't know how many events there were left when it started reserving space for index
        if self.file_metadata['events_in_file'] < self.file_metadata['event_index_size']:
            self.log.info(
                ("The XED file claims there are %d events in the file, "
                 "while the event position index has %d entries. \n"
                 "Is this the last XED file of a dataset?") %
                (self.file_metadata['events_in_file'], self.file_metadata['event_index_size'])
            )
            self.event_positions = self.event_positions[:self.file_metadata['events_in_file']]

    def get_single_event_in_current_file(self, event_position):

        # Seek to the requested event
        self.current_xedfile.seek(self.event_positions[event_position])

        # Read event metadata, check if we can read this event type.
        event_layer_metadata = np.fromfile(self.current_xedfile,
                                           dtype=xed_event_header,
                                           count=1)[0]
        if event_layer_metadata['chunks'] != 1:
            raise NotImplementedError("Can't read this XED file: event with %s chunks found!"
                                      % event_layer_metadata['chunks'])

        # Check if voltage range and digitizer dt are the same as in the settings
        # If not, raise error. Would be simple matter to change settings dynamically, but that's weird.
        values_to_check = (
            ('Voltage range',   self.config['digitizer_voltage_range'],
             event_layer_metadata['voltage_range']),
            ('Digitizer dt',    self.config['sample_duration'],
             1 / (event_layer_metadata['sampling_frequency'] * units.Hz)),
        )
        for name, ini_value, xed_value in values_to_check:
            if ini_value != xed_value:
                raise RuntimeError(
                    '%s from XED event metadata (%s) is different from ini file setting (%s)!'
                    % (name, xed_value, ini_value)
                )

        # Start building the event
        event = Event(
            n_channels=self.config['n_channels'],
            start_time=int(
                event_layer_metadata['utc_time'] * units.s +
                event_layer_metadata['utc_time_usec'] * units.us
            ),
            sample_duration=int(self.config['sample_duration']),
            length=event_layer_metadata['samples_in_event']
        )
        event.dataset_name = self.file_metadata['dataset_name'].decode("utf-8")
        event.event_number = int(event_layer_metadata['event_number'])

        if event_layer_metadata['type'] == b'raw0':
            # Grok 'raw' XEDs - these probably come from the LED calibration

            # 4 unused bytes at start (part of 'chunk header')
            self.current_xedfile.read(4)

            # Data is just a big bunch of samples from one channel, then next channel, etc
            # Each channel has an equal number of samples.
            data = np.fromfile(self.current_xedfile,
                               dtype='<i2',
                               count=event_layer_metadata['channels'] *
                                     event_layer_metadata['samples_in_event'])
            data = np.reshape(data, (event_layer_metadata['channels'],
                                     event_layer_metadata['samples_in_event']))
            for ch_i, chdata in enumerate(data):
                event.pulses.append(Pulse(
                    channel=ch_i + 1,       # +1 as first channel is 1 in Xenon100
                    left=0,
                    raw_data=chdata
                ))
        elif event_layer_metadata['type'] == b'zle0':
            # Read the channel bitmask to find out which channels are included in this event.
            # Lots of possibilities for errors here: 4-byte groupings, 1-byte groupings, little-endian...
            # Checked (for 14 events); agrees with channels from
            # LibXDIO->Moxie->MongoDB->MongoDBInput plugin
            mask_bytes = 4 * math.ceil(event_layer_metadata['channels'] / 32)
            mask_bits = np.unpackbits(np.fromfile(self.current_xedfile,
                                                  dtype='uint8',
                                                  count=mask_bytes))

            # +1 as first pmt is 1 in Xenon100
            channels_included = [i + 1 for i, bit in enumerate(reversed(mask_bits))
                                 if bit == 1]

            # Decompress the event data (actually, the data from a single 'chunk')
            # into fake binary file (io.BytesIO)
            # 28 is the chunk header size.

            data_to_decompress = self.current_xedfile.read(event_layer_metadata['size'] - 28 - mask_bytes)
            try:
                chunk_fake_file = io.BytesIO(bz2.decompress(data_to_decompress))
            except OSError:
                # Maybe it wasn't compressed after all? We can at least try
                # TODO: figure this out from flags
                chunk_fake_file = io.BytesIO(data_to_decompress)

            # Loop over all channels in the event to get the pulses
            for channel_id in channels_included:
                # Read channel size (in 4bit words), subtract header size, convert
                # from 4-byte words to bytes
                channel_data_size = int(4 * (np.fromstring(chunk_fake_file.read(4),
                                                           dtype='<u4')[0] - 1))

                # Read the channel data into another fake binary file
                channel_fake_file = io.BytesIO(chunk_fake_file.read(channel_data_size))

                # Read the channel data control word by control word.
                # sample_position keeps track of where in the waveform a new
                # pulse should be placed.
                sample_position = 0
                while 1:
                    # Is there a new control word?
                    control_word_string = channel_fake_file.read(4)
                    if not control_word_string:
                        break

                    # Control words starting with zero indicate a number of sample PAIRS to skip
                    control_word = int(np.fromstring(control_word_string,
                                                     dtype='<u4')[0])
                    if control_word < 2 ** 31:
                        sample_position += 2 * control_word
                        continue

                    # Control words starting with one indicate a number of sample PAIRS follow
                    else:
                        # Subtract the control word flag
                        data_samples = 2 * (control_word - (2 ** 31))

                        # Note endianness
                        samples_pulse = np.fromstring(channel_fake_file.read(2 * data_samples),
                                                      dtype="<i2")

                        event.pulses.append(Pulse(
                            channel=channel_id,
                            left=sample_position,
                            raw_data=samples_pulse
                        ))

                        sample_position += len(samples_pulse)

        else:
            raise NotImplementedError("XED type %s not supported" % event_layer_metadata['type'])

        # Check we have read all data for this event

        if event_position != len(self.event_positions) - 1:
            current_pos = self.current_xedfile.tell()
            should_be_at_pos = self.event_positions[event_position + 1]
            if current_pos != should_be_at_pos:
                raise RuntimeError("Error during XED reading: after reading event %d from file "
                                   "(event number %d) we should be at position %d, but we are at position %d!" % (
                                       event_position, event.event_number, should_be_at_pos, current_pos))

        return event


class WriteXED(WriteToFolder):
    """
    The XED is written 'inside out' in memory, then written to disk
    This way we don't have to update size fields in headers.
    """

    file_extension = 'xed'

    def open(self, filename):
        self.current_xed = open(filename, 'wb')
        self.events = []
        self.first_event_number = None

    def write_event_to_current_file(self, event):

        if self.first_event_number is None:
            self.first_event_number = event.event_number
        event_data = b''

        # First write the pulse data
        for channel, pulses_in_channel in groupby(event.pulses, key=lambda p: p.channel):
            pulses_in_channel = list(pulses_in_channel)
            channel_data = b''
            prevpulse = None
            for pulse in sorted(pulses_in_channel, key=lambda p: p.left):
                # Write skip control word, unless we're at the very start
                if pulse.left != 0:
                    if prevpulse is not None:
                        skip = (pulse.left - prevpulse.right - 1) / 2
                    else:
                        skip = pulse.left / 2
                    channel_data += np.array([skip], dtype='<u4').tobytes()
                # Write data control word
                channel_data += np.array([2 ** 31 + pulse.length / 2], dtype='<u4').tobytes()
                # Write data
                channel_data += pulse.raw_data.tobytes()
                prevpulse = pulse
            # No final skip control word?
            # Add the channel size to the start
            channel_data = np.array([1 + len(channel_data) / 4], dtype='<u4').tobytes() + channel_data
            # Add this to the event_data
            event_data += channel_data

        # Compress the event data so far
        event_data = bz2.compress(event_data, compresslevel=self.config['compresslevel'])

        # Required size of the channel mask in bytes -- for some reason it needs to be rounded to 4-byte blocks
        n_mask_bytes = 4 * math.ceil(self.config['n_channels'] / 32)

        channels_included = set([p.channel for p in event.pulses])
        is_channel_included = np.zeros(n_mask_bytes * 8, dtype=np.int)
        if self.config['pmt_0_is_fake']:
            for ch in range(1, self.config['n_channels'] - 1):
                is_channel_included[ch - 1] = ch in channels_included
        else:
            for ch in range(self.config['n_channels']):
                is_channel_included[ch] = ch in channels_included

        # Reverse channel included bits... for some reason?
        is_channel_included = is_channel_included[::-1]

        channel_mask = np.packbits(is_channel_included)

        # Add event header and channel mask to event data
        event_data_dict = dict(dataset_name=b'generated_by_pax',
                               utc_time=int(event.start_time / units.s),
                               utc_time_usec=int(event.start_time % units.s / units.us),
                               event_number=event.event_number,
                               chunks=1,
                               type=b'zle0',     # ??
                               size=28 + n_mask_bytes + len(event_data),    # 28 is the size of the chunk header
                               sample_precision=self.config['digitizer_bits'],
                               flags=2,    # Stolen from X100 files, probably
                               samples_in_event=event.length(),
                               voltage_range=self.config['digitizer_voltage_range'],
                               sampling_frequency=1 / self.config['sample_duration'] / units.Hz,
                               channels=self.config['n_channels'])
        event_header_arr = _dict_to_numpy_recarray(event_data_dict, dtype=xed_event_header)
        event_data = event_header_arr.tobytes() + channel_mask.tobytes() + event_data

        self.events.append(event_data)

    def close(self):

        # Write file header
        file_header_dict = dict(dataset_name=b'generated_by_pax',
                                creation_time=time.time(),
                                first_event_number=self.first_event_number,
                                events_in_file=len(self.events),
                                event_index_size=len(self.events))
        _dict_to_numpy_recarray(file_header_dict, dtype=xed_file_header).tofile(self.current_xed)

        # Write event index. 80 is the event header size.
        event_index = np.cumsum(np.concatenate((np.array([0], dtype='<u4'),
                                                np.array([len(e) for e in self.events], dtype='<u4')[:-1])))
        event_index += 80 + 4 * len(event_index)
        event_index.tofile(self.current_xed)

        # Write event data
        for e in self.events:
            self.current_xed.write(e)

        self.current_xed.close()

        self.events = []
        self.first_event_number = None


def _dict_to_numpy_recarray(d, dtype):
    """Why is this so difficult? Stackoverflow says np.array(dict.items(), dtype=dt) should work, but it doesn't...
    """
    fieldnames = list(map(lambda x: x[0], dtype.descr))
    data_tuple = tuple(d[f] for f in fieldnames)
    return np.array([data_tuple], dtype=dtype)

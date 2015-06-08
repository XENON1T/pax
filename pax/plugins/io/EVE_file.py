"""This plugin will read binary data from the Münster TPC. The data is stored in .eve files that are created
with the daq software "FPPGUI" written by Volker Hannen.

In short the file is built up like this:
1 file header
2 event header
3 configuration header
4 event header
5 signal header
6 data
7 32bit word "0xaffe"
<repeat 4 to 7 until end of file>

For a more detailed understanding of the file read the short manual.
"""

import bz2
import io

import numpy as np

import math
from pax import units
from pax.datastructure import Event, Pulse

from pax.plugins.io.FolderIO import InputFromFolder

"""  File header provided by FPPGui. This has only to be read once. Byte order is a relict from times where "Big Endian"
processors where not unusual. See Wikipedia "byte order" about that topic in necessary.
"""
eve_file_header = np.dtype([
    ("byte_order", "<u4"),
    ("version", "<u4"),
    ("buffsize", "<u4"),
    ("timestamp", "<u4"),
    ("not_used", "<4u4"),
])

"""This is the datastructure of a header event by the caen boards. It is copied from wfread.cpp from sisdac by Volker Hannen
and adapted to numpys dtype. The value "20" stands for a hardcoded maximum number of caen boards within the DAQ software.

This has to be considered once the DAQ software changes.
The values stored in this header are the same as in the file caen1724.par it was created with.
"""
eve_caen1724_par_t = np.dtype([
    ("no_mod", "<i"),
    ("base", "<20u4"),
    ("downsample_factor", "<i4"),
    ("page_size", "<i4"),
    ("post_trigger_samples", "<i4"),
    ("enable_external_trig", "<i4"),
    ("trigger_logic", "<i4"),
    ("trigger_coinc_level", "<i4"),
    ("trigger_n_quartets", "<i4"),

    ("zle", "<i4"),  # zero length encoding
    ("zle_logic", "<i4"),  # zle, positive(0) or negative(1) logic
    ("zle_nlbk", "<i4"),  # zle, number of look back words
    ("zle_nlfwd", "<i4"),  # zle, number of look forward words

    ("integrate_signals", "<i4"),
    ("substract_offset", "<i4"),

    ("chan_active", "<(20,8)i4"),
    ("chan_dac", "<(20,8)u4"),
    ("threshold", "<(20,8)i4"),
    ("zs_threshold", "<(20,8)i4"),
    ("zero_offset", "<(20,8)i4"),

    ("nof_active_channels", "<20i4"),  # 0...8, number of active channels on each caen board
    ("nof_samples", "<u4"),  # Redundant with page size? Only 512 instead of 10 ?
    ("event_count", "<i4"),
])

eve_event_header = np.dtype([
    ("event_size", "u4"),
    ("event_type", "<u4"),
    ("event_timestamp", "<i4"),
])

eve_signal_header = np.dtype([
    ("nsamp", "<u4"),
    ("page_size", "<u4"),
    ("event_size", "<u4"),
    # TODO: separate this header into smaller fields. numpy doesnt support 24bit numbers aka "<u3" or smaller structures
    # than 1 byte. Need to fix this
    ("board_res_0_pattern_channelmask", "<u4"),
    ("reserved_eventcounter", "<u4"),
    ("trigger_time_tag", "<i4"),
])

eve_signal_header_unpacked_noZLE = np.dtype([
    ("nsamp", "<u4"),
    ("page_size", "<u4"),
    ("event_size", "<u4"),
    ("board", "<u1"),
    ("res_0", "<u1"),
    ("pattern", "<u2"),
    ("channel_mask", "<u1"),
    ("reserved", "<u1"),
    ("event_counter", "<u4"),
    ("trigger_time_tag", "<u4")
])

def header_unpacker(raw_header):
    # unpacking
    """
    Header format for ZLE DISABLED!!!
    See caen-v1724 manual page 27

    4bits <1010> 28bits <EVENT SIZE>                                                word 1
    5bits <BOARD ID> 3bits <RES> <0> 16bits <PATTERN> 8bits <CHANNEL MASK>          word 2
    8bits <reserved> 24bits <EVENT COUNTER>                                         word 3
    32bits <TRIGGER TIME TAG>                                                       word 4

    """
    unpacked_header = np.zeros(1, dtype=eve_signal_header_unpacked_noZLE)[0]
    unpacked_header["nsamp"] = raw_header["nsamp"]
    unpacked_header["page_size"] = raw_header["page_size"]
    unpacked_header["event_size"] = raw_header[
                                        "event_size"] & 0xfffffff  # throwing leading 1010 from first header word away
    unpacked_header["board"] = (raw_header["board_res_0_pattern_channelmask"] >> 27) & 0x1f  # selecting only board bits
    unpacked_header["res_0"] = (
                                   raw_header[
                                       "board_res_0_pattern_channelmask"] >> 24) & 0x7  # will probably never be used
    unpacked_header["pattern"] = (raw_header["board_res_0_pattern_channelmask"] >> 8) & 0xffff  # selecting pattern bits
    unpacked_header["channel_mask"] = (raw_header[
                                           "board_res_0_pattern_channelmask"]) & 0xff  # selecting channel mask bits
    unpacked_header["reserved"] = (raw_header["reserved_eventcounter"] >> 24) & 0xff
    unpacked_header["event_counter"] = raw_header["reserved_eventcounter"] & 0xffffff
    unpacked_header["trigger_time_tag"] = raw_header["trigger_time_tag"]
    return unpacked_header


class EveInput(InputFromFolder):
    file_extension = 'eve'

    def __init__(self, config_values, processor):
        super().__init__(config_values, processor)
        self.event_number = 0

    def open(self, filename):
        """Opens an EVE file so we can start reading events"""
        print("Opening .eve file")
        self.current_evefile = open(filename, "rb")

        # Read in the file metadata
        self.file_metadata = np.fromfile(self.current_evefile, dtype=eve_file_header, count=1)[0]
        fmd = np.fromfile(self.current_evefile, dtype=eve_event_header, count=1)[0]
        # self.file_metadata = header_unpacker(self.file_metadata)
        self.file_caen_pars = np.fromfile(self.current_evefile, dtype=eve_caen1724_par_t, count=1)[0]
        first, last = self.get_first_and_last_event_number(filename)
        #print(self.event_positions)
        self.start_time = self.file_metadata["timestamp"]
        self.sample_duration = 10 * units.ns
        self.stop_time = int(
            self.start_time + self.file_caen_pars["nof_samples"] * self.sample_duration)
        """
        # Handle for special case of last EVE file
        # Index size is larger than the actual number of events written:
        # The writer didn't know how many events there were left at s
        if self.file_metadata['events_in_file'] < self.file_metadata['event_index_size']:
            self.log.info(
                ("The EVE file claims there are %d events in the file, "
                 "while the event position index has %d entries. \n"
                 "Is this the last EVE file of a dataset?") %
                (self.file_metadata['events_in_file'], self.file_metadata['event_index_size'])
            )
            self.event_positions = self.event_positions[:self.file_metadata['events_in_file']]
        """

    def get_first_and_last_event_number(self, filename):
        """Return the first and last event number in file specified by filename"""
        print("getting first and last event number")
        with open(filename, 'rb') as evefile:
            evefile.seek(0,2)
            filesize=evefile.tell()
            evefile.seek(0,0)
            positions = []
            fmd = np.fromfile(evefile, dtype=eve_file_header, count=1)[0]
            #print(fmd['byte_order'], fmd['version'], fmd['buffsize'], fmd["timestamp"], [hex(z) for z in fmd["not_used"]])
            j= 0
            while evefile.tell()< filesize: # maybe better with while(true) and try catch IndexOutofBounds for performance reasons
                fmd = np.fromfile(evefile, dtype=eve_event_header, count=1)[0]
                evefile.seek(fmd["event_size"]*4-12,1)
                #print(evefile.tell())
                #np.fromfile(evefile,dtype=np.uint32, count=1)[0]
                j += 1
                positions.append(evefile.tell())
            print("There are %d events in this file"%j)
            #print(positions)
            self.event_positions = positions
            return 1, j

    def close(self):
        """Close the currently open file"""
        print("Closing .eve file")
        #self.current_evefile.close()

    def get_single_event_in_current_file(self, event_position=1):
        # Seek to the requested event
        # print(self.event_positions)
        self.current_evefile.seek(self.event_positions[event_position])
        # self.current_evefile.seek(event_position, whence=io.SEEK_CUR)
        # Read event event header, check if it is a real data event or file event header or something different.
        event_event_header = np.fromfile(self.current_evefile, dtype=eve_event_header, count=1)[0]
        if event_event_header['event_type'] not in [3, 4]:  # 3 = signal event, 4 = header event. Do others occur?
            raise NotImplementedError("Event type %i not yet implemented!"
                                      % event_event_header['event_type'], self.current_evefile.tell())
        if event_event_header['event_type'] == 4:
            # it might be possible to get another event header along with caen1724.par stuff
            print("Unexpected event header at this position, trying to go on")
            self.file_caen_pars = np.fromfile(self.current_evefile, dtype=eve_caen1724_par_t, count=1)[0]
            event_event_header = np.fromfile(self.current_evefile, dtype=eve_event_header, count=1)[0]

        # Start building the event
        event = Event(
            n_channels=self.file_caen_pars["chan_active"].sum(),  # never trust the config file
            start_time=int(
                event_event_header['event_timestamp'] * units.s  # +
                # event_layer_metadata['utc_time_usec'] * units.us
            ),
            sample_duration=int( 10 * units.ns),
            # 10 ns is the inverse of the sampling  frequency 10MHz
            length=self.file_caen_pars['nof_samples']  # nof samples per each channel and event
        )

        event.dataset_name = self.current_filename  # now metadata available
        # as eve files do not have event numbers just count them
        self.event_number += 1
        event.event_number = self.event_number
        if self.file_caen_pars['zle'] == 0:
            # Zero length encoding disabled
            # Data is just a big bunch of samples from one channel, then next channel, etc
            # unless board's last channel is read. Then signal header from next board and then again data
            # Each channel has an equal number of samples.
            for i, board_i in enumerate(self.file_caen_pars["chan_active"]):
                if board_i.sum() == 0:  # if no channel is active there should be no signal header of the current board TODO: Check if that is really the case!
                    continue  # skip the current board
                event_signal_header = np.fromfile(self.current_evefile, dtype=eve_signal_header, count=1)[0]
                event_signal_header = header_unpacker(event_signal_header)
                for ch_i, channel in enumerate(board_i):
                    if channel == 0:
                        continue  # skip unused channels
                    chdata = []
                    print(event_signal_header["page_size"])
                    for j in range(int(event_signal_header[
                                           "page_size"] / 2)):  # divide by 2 because there are two samples in each word
                        data_word = np.fromfile(self.current_evefile, dtype=np.uint32, count=1)[0]
                        # print("entry:\t", i, "\t" ,a & 0x7fffffff, "\t", format(a,"#01b"),"\t",  a & 0x3fff, "\t", (a>> 16) &0x3fff)
                        chdata.append(data_word & 0x3fff)  # first sample in 4byte word
                        chdata.append((data_word >> 16) & 0x3fff)  # second sample in 4byte word
                    # chdata = np.array(chdata)
                    event.pulses.append(Pulse(
                        channel=ch_i + 8 * i,
                        left=0,# int(event_signal_header["trigger_time_tag"]),
                        raw_data=chdata
                    ))
                if i == 1:  # select trigger time tag from second board only
                    event.start_time += event_signal_header["trigger_time_tag"]
            print(hex(np.fromfile(self.current_evefile, dtype=np.uint32, count=1)[0]))

        elif self.file_caen_pars['zle'] == 0:
            print("This should not be executed yet")
            # ZLE has the advantage to push down the file sizes by a huge factor.
            # The downside is an increased complexity for the stored data
            # TODO: write a reading in routine
            # TODO: Test that routine
            for i, board_i in enumerate(self.file_caen_pars["chan_active"]):
                if board_i.sum() == 0:  # if no channel is active there should be no signal header of the current board TODO: Check if that is really the case!
                    continue  # skip the current board
                event_signal_header = np.fromfile(self.current_evefile, dtype=eve_signal_header, count=1)[0]
                event_signal_header = header_unpacker(event_signal_header)
                for ch_i, channel in enumerate(board_i):
                    if channel == 0:
                        continue  # skip unused channels
                    # TODO: read Size and Cwords
                    chdata = []
                    channel_size = np.fromfile(self.current_evefile, dtype=np.uint32, count=1)[0]
                    begin = self.current_evefile.tell()
                    for j in range(channel_size/2):     # divide by 2 because there are two samples in each
                        cword = np.fromfile(self.current_evefile, dtype=np.uint32, count=1)[0]
                        if cword > 2e9: # TODO: proper equivalent to reading first control bit which determines signal or zeros
                            chdata.extend(np.zeros((cword - 0x80000000)*2))     # fill chdata with zeros
                            j += (cword - 0x80000000)
                        else:
                            for l in range(cword):
                                data_word = np.fromfile(self.current_evefile, dtype=np.uint32, count=1)[0]
                                # print("entry:\t", i, "\t" ,a & 0x7fffffff, "\t", format(a,"#01b"),"\t",  a & 0x3fff, "\t", (a>> 16) &0x3fff)
                                chdata.append(data_word & 0x3fff)  # first sample in 4byte word
                                chdata.append((data_word >> 16) & 0x3fff)  # second sample in 4byte word
                    # chdata = np.array(chdata)
                    event.pulses.append(Pulse(
                        channel=ch_i + 8 * i,
                        left=0,
                        raw_data=chdata
                    ))
                if i == 1:  # select trigger time tag from second board only
                    event.start_time += event_signal_header["trigger_time_tag"]
            print(hex(np.fromfile(self.current_evefile, dtype=np.uint32, count=1)[0]))

        # TODO: Check we have read all data for this event
        if event_position != len(self.event_positions) - 1:
            current_pos = self.current_evefile.tell()
            should_be_at_pos = self.event_positions[event_position + 1]
            if current_pos != should_be_at_pos:
                raise RuntimeError("Error during XED reading: after reading event %d from file "
                                   "(event number %d) we should be at position %d, but we are at position %d!" % (
                                       event_position, event.event_number, should_be_at_pos, current_pos))

        return event

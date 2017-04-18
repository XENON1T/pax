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

import numpy as np

from pax import units
from pax.datastructure import Event, Pulse
from pax.FolderIO import InputFromFolder

"""  File header provided by FPPGui. This has only to be read once. Byte order is a relict from times where "Big Endian"
processors where not unusual. See Wikipedia "byte order" about that topic in necessary.
"""
eve_file_header = np.dtype([
    ("byte_order", "<u4"),
    ("version", "<u4"),
    ("buffsize", "<u4"),
    ("timestamp", "<u4"),
    ('first_event_number', "<u4"),
    ("not_used", "<3u4"),
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

    ("Channel_active", "<(20,8)i4"),
    ("Channel_DAC", "<(20,8)u4"),
    ("Thresholds", "<(20,8)i4"),
    ("ZS_Thresholds", "<(20,8)i4"),
    ("Zero_offset", "<(20,8)i4"),

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

    def _write_caen_pars(self, filename):
        def write_line(line, caenname=None):

            if caenname is None:
                caenfile.write(line + "\t\t%d\n" % self.file_caen_pars[line])
            else:
                caenfile.write(caenname + "\t\t%d\n" % self.file_caen_pars[line])
        with open(filename + ".caenpars", "w") as caenfile:
            caenfile.write("# This file was autogenerated while processing the file\n# %s\n" % filename)
            write_line("no_mod", "No_of_Modules")
            caenfile.write(
                    "base address\t0x%x   0x%x\n" % (self.file_caen_pars["base"][0], self.file_caen_pars["base"][1]))
            write_line("downsample_factor")
            write_line("page_size")
            write_line("post_trigger_samples")
            write_line("enable_external_trig")
            write_line("trigger_logic")
            write_line("trigger_coinc_level")
            write_line("zle", "zero_length_encoding")
            write_line("zle_logic")
            write_line("zle_nlbk")
            write_line("zle_nlfwd")
            write_line("integrate_signals")
            write_line("substract_offset")
            # first board
            caenfile.write("Module 1\n")
            caenfile.write("\tChannel_active" + (8 * "\t %d") % tuple(self.file_caen_pars["Channel_active"][0]) + "\n")

            caenfile.write("\tChannel_DAC" + (8 * "\t 0x%x") % tuple(self.file_caen_pars["Channel_DAC"][0]) + "\n")

            caenfile.write("\tThresholds" + (8 * "\t %d") % tuple(self.file_caen_pars["Thresholds"][0]) + "\n")

            caenfile.write(
                    "\tZS_Thresholds" + (8 * "\t %d") % tuple(self.file_caen_pars["ZS_Thresholds"][0]) + "\n")

            caenfile.write("\tZero_offset" + (8 * "\t %d") % tuple(self.file_caen_pars["Zero_offset"][0]) + "\n")
            # second board
            caenfile.write("Module 2\n")
            caenfile.write("\tChannel_active" + (8 * "\t %d") % tuple(self.file_caen_pars["Channel_active"][1]) + "\n")

            caenfile.write("\tChannel_DAC" + (8 * "\t 0x%x") % tuple(self.file_caen_pars["Channel_DAC"][1]) + "\n")

            caenfile.write("\tThresholds" + (8 * "\t %d") % tuple(self.file_caen_pars["Thresholds"][1]) + "\n")

            caenfile.write(
                    "\tZS_Thresholds" + (8 * "\t %d") % tuple(self.file_caen_pars["ZS_Thresholds"][1]) + "\n")

            caenfile.write("\tZero_offset" + (8 * "\t %d") % tuple(self.file_caen_pars["Zero_offset"][1]) + "\n")
            write_line("nof_samples")
            write_line("event_count")

    def open(self, filename):
        """Opens an EVE file so we can start reading events"""
        # print("Opening .eve file: %s" % filename.split("/")[-1])
        self.current_evefile = open(filename, "rb")

        # Read in the file metadata
        self.file_metadata = np.fromfile(self.current_evefile, dtype=eve_file_header, count=1)[0]
        fmd = np.fromfile(self.current_evefile, dtype=eve_event_header, count=1)[0]
        if (fmd["event_type"] == 4):
            # self.file_metadata = header_unpacker(self.file_metadata)
            self.file_caen_pars = np.fromfile(self.current_evefile, dtype=eve_caen1724_par_t, count=1)[0]
            try:
                self._write_caen_pars(filename)
            except PermissionError:
                self.log.warning("No permission to write .caenpars file in directory of input file")
            # with open(filename+".caen_par", "w") as caen_pars_file:
            #    for str in self.file_caen_pars:
            #       caen_pars_file.write(str(self.file_caen_pars))
        else:
            if self.file_caen_pars == None:
                raise LookupError(
                        "self.file_caen_pars has not been set, yet! Is caen_event not in first file of the file list?")
            print("no caen event in this file. Sticking with old parameters")
        self.get_event_number_info(filename)
        self.start_time = self.file_metadata["timestamp"]
        self.sample_duration = 10 * units.ns
        self.stop_time = int(
                self.start_time + self.file_caen_pars["nof_samples"] * self.sample_duration)

    def get_event_number_info(self, filename):
        """Return the first, last and total event numbers in file specified by filename"""
        print("getting first, last and total event number")
        with open(filename, 'rb') as evefile:
            evefile.seek(0, 2)
            filesize = evefile.tell()
            evefile.seek(0, 0)
            positions = []
            fmd = np.fromfile(evefile, dtype=eve_file_header, count=1)[0]
            if fmd[
                "first_event_number"] != 45054:  # before unused field is unequal to 45054 in newer versions of fppdaq
                # newer versions of fppgui support an basic event_counter for multiple files
                j = fmd["first_event_number"]
                # print(j)
            else:
                j = 0
            while evefile.tell() < filesize:
                # maybe better with while(true) and try catch IndexOutofBounds for performance reasons
                fmd = np.fromfile(evefile, dtype=eve_event_header, count=1)[0]
                evefile.seek(fmd["event_size"] * 4 - 12, 1)
                positions.append(evefile.tell())
            print("There are events %d-%d in this file" % (j, j + len(positions)))
            # if j == 0:
            #    positions.pop(0)  # throw away first event as it is the cae1724_par event

            self.event_positions = positions[3:]

            return j, j + len(positions) - 2-3, len(positions)

    def close(self):
        """Close the currently open file"""
        print("Closing .eve file")
        # self.current_evefile.close()

    def get_single_event_in_current_file(self, event_number=1):
        # Seek to the requested event
        event_position = event_number - self.file_metadata['first_event_number']
        self.current_evefile.seek(self.event_positions[event_position])
        # self.current_evefile.seek(event_position, whence=io.SEEK_CUR)
        # Read event event header, check if it is a real data event or file event header or something different.
        event_event_header = np.fromfile(self.current_evefile, dtype=eve_event_header, count=1)[0]

        """ commented these lines  as they never happened !!!
        if event_event_header['event_type'] not in [3, 4]:  # 3 = signal event, 4 = header event. Do others occur?
            raise NotImplementedError("Event type %i not yet implemented!"
                                      % event_event_header['event_type'], self.current_evefile.tell())

        if event_event_header['event_type'] == 4:
            # it might be possible to get another event header along with caen1724.par stuff
            self.log.error("Unexpected event header at this position, trying to go on")
            self.file_caen_pars = np.fromfile(self.current_evefile, dtype=eve_caen1724_par_t, count=1)[0]
            event_event_header = np.fromfile(self.current_evefile, dtype=eve_event_header, count=1)[0]
        """
        # Start building the event
        event = Event(
                n_channels=14,  # never trust the config file
                start_time=int(
                        event_event_header['event_timestamp'] * units.s  # +
                        # event_layer_metadata['utc_time_usec'] * units.us
                ),
                sample_duration=int(10 * units.ns),
                # 10 ns is the inverse of the sampling  frequency 10MHz
                length=self.file_caen_pars['nof_samples']  # nof samples per event
        )

        event.dataset_name = self.current_filename  # now metadata available
        # as eve files do not have event numbers just count them
        event.event_number = event_number
        if self.file_caen_pars['zle'] == 0:
            # Zero length encoding disabled
            # Data is just a big bunch of samples from one channel, then next channel, etc
            # unless board's last channel is read. Then signal header from next board and then again data
            # Each channel has an equal number of samples.
            for board_i, channels_active in enumerate(self.file_caen_pars["Channel_active"]):
                if channels_active.sum() == 0:  # if no channel is active there should be no signal header of the current board TODO: Check if that is really the case!
                    continue  # skip the current board
                event_signal_header_raw = np.fromfile(self.current_evefile, dtype=eve_signal_header, count=1)[0]
                event_signal_header = header_unpacker(event_signal_header_raw)
                # if board_i == 0:
                #    event.start_time = int(event.start_time + event_signal_header["trigger_time_tag"]*10*units.ns)
                for ch_i, channel_is_active in enumerate(channels_active):
                    if channel_is_active == 0:
                        continue  # skip unused channels
                    chdata = np.fromfile(self.current_evefile, dtype=np.int16,
                                         count=int(event_signal_header["page_size"]))
                    event.pulses.append(Pulse(
                            channel=self.config["Channel_to_PMT_mapping"][ch_i + 8 * board_i],
                            left=0,
                            raw_data=np.array(chdata, dtype=np.int16)
                    ))

        elif self.file_caen_pars['zle'] == 1:
            # print(len(self.file_caen_pars["chan_active"]))
            for board_i, channels_active in enumerate(self.file_caen_pars["Channel_active"]):
                # Skip nonexistent board
                if channels_active.sum() == 0:  # if no channel is active there should be no signal header of the current board TODO: Check if that is really the case!
                    continue  # skip the current board

                event_signal_header_raw = np.fromfile(self.current_evefile,
                                                      dtype=eve_signal_header,
                                                      count=1)[0]
                # event_signal_header = header_unpacker(event_signal_header_raw)
                # channel_mask = event_signal_header["channel_mask"]
                channel_mask = event_signal_header_raw["board_res_0_pattern_channelmask"] & 0xff
                channels_included = [i for i in range(8)
                                     if (2 ** i & channel_mask) > 0]

                for ch_i in channels_included:  # enumerate(channels_active):
                    position = self.current_evefile.tell()
                    channel_size = np.fromfile(self.current_evefile, dtype=np.uint32, count=1)[0]
                    sample_position = 0
                    while self.current_evefile.tell() < position + channel_size * 4:
                        cword = np.fromfile(self.current_evefile, dtype=np.uint32, count=1)[0]
                        if cword < 0x80000000:  # if cword is less than 0x80000000 waveform is below zle threshold
                            # skip word
                            sample_position += 2 * cword
                            continue
                        else:
                            chdata = np.fromfile(self.current_evefile, dtype=np.int16, count=2 * (cword - 0x80000000))
                            event.pulses.append(Pulse(
                                    channel=self.config["Channel_to_PMT_mapping"][ch_i + 8 * board_i],
                                    left=sample_position,
                                    raw_data=chdata
                            ))
                            sample_position += 2 * (cword & 512)  # 1048575 == 2**20 -1

        # TODO: Check we have read all data for this event
        # affe = hex(np.fromfile(self.current_evefile, dtype=np.uint32, count=1)[0])
        # self.current_evefile.seek(4, 1)
        """
        if affe != '0xaffe':
            print("WARNING : EVENT DID NOT END WITH 0XAFFE!! INSTEAD IT ENDED WITH ", affe)
        if event_position != len(self.event_positions) - 1:
            current_pos = self.current_evefile.tell()
            should_be_at_pos = self.event_positions[event_position + 1]
            if current_pos != should_be_at_pos:
                raise RuntimeError("Error during XED reading: after reading event %d from file "
                                   "(event number %d) we should be at position %d, but we are at position %d!" % (
                                       event_position, event.event_number, should_be_at_pos, current_pos))
        """
        return event

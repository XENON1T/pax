__author__ = 'axel'
import numpy as np
from pax.plugins.io import EVE_file

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
    ("no_mod", "<i4"),
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
    ("event_size", "<u4"),
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

eve_signal_header_unpacked_noZLE = np.dtype = ([
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
    unpacked_header["res_0"] = (raw_header["board_res_0_pattern_channelmask"] >> 24) & 0x7  # will probably never be used
    unpacked_header["pattern"] = (raw_header["board_res_0_pattern_channelmask"] >> 8) & 0xffff  # selecting pattern bits
    unpacked_header["channel_mask"] = (raw_header["board_res_0_pattern_channelmask"]) & 0xff  # selecting channel mask bits
    unpacked_header["reserved"] = (raw_header["reserved_eventcounter"] >> 24) & 0xff
    unpacked_header["event_counter"] = raw_header["reserved_eventcounter"] & 0xffffff
    unpacked_header["trigger_time_tag"] = raw_header["trigger_time_tag"]
    return unpacked_header

#eve = EVE_file
#eve.open("Background_150528_no_ZLE_1K_14channels.eve")



with open("Background_150528_no_ZLE.eve", 'rb') as evefile:
    evefile.seek(0,2)
    filesize=evefile.tell()
    evefile.seek(0,0)
    fmd = np.fromfile(evefile, dtype=eve_file_header, count=1)[0]
    print(fmd['byte_order'], fmd['version'], fmd['buffsize'], fmd["timestamp"], [hex(z) for z in fmd["not_used"]  ])
    fmd = np.fromfile(evefile, dtype=eve_event_header, count=1)[0]
    print(fmd)
    #evefile.seek(4*856,1)
    caenpars = np.fromfile(evefile, dtype=eve_caen1724_par_t, count=1)[0]
    print("evefile.tell(): ", evefile.tell())
    nof_signals = 0
    # while j<2: # maybe better with while(true) and try catch IndexOutofBounds for performance reasons
    #     fmd = np.fromfile(evefile, dtype=eve_event_header, count=1)[0]
    #     #evefile.seek(fmd["event_size"]*4-16,1)
    #     for i, board_i in enumerate(caenpars["chan_active"]):
    #         if board_i.sum() == 0:  # if no channel is active there should be no signal header of the current board TODO: Check if that is really the case!
    #             continue  # skip the current board
    #         event_signal_header = np.fromfile(evefile, dtype=eve_signal_header, count=1)[0]
    #         event_signal_header = header_unpacker(event_signal_header)
    #         for ch_i, channel in enumerate(board_i):
    #             if channel == 0:
    #                 continue  # skip unused channels
    #             # TODO: read Size and Cwords
    #             chdata = []
    #             channel_size = np.fromfile(evefile, dtype=np.uint32, count=1)[0]
    #             begin = evefile.tell()
    #             for k in range(int(channel_size/2)):     # divide by 2 because there are two samples in each
    #                 cword = np.fromfile(evefile, dtype=np.uint32, count=1)[0]
    #                 print(cword, cword &0x80000000)
    #                 if cword > 2147483648: # TODO: proper equivalent to reading first control bit which determines signal or zeros
    #                     for l in range(cword - 0x80000000):
    #                         data_word = np.fromfile(evefile, dtype=np.uint32, count=1)[0]
    #                         # print("entry:\t", i, "\t" ,a & 0x7fffffff, "\t", format(a,"#01b"),"\t",  a & 0x3fff, "\t", (a>> 16) &0x3fff)
    #                         chdata.append(data_word & 0x3fff)  # first sample in 4byte word
    #                         chdata.append((data_word >> 16) & 0x3fff)  # second sample in 4byte word
    #                         # chdata = np.array(chdata)
    #                 else:
    #                     chdata.extend(np.zeros(cword*2))   # fill chdata with zeros
    #                     k += cword
    #             event.pulses.append(Pulse(
    #                 channel=ch_i + 8 * i,
    #                 left=0,
    #                 raw_data=chdata
    #             ))
    #         if i == 1:  # select trigger time tag from second board only
    #             event.start_time += event_signal_header["trigger_time_tag"]
    #    print("Hex hex: ", hex(np.fromfile(self.current_evefile, dtype=np.uint32, count=1)[0]))
    while(evefile.tell()<filesize):
        fmd = np.fromfile(evefile, dtype=eve_event_header, count=1)[0]
        if fmd["event_type"]==4:
            print("unexpected event 4")
            break
        #print("event header: ", fmd)
        fmd = np.fromfile(evefile, dtype=eve_signal_header, count=1)[0]
        unpacked_header = header_unpacker(fmd)
        print("board 1 trigger time tag: ", unpacked_header["trigger_time_tag"])
        #print("raw signal header: ",fmd)
        #print("unpacked header: ",unpacked_header)
        for i in range(unpacked_header["event_size"]-4): # minus 4 because header is counted with
            a = np.fromfile(evefile, dtype=np.uint32,count=1)[0]
            #    print("entry:\t", i, "\t" ,a & 0x7fffffff, "\t", format(a,"#01b"),"\t",  a & 0x3fff, "\t", (a>> 16) &0x3fff)
        fmd=np.fromfile(evefile, dtype=eve_signal_header, count=1)[0]
        #print("raw signal header: ",fmd)
        unpacked_header = header_unpacker(fmd)
        print("board 2 trigger time tag: ", unpacked_header["trigger_time_tag"])
        #print("unpacked header: ", unpacked_header)
        for i in range(unpacked_header["event_size"]-4): # minus 4 because header is counted with
            a = np.fromfile(evefile, dtype=np.uint32,count=1)[0]
            #    print("entry:\t", i, "\t" ,a & 0x7fffffff, "\t", format(a,"#01b"),"\t",  a & 0x3fff, "\t", (a>> 16) &0x3fff)
        print(hex(np.fromfile(evefile,dtype=np.uint32, count=1)[0]))
        nof_signals += 1
    print("nof signals: ", nof_signals)


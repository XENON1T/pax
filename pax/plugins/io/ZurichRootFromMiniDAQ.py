"""
This plug-in reads raw waveform data from root files made in Zurich and uses
the PAX singlePE peakfinder

!!PyROOT must be working with py3.

"""

import numpy as np
import os
import time
import ROOT
from pax import plugin, units
from pax.datastructure import Event, Occurrence


class ZurichPMTTestRootFileReader(plugin.InputPlugin):

    def startup(self):
        # Open root tree
        self.filename = self.config.get('input_name',)
        #    '/home/xedaq/DATA/141215_1522/mDAX_141215_1522_00000.root')
        if not os.path.exists(self.filename):
            raise ValueError("File %s does not exists" % self.filename)
        self.dataset_name = os.path.basename(self.filename)

        # Open Root file and get tree
        self.root_file = ROOT.TFile(self.filename)
        self.root_tree = self.root_file.Get("t1")

        self.number_of_events = self.root_tree.GetEntries()
        self.log.debug("Number of events found in tree:",
                       self.number_of_events)

    def get_single_event(self, event_number):

        # open waveform
        # the t.wf1 object is a buffer of ints.
        # The list() is one of the few options to easily get the data out

        now = time.time() * units.s
        event_length = self.config.get('event_length', 500)
        event = Event(
            n_channels=self.config['n_channels'],
            start_time=int(now),
            sample_duration=self.config['sample_duration'],
            length=event_length
        )
        event.dataset_name = self.dataset_name
        event.event_number = event_number

        for channel in range(0, 5):

            self.root_tree.GetEntry(event_number)
            # +1 for one->zero based indexing
            weird_wv_buffer = getattr(self.root_tree, "wf" + str(channel + 1))
            raw_data = np.array(list(weird_wv_buffer), dtype=np.int16)

            if len(raw_data) != event_length:
                raise ValueError('Raw data for channel %s in event %s is %s samples long, should be %s' % (
                    channel, event_number, len(raw_data), event_length))

            event.occurrences.append(Occurrence(channel=channel,
                                                left=0,
                                                raw_data=raw_data))

        return event

    def get_events(self):
        for event_number in range(self.number_of_events):
            yield self.get_single_event(event_number)

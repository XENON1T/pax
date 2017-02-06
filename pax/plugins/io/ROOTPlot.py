"""
Event display using ROOT framework
"""
import ROOT #sigh
import os
import numpy as np
import pytz
import datetime

from pax import plugin, units


def epoch_to_human_time(timestamp):
        # Unfortunately the python datetime, explicitly choose UTC timezone (default is local)
        tz = pytz.timezone('UTC')
        return datetime.datetime.fromtimestamp(timestamp / units.s, tz=tz).strftime("%Y/%m/%d, %H:%M:%S")  

class ROOTWaveformDisplay(plugin.OutputPlugin):
    
    def startup(self):
        self.output_dir = self.config['output_name']
        self.samples_to_us = self.config['sample_duration'] / units.us
                
        if self.output_dir is None:
            raise RuntimeError("You must supply an output directory for ROOT display")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        ROOT.gROOT.SetBatch(True)
        ROOT.gStyle.SetOptStat(0000000);
        ROOT.gStyle.SetOptFit(1100);
        ROOT.gStyle.SetTitleFillColor(0);
        ROOT.gStyle.SetTitleBorderSize(0);
        ROOT.gStyle.SetStatColor(0);       

        self.peak_colors = {"lone_hit": 40, "s1": 4, "s2": 2, "unknown": 29, "noise": 30}
        
    def write_event(self, event):
        
        # Super simple. Just make a sum waveform and dump it
        name = "test.root"
        outfile = os.path.join(self.output_dir, name)
        self.outfile = ROOT.TFile(outfile, 'RECREATE')

        # Borrow liberally from xerawdp
        titlestring = 'Event %s from %s Recorded at %s UTC, %09d ns' % (
            event.event_number, event.dataset_name,
            epoch_to_human_time(event.start_time / units.ns),
            (event.start_time/units.ns) % (units.s))
        namestring = '%s_%s' % (event.dataset_name, event.event_number)
        win = ROOT.TCanvas(namestring, titlestring)
        #win.Divide(1,2);
        win.cd();        
        win.SetFillColor(0);
        win.SetBorderMode(0);
        win.SetFrameBorderMode(0);

        # Plot Sum Waveform
        win.cd()
        sum_waveform_x = np.arange(0, event.length()-1, 
                                   dtype=np.float32)

        for w in self.config['waveforms_to_plot']:
            waveform = event.get_sum_waveform(w['internal_name'])
            h = ROOT.TH1D("g",titlestring,len(sum_waveform_x),0,
                          self.samples_to_us*(len(sum_waveform_x)-1))
            for i, s in enumerate(waveform.samples):
                h.SetBinContent(i, s)
            h.SetStats(0)
            h.GetXaxis().SetTitle("Time [#mus]")
            h.GetYaxis().SetTitle("Amplitude [pe/bin]")
            h.Draw("same")

        # Add peak labels and pretty boxes
        peaks = {}
        boxes = []
        text = []
        s1 = 0
        s2 = 0
        for peak in event.peaks:
            if peak.type not in peaks:
                peaks[peak.type] = {"x": [], "y": []}
            peaks[peak.type]["x"].append(peak.index_of_maximum * self.samples_to_us)
            peaks[peak.type]["y"].append(peak.height)

            if peak.type == 's1' or peak.type == 's2':
                boxes.append(ROOT.TBox(peak.left*self.samples_to_us, 0,
                                       peak.right*self.samples_to_us, peak.height))
                boxes[len(boxes)-1].SetFillStyle(0)
                boxes[len(boxes)-1].SetLineColorAlpha(self.peak_colors[peak.type], 0.2)
                boxes[len(boxes)-1].SetLineStyle(3)
                #boxes[len(boxes)-1].SetLineWidth(0)
                boxes[len(boxes)-1].Draw("same")            

        # Labels, want them sorted
        for i, peak in enumerate(event.s2s()):
            if i > 4:
                break
            label = "s2["+str(s2)+"]: " + "{:.2f}".format(peak.area)
            if len(peak.contributing_channels)<5:
                label += str(peak.contributing_channels)
            else:
                label += "["+str(len(peak.contributing_channels))+" channels]"
            s2+=1              
            text.append(ROOT.TText(peak.left*self.samples_to_us, peak.height, label))
            text[len(text)-1].SetTextColor(self.peak_colors[peak.type])
            text[len(text)-1].SetTextSize(0.02)
            text[len(text)-1].Draw("same")
        for i, peak in enumerate(event.s1s()):
            if i > 4:
                break
            label = "s1["+str(s1)+"]: " + "{:.2f}".format(peak.area)
            if len(peak.contributing_channels)<5:
                label += str(peak.contributing_channels)
            else:
                label += "["+str(len(peak.contributing_channels))+" channels]"  
            s1+=1
            text.append(ROOT.TText(peak.left*self.samples_to_us, peak.height, label))
            text[len(text)-1].SetTextColor(self.peak_colors[peak.type])
            text[len(text)-1].SetTextSize(0.02)
            text[len(text)-1].Draw("same")      

                                
               
        # Polymarkers don't like being overwritten I guess, so make a container
        pms = {}
        for peaktype, plist in peaks.items():            
            pms[peaktype] = ROOT.TPolyMarker();
            pms[peaktype].SetMarkerStyle(23);
            if peaktype in self.peak_colors:
                pms[peaktype].SetMarkerColor(self.peak_colors[peaktype]);
            else:
                pms[peaktype].SetMarkerColor(0)
            pms[peaktype].SetMarkerSize(1.1);            
            pms[peaktype].SetPolyMarker(len(plist['x']), np.array(plist['x']), np.array(plist['y']))
            pms[peaktype].Draw("same")
            
        win.Update()
        self.outfile.cd()
        win.Write()
        #self.outfile.Write()
        self.outfile.Close()


            

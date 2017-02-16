"""
Event display using ROOT framework
"""
import ROOT  # sigh
import os
import numpy as np

import gzip
import shutil

from itertools import islice
from six import iteritems
from pax import plugin, units

from pax.plugins.plotting.Plotting import epoch_to_human_time


class ROOTSumWaveformDump(plugin.OutputPlugin):
    def startup(self):
        self.output_dir = self.config['output_name']
        self.samples_to_us = self.config['sample_duration'] / units.us

        if self.output_dir is None:
            raise RuntimeError("You must supply an output directory for ROOT display")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        ROOT.gROOT.SetBatch(True)

        self.considered_peak_types = ["s1", "s2", "unknown"]
        self.peak_colors = {"s1": 4, "s2": 2, "unknown": 29, "noise": 30}

        self.pmt_locations = np.array([[self.config['pmts'][ch]['position']['x'],
                                        self.config['pmts'][ch]['position']['y']]
                                       for ch in range(self.config['n_channels'])])
        self.hitpattern_limit_low = 1e-1
        self.hitpattern_limit_high = 1e4
        self.pmts = {array: self.config['channels_%s' % array] for array in ('top', 'bottom')}

        # ROOT does some weird stuff with memory. Don't ask. Don't think. Just obey.
        self.plines = []
        self.latexes = []

    def setRootGStyle(self):
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptFit(0)
        ROOT.gStyle.SetLegendBorderSize(0)
        ROOT.gStyle.SetLegendFillColor(0)

        ROOT.gStyle.SetCanvasColor(0)
        ROOT.gStyle.SetLegendTextSize(0.04)
        ROOT.gStyle.SetFrameBorderMode(0)
        ROOT.gStyle.SetPadBorderMode(0)

        ROOT.gStyle.SetPadColor(1)
        ROOT.gStyle.SetPadLeftMargin(0.05)
        ROOT.gStyle.SetPadRightMargin(0.02)
        ROOT.gStyle.SetPadBottomMargin(0.10)

        ROOT.gStyle.SetTitleFont(132, "t")
        ROOT.gStyle.SetLabelFont(132, "xyz")
        ROOT.gStyle.SetTitleFont(132, "xyz")

        self.colorwheel = [ROOT.kGray + 3, ROOT.kGray, ROOT.kMagenta + 2, ROOT.kRed + 2, ROOT.kGreen + 2]

    def write_event(self, event):

        self.setRootGStyle()

        namestring = '%s_%s' % (event.dataset_name, event.event_number)

        outfile_dotC = os.path.join(self.output_dir, "event_" + namestring + ".C")

        # Borrow liberally from xerawdp
        titlestring = 'Event %s from %s Recorded at %s UTC, %09d ns' % (
            event.event_number, event.dataset_name,
            epoch_to_human_time(event.start_time / units.ns),
            (event.start_time / units.ns) % (units.s))
        win = ROOT.TCanvas("canvas", titlestring, 1600, 600)

        # ROOT.TH1D: bin[0]=underflow, bin[-1]=overflow
        sum_waveform_x = np.arange(0, event.length() + 1,
                                   dtype=np.float32)

        leg = ROOT.TLegend(0.85, 0.75, 0.98, 0.90)
        leg2 = ROOT.TLegend(0.85, 0.60, 0.98, 0.75)
        leg.SetFillStyle(0)
        leg2.SetFillStyle(0)

        hlist = []
        ymin, ymax = 0, 0
        for jj, w in enumerate(self.config['waveforms_to_plot']):
            if jj > 8:
                break
            waveform = event.get_sum_waveform(w['internal_name'])
            hist_name = "g_{}".format(jj)
            hlist.append(ROOT.TH1D(hist_name, "", len(sum_waveform_x), 0,
                                   self.samples_to_us * (len(sum_waveform_x) - 1)))
            hlist[jj].SetLineColor(self.colorwheel[jj % len(self.colorwheel)])
            for i, s in enumerate(waveform.samples):
                hlist[jj].SetBinContent(i + 1, s)
            hlist[jj].SetLineWidth(1)

            if jj == 0:
                hlist[jj].SetTitle(titlestring)
                hlist[jj].GetXaxis().SetTitle("Time [#mus]")
                hlist[jj].GetYaxis().SetTitleOffset(0.5)
                hlist[jj].GetYaxis().SetTitleSize(0.05)
                hlist[jj].GetXaxis().SetTitleOffset(0.8)
                hlist[jj].GetXaxis().SetTitleSize(0.05)
                hlist[jj].GetYaxis().SetTitle("Amplitude [pe/bin]")
                hlist[jj].Draw("")
            else:
                hlist[jj].Draw("same")

            ymin = min(ymin, np.amin(waveform.samples))
            ymax = max(ymax, np.amax(waveform.samples))
            lhentry = leg.AddEntry(hlist[-1], w['plot_label'], "l")
            lhentry.SetTextFont(43)
            lhentry.SetTextSize(22)

        yoffset = 0.05 * ymax
        hlist[0].GetYaxis().SetRangeUser(ymin - yoffset, ymax + yoffset)

        # Add peak labels and pretty boxes
        # Semi-transparent colors ('SetLineColorAlpha') apparently won't work
        peaks = {}
        boxes = {}
        boxs1 = ROOT.TBox()
        boxs2 = ROOT.TBox()
        boxlh = ROOT.TBox()
        boxun = ROOT.TBox()
        boxes = {'s1':boxs1, 's2':boxs2, 'unknown':boxun}
        for peaktype, tbox in iteritems(boxes):
            tbox.SetLineColor(0)
            tbox.SetFillStyle(3003)
            tbox.SetFillColor(self.peak_colors[peaktype])

        vls1 = ROOT.TLine()
        vls2 = ROOT.TLine()
        vllh = ROOT.TLine()
        vlun = ROOT.TLine()
        vlines = {'s1':vls1, 's2':vls2, 'unknown':vlun}
        for peaktype, tline in iteritems(vlines):
            tline.SetLineColor(self.peak_colors[peaktype])
            tline.SetLineWidth(1)
            tline.SetLineStyle(1)

        for peak in event.peaks:
            if peak.type not in peaks:
                peaks[peak.type] = {"x": [], "y": []}
            peaks[peak.type]["x"].append(peak.index_of_maximum * self.samples_to_us)
            peaks[peak.type]["y"].append(peak.height)
            
            # Loop considered peaks and draw TBoxes and TLines:
            if peak.type in self.considered_peak_types:
                boxes[peak.type].DrawBox(peak.left * self.samples_to_us, 0, peak.right * self.samples_to_us, peak.height)
                vlines[peak.type].DrawLine(peak.left * self.samples_to_us, 0, peak.left * self.samples_to_us, peak.height)
                vlines[peak.type].DrawLine(peak.right * self.samples_to_us, 0, peak.right * self.samples_to_us, peak.height)
            

        # Polymarkers don't like being overwritten I guess, so make a container
        marker = []
        legmarker = []
        mstyle = 23
        msize = 1.7

        for peaktype, plist in iteritems(peaks):
            marker.append(ROOT.TMarker())
            # For whatever reasons only TPolyMarkers are displayed in different colors in the legend
            legmarker.append(ROOT.TPolyMarker())
            marker[-1].SetMarkerStyle(mstyle)
            marker[-1].SetMarkerSize(msize)
            legmarker[-1].SetMarkerStyle(mstyle)
            legmarker[-1].SetMarkerSize(msize)
            mcolor = 0
            if peaktype in self.peak_colors:
                mcolor = self.peak_colors[peaktype]

            if mcolor == 0:
                continue

            marker[-1].SetMarkerColor(mcolor)
            legmarker[-1].SetMarkerColor(mcolor)
            lmentry = leg2.AddEntry(legmarker[-1], "{}".format(peaktype), "P")
            lmentry.SetTextFont(43)
            lmentry.SetTextSize(22)
            for px, py in zip(plist['x'], plist['y']):
                marker[-1].DrawMarker(px, py)

        # Put histograms (in reversed order) above boxes and markers
        for hj in hlist[::-1]:
            hj.Draw("same")


        ## plot TLatex labels to indicate non-TPC peaks
        tlatex_nonTPC = ROOT.TLatex()
        tlatex_nonTPC.SetTextAlign(12)
        default_nonTPC_color = ROOT.kBlack
        tlatex_nonTPC.SetTextColor(default_nonTPC_color)
        tlatex_nonTPC.SetTextFont(43)
        tlatex_nonTPC.SetTextSize(24)

        peakdetails = event.get_peaks_by_type(desired_type='all', detector='all')
        for i, peak in enumerate(peakdetails):
            if peak.detector == 'tpc' or peak.detector == 'sum_wv':
                continue
            if peak.type.lower() in self.considered_peak_types:
                tlatex_nonTPC.SetTextColor(self.peak_colors[peak.type.lower()])
            else:
                tlatex_nonTPC.SetTextColor(default_nonTPC_color)
            label = "  {} in {}".format(peak.type.lower(), peak.detector.upper())
            tlatex_nonTPC.DrawLatex(peak.index_of_maximum * self.samples_to_us, peak.height, label)


        # Plot TLatex labels to interesting (i.e., largest) peaks
        maxNS1Labels = 3
        maxNS2Labels = 3
        maxNLHLabels = 3
        maxNUNLabels = 3

        if 'number_of_labeled_s1' in self.config:
            maxNS1Labels = int(self.config['number_of_labeled_s1'])
        if 'number_of_labeled_s2' in self.config:
            maxNS2Labels = int(self.config['number_of_labeled_s2'])
        if 'number_of_labeled_unknown' in self.config:
            maxNUNLabels = int(self.config['number_of_labeled_unknown'])

        maxLabels = {'s1':maxNS1Labels, 's2':maxNS2Labels, 'unknown':maxNUNLabels}

        tls1 = ROOT.TLatex()
        tls2 = ROOT.TLatex()
        tllh = ROOT.TLatex()
        tlun = ROOT.TLatex()
        tlatex = {'s1':tls1, 's2':tls2, 'unknown':tlun}
        
        for peaktype,tlab in iteritems(tlatex):
            tlab.SetTextAlign(12)
            tlab.SetTextColor(self.peak_colors[peaktype])
            tlab.SetTextFont(43)
            tlab.SetTextSize(24)

        # assessing peaks by type
        for peaktype in self.considered_peak_types:
            peakdetails = event.get_peaks_by_type(desired_type=peaktype)
            for i, peak in enumerate(peakdetails):

                if i >= maxLabels[peaktype]:
                    break

                label = "  {}[{}]:{:.1e} pe ".format(peaktype,i,peak.area)

                # Star the main interaction peaks
                if peaktype == 's1' and event.interactions[0].s1 >= 0:
                    if peak is event.peaks[event.interactions[0].s1]:
                        label = "  {}[{}]*:{:.1e} pe ".format(peaktype,i,peak.area)
                elif peaktype == 's2' and event.interactions[0].s2 >= 0:
                    if peak is event.peaks[event.interactions[0].s2]:
                        label = "  {}[{}]*:{:.1e} pe ".format(peaktype,i,peak.area)

                label = label.replace("e+0", "e")

                if len(peak.contributing_channels) < 5:
                    ## if there are less than 5 PMTs contributing to a peak, list them by name:
                    label += "{{{}}}".format(", ".join(["PMT{}".format(pmt) for pmt in map(str, peak.contributing_channels)]))
                else:
                    ## if there are 5 or more PMTs that contribute to a peak, print their total number only:
                    label += "({})".format(len(peak.contributing_channels))
                tlatex[peaktype].DrawLatex(peak.index_of_maximum * self.samples_to_us, peak.height, label)

        # Draw Legends and update canvas
        leg.Draw()
        leg2.Draw()
        win.Update()

        # Dump canvas to ROOT macro (.C file)
        win.SaveAs(outfile_dotC)
        # Compress ROOT macro (.C.gz) and remove original file:
        if 'compress_output' in self.config:
            if self.config['compress_output']:
                with open(outfile_dotC, 'rb') as f_in, gzip.open("{}.gz".format(outfile_dotC), 'wb') as f_out:
                   shutil.copyfileobj(f_in, f_out)
                os.remove(outfile_dotC)

        # Uncomment if .ROOT file is desired:
        # outfile = os.path.join(self.output_dir, "event_" + namestring + ".root")
        # self.outfile = ROOT.TFile(outfile, 'RECREATE')
        # win.Update()
        # win.Write()
        # self.outfile.Write()
        # self.outfile.Close()

    def Draw2DDisplay(self, event, peaktype, tb, index, pad):

        pad.cd()
        if peaktype == 's1':
            try:
                thepeak = next(islice(event.s1s(), index, index + 1))
            except:
                return -1
        if peaktype == 's2':
            try:
                thepeak = next(islice(event.s2s(), index, index + 1))
            except:
                return -1

        hitpattern = []
        maxhit = 0.
        minhit = 1e10
        for pmt in self.pmts[tb]:
            ch = {
                "x": self.pmt_locations[pmt][0],
                "y": self.pmt_locations[pmt][1],
                "hit": thepeak.area_per_channel[pmt],
                "id": pmt
            }
            if thepeak.area_per_channel[pmt] > maxhit:
                maxhit = thepeak.area_per_channel[pmt]
            if thepeak.area_per_channel[pmt] < minhit:
                minhit = thepeak.area_per_channel[pmt]
            hitpattern.append(ch)
        if maxhit > self.hitpattern_limit_high:
            maxhit = self.hitpattern_limit_high
        if minhit < self.hitpattern_limit_low:
            minhit = self.hitpattern_limit_low

        # Shameless port from xedview
        for pmt in hitpattern:
            col = self.GetColor(pmt, maxhit, minhit)
            w = 0.035
            yoff = 0

            x1 = pmt['x']
            y1 = pmt['y']
            x1 = ((x1 + 50) / 115) + .02
            y1 = ((y1 + 50) / 115) + .02

            self.plines.append(ROOT.TEllipse(x1, y1 + yoff, w, w))
            self.plines[len(self.plines) - 1].SetFillColor(col)
            self.plines[len(self.plines) - 1].SetLineColor(1)
            self.plines[len(self.plines) - 1].SetLineWidth(1)
            self.plines[len(self.plines) - 1].Draw("f")
            self.plines[len(self.plines) - 1].Draw("")

            self.latexes.append(ROOT.TLatex())
            self.latexes[len(self.latexes) - 1].SetTextAlign(12)
            self.latexes[len(self.latexes) - 1].SetTextSize(0.035)
            if col == 1:
                self.latexes[len(self.latexes) - 1].SetTextColor(0)
            self.latexes[len(self.latexes) - 1].DrawLatex(x1 - (2 * w / 3), y1 + yoff, str(pmt['id']))
            pad.Update()
        # Save from garbage collection
        return 0

    def GetColor(self, pmt, maxhit, minhit):
        if pmt['hit'] > maxhit:
            return 2
        if pmt['hit'] < minhit:
            return 0
        color = (int)((pmt['hit'] - minhit) / (maxhit - minhit) * 50) + 51
        if color > 100:
            color = 2
        if color == 0:
            color = 51
        return color

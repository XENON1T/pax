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
from pax import plugin, units, dsputils

from pax.plugins.plotting.Plotting import epoch_to_human_time


class ROOTSumWaveformDump(plugin.OutputPlugin):
    def startup(self):
        self.output_dir = self.config['output_name']
        self.samples_to_us = self.config['sample_duration'] / units.us

        self.full_plot = False
        if 'include_hits' in self.config:
            self.full_plot = self.config['include_hits']
            
            self.write_hits_for_max_number_of_s2 = 3
            self.write_hits_for_max_number_of_s1 = 3
            if 'write_hits_for_max_number_of_s2' in self.config:
                self.write_hits_for_max_number_of_s2 = self.config['write_hits_for_max_number_of_s2']
            if 'write_hits_for_max_number_of_s1' in self.config:
                self.write_hits_for_max_number_of_s1 = self.config['write_hits_for_max_number_of_s1']

        self.draw_hit_pattern = False
        if 'draw_hit_pattern' in self.config:
            self.draw_hit_pattern = self.config['draw_hit_pattern']

            self.write_hitpattern_for_max_number_of_s2 = 3
            self.write_hitpattern_for_max_number_of_s1 = 3
            if 'write_hitpattern_for_max_number_of_s2' in self.config:
                self.write_hitpattern_for_max_number_of_s2 = self.config['write_hitpattern_for_max_number_of_s2']
            if 'write_hitpattern_for_max_number_of_s1' in self.config:
                self.write_hitpattern_for_max_number_of_s1 = self.config['write_hitpattern_for_max_number_of_s1']

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
        # set high quality predefined palettes, e.g., 'kInvertedDarkBodyRadiator=56'
        #ROOT.gStyle.SetPalette(ROOT.kInvertedDarkBodyRadiator)
        #ROOT.gStyle.SetPalette(ROOT.kDarkBodyRadiator)
        ROOT.gStyle.SetPalette(ROOT.kBird)

    def write_event(self, event):

        self.setRootGStyle()

        namestring = '%s_%s' % (event.dataset_name, event.event_number)

        outfile_dotC = os.path.join(self.output_dir, "event_" + namestring + ".C")

        # Borrow liberally from xerawdp
        self.titlestring = 'Event %s from %s Recorded at %s UTC, %09d ns' % (
            event.event_number, event.dataset_name,
            epoch_to_human_time(event.start_time / units.ns),
            (event.start_time / units.ns) % (units.s))
        win = ROOT.TCanvas("canvas", self.titlestring, 1600, 600)

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
            # ROOT.TH1D: bin[0]=underflow, bin[-1]=overflow
            for i, s in enumerate(waveform.samples):
                hlist[jj].SetBinContent(i + 1, s)
            hlist[jj].SetLineWidth(1)

            if jj == 0:
                hlist[jj].SetTitle(self.titlestring)
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
                if peaktype == 's1' and len(event.interactions) and event.interactions[0].s1 >= 0:
                    if peak is event.peaks[event.interactions[0].s1]:
                        label = "  {}[{}]*:{:.1e} pe ".format(peaktype,i,peak.area)
                elif peaktype == 's2' and len(event.interactions) and event.interactions[0].s2 >= 0:
                    if peak is event.peaks[event.interactions[0].s2]:
                        label = "  {}[{}]*:{:.1e} pe ".format(peaktype,i,peak.area)

                label = label.replace("e+0", "e")

                if len(peak.contributing_channels) < 5:
                    # if there are less than 5 PMTs contributing to a peak, list them by name:
                    label += "{{{}}}".format(", ".join(["PMT{}".format(pmt) for pmt in map(str, peak.contributing_channels)]))
                else:
                    # if there are 5 or more PMTs that contribute to a peak, print their total number only:
                    label += "({})".format(len(peak.contributing_channels))
                tlatex[peaktype].DrawLatex(peak.index_of_maximum * self.samples_to_us, peak.height, label)

        # Draw Legends and update canvas
        leg.Draw()
        leg2.Draw()
        win.Update()


        #self.full_plot = True
        if not self.full_plot:

            # Dump canvas to ROOT macro (.C file)
            win.SaveAs(outfile_dotC)
            # Compress ROOT macro (.C.gz) and remove original file:
            if 'compress_output' in self.config and self.config['compress_output']:
                with open(outfile_dotC, 'rb') as f_in, gzip.open("{}.gz".format(outfile_dotC), 'wb') as f_out:
                   shutil.copyfileobj(f_in, f_out)
                os.remove(outfile_dotC)

            if 'draw_hit_pattern' in self.config and self.config['draw_hit_pattern']:
                outfile_hp_base = os.path.join(self.output_dir, "event_{}_hp".format(namestring))
                for i, peak in enumerate(event.S2s()):
                    if i >= self.write_hitpattern_for_max_number_of_s2:
                        break
                    chitp = self.Draw2DDisplay(event=event, peaktype=peak.type, tb='top', index=i, directory_name=None)
                    chitp.SaveAs("{}_{}_{}_{}.png".format(outfile_hp_base,peak.type,i,'top'))
                for i, peak in enumerate(event.S1s()):
                    if i >= self.write_hitpattern_for_max_number_of_s1:
                        break
                    chitp = self.Draw2DDisplay(event=event, peaktype=peak.type, tb='bottom', index=i, directory_name=None)
                    chitp.SaveAs("{}_{}_{}_{}.png".format(outfile_hp_base,peak.type,i,'bottom'))



        elif self.full_plot:
            outfile_name = os.path.join(self.output_dir, "event_" + namestring + '.root')
            self.outfile = ROOT.TFile(outfile_name, 'RECREATE')
            win.Update()
            win.Write()

            self.outfile.cd()

            # Now want all pulses written out
            hist_objs = []  # suppress warnings
            pmt_pattern = []
            
            pattern_dir_name = "pmt_pattern"
            for i, peak in enumerate(event.S2s()):
                if i >= self.write_hits_for_max_number_of_s2:
                    break
                hist_objs.append(self.WriteHits(peak, event, i))
            for i, peak in enumerate(event.S1s()):
                if i >= self.write_hits_for_max_number_of_s1:
                    break
                hist_objs.append(self.WriteHits(peak, event, i))

            if 'draw_hit_pattern' in self.config and self.config['draw_hit_pattern']:
                for i, peak in enumerate(event.S2s()):
                    if i >= self.write_hitpattern_for_max_number_of_s2:
                        break
                    pmt_pattern.append(self.Draw2DDisplay(event=event, peaktype=peak.type, tb='top', index=i, directory_name=pattern_dir_name))
                for i, peak in enumerate(event.S1s()):
                    if i >= self.write_hitpattern_for_max_number_of_s1:
                        break
                    pmt_pattern.append(self.Draw2DDisplay(event=event, peaktype=peak.type, tb='bottom', index=i, directory_name=pattern_dir_name))

            self.outfile.Close()
            print ("Info: {} has been generated".format(outfile_name))

    def WriteHits(self, peak, event, index):
        ''' Write out 1D histos for this peak '''
        # hit_indices = {}
        directory = self.outfile.mkdir(peak.type+"_"+str(index))
        hists = []

        pulses_to_write = {}
        hitlist = {}

        for hit in peak.hits:
            hitdict = {
                "found_in_pulse": hit[3],
                "area": hit[0],
                "channel": hit[2],
                "center": hit[1]/10,
                "left": hit[7]-1,
                "right": hit[10],
                "max_index": hit[5],
                "height": hit[4]/dsputils.adc_to_pe(self.config, hit[2],
                                                    use_reference_gain=True)
            }
            if hitdict['channel'] not in hitlist:
                hitlist[hitdict['channel']] = []
                pulses_to_write[hitdict['channel']] = []
            hitlist[hitdict['channel']].append(hitdict)
            if hitdict['found_in_pulse'] not in pulses_to_write[hitdict['channel']]:
                pulses_to_write[hitdict['channel']].append(hitdict['found_in_pulse'])

        # Now we should have pulses_to_write with which
        # pulses to plot per channel and
        # hitlist with a list of all hit properties to add
        for channel, pulselist in pulses_to_write.items():
            leftbound = -1
            rightbound = -1

            # Needs an initial scan to find histogram range
            for pulseid in pulselist:
                if leftbound == -1 or event.pulses[pulseid].left < leftbound:
                    leftbound = event.pulses[pulseid].left
                if rightbound == -1 or event.pulses[pulseid].right > rightbound:
                    rightbound = event.pulses[pulseid].right

            # Make and book the histo. Put into hists so doesn't get overwritten
            histname = "%s_%i_channel_%i" % (peak.type, index, channel)
            histtitle = "Channel %i in %s[%i]" % (channel, peak.type, index)
            c = ROOT.TCanvas(histname, "",1050,450)
            h = ROOT.TH1F(histname, histtitle, int(rightbound-leftbound),
                          self.samples_to_us * float(leftbound), self.samples_to_us *  float(rightbound))

            # Now put the bin values in the histogram
            for pulseid in pulselist:
                pulse = event.pulses[pulseid]
                w = (self.config['digitizer_reference_baseline'] + pulse.baseline -
                     pulse.raw_data.astype(np.float64))
                # ROOT.TH1D: bin[0]=underflow, bin[-1]=overflow
                for i, sample in enumerate(w):
                    h.SetBinContent(int(i+1+pulse.left-leftbound), sample)

            h.SetStats(0)
            #h.GetXaxis().SetTitle("Time [samples]")
            h.GetXaxis().SetTitle("Time [#mus]")
            h.GetYaxis().SetTitleOffset(0.5)
            h.GetYaxis().SetTitleSize(0.05)
            h.GetXaxis().SetTitleOffset(0.8)
            h.GetXaxis().SetTitleSize(0.05)
            h.GetYaxis().SetTitle("ADC Reading (baseline corrected)")

            c.cd()
            h.Draw()

            # Indicate central position of peak the hits belong to:
            c.Update()
            umin = c.GetUymin()
            umax = c.GetUymax()
            hlpeak = ROOT.TLine(self.samples_to_us * peak.index_of_maximum, umin, self.samples_to_us * peak.index_of_maximum, umax)
            hlpeak.SetLineStyle(3)
            hlpeak.SetLineWidth(2)
            hlpeak.SetLineColor(ROOT.kGray+2)
            hlpeak.Draw("same") 
            hllab = ROOT.TText(self.samples_to_us * peak.index_of_maximum, umin+0.99*(umax-umin) ,"{}[{}]".format(peak.type, index))
            hllab.SetTextAlign(31)
            hllab.SetTextColor(ROOT.kGray+2)
            hllab.SetTextFont(43)
            hllab.SetTextSize(24)
            hllab.SetTextAngle(90)
            hllab.Draw("same")



            plist = {"x": [], "y": []}
            for i, hitdict in enumerate(hitlist[channel]):

                baseline = ROOT.TLine(self.samples_to_us * hitdict['left'], pulse.baseline,
                                      self.samples_to_us * hitdict['right'], pulse.baseline)
                leftline = ROOT.TLine(self.samples_to_us * hitdict['left'], 0, self.samples_to_us * hitdict['left'], hitdict['height'])
                rightline = ROOT.TLine(self.samples_to_us * hitdict['right'], 0, self.samples_to_us * hitdict['right'], hitdict['height'])

                plist['x'].append(self.samples_to_us * hitdict['center'])
                plist['y'].append(hitdict['height'])
                leftline.SetLineStyle(2)
                rightline.SetLineStyle(2)
                leftline.SetLineColor(2)
                rightline.SetLineColor(2)
                baseline.SetLineStyle(2)
                baseline.SetLineColor(4)
                baseline.Draw("same")
                leftline.Draw("same")
                rightline.Draw("same")

                label = " hit[{}] ({:.2f} pe)".format(i, hitdict['area'])
                text = ROOT.TText(self.samples_to_us * hitdict['center'], hitdict['height'], label)
                text.SetTextAlign(12)
                text.SetTextColor(2)
                text.SetTextFont(43)
                text.SetTextSize(24)
                text.Draw("same")
                c.Update()
                hists.append({"lline": leftline, "text": text,
                              "rline": rightline, "bline": baseline})
            c.cd()
            polymarker = ROOT.TPolyMarker()
            polymarker.SetMarkerStyle(23)
            polymarker.SetMarkerColor(2)
            polymarker.SetMarkerSize(1.7)
            polymarker.SetPolyMarker(len(plist['x']), np.array(plist['x']),
                                     np.array(plist['y']))
            polymarker.Draw("same")
            c.Update()
            hists.append({"poly": polymarker, "hist": h, "c": c})

            directory.cd()
            c.Write()
        return hists


    def Draw2DDisplay(self, event, peaktype, tb, index, directory_name):

        if directory_name:
            if not directory_name in self.outfile.GetListOfKeys():
                directory = self.outfile.mkdir(directory_name)
            else:
                directory = self.outfile.Get(directory_name)
            directory.cd()

        c = ROOT.TCanvas("pmt_pattern_{}_{}".format(peaktype,index),'canvas with {}[{}] pmt hitpattern'.format(peaktype,index),800,800)
        c.SetLeftMargin(0.09) #0.05
        c.SetRightMargin(0.15)
        c.SetTopMargin(0.12)
        c.SetBottomMargin(0.12)
        if peaktype == 's1':
            try:
                thepeak = next(islice(event.S1s(), index, index + 1))
            except:
                return -1
        if peaktype == 's2':
            try:
                thepeak = next(islice(event.S2s(), index, index + 1))
            except:
                return -1

        hitpattern = []
        for pmt in self.pmts[tb]:
            ch = {
                "x": self.pmt_locations[pmt][0],
                "y": self.pmt_locations[pmt][1],
                "hit": thepeak.area_per_channel[pmt],
                "id": pmt
            }
            hitpattern.append(ch)

        self.plines.append(ROOT.TH2Poly("hitpattern_{}_{}_{}".format(peaktype,index,tb),"hitpattern_{}_{}_{}".format(peaktype,index,tb),-52.,+52.,-52.,52.))
        self.plines[-1].SetTitle("Event {} from {}; x [cm]; y [cm]".format(event.event_number, event.dataset_name))
        w = 3.81 #i.e., radius of our 3inch PMTs in cm
        # add PMT polygons to TH2Poly and set content of bin
        for pmt in hitpattern:

            x1 = pmt['x']
            y1 = pmt['y']

            xpol, ypol = self.drawPMT(x1,y1,r=w)
            self.plines[-1].AddBin(len(xpol),xpol,ypol)
            self.plines[-1].Fill(x1,y1,pmt['hit'])

        self.plines[-1].GetYaxis().SetTitleOffset(1.20)
        self.plines[-1].Draw("colz")
        
        # print PMT id on top of TH2Poly
        for pmt in hitpattern:
            self.latexes.append(ROOT.TLatex())
            self.latexes[-1].SetTextFont(132)
            self.latexes[-1].SetTextAlign(22)
            self.latexes[-1].SetTextSize(0.030)

            x1 = pmt['x']
            y1 = pmt['y']
            #self.latexes[-1].DrawLatex(x1 - (2 * w / 3), y1, str(pmt['id']))
            self.latexes[-1].DrawLatex(x1, y1, str(pmt['id']))

        self.latexes[-1].SetTextAlign(13)
        self.latexes[-1].SetTextFont(132)
        self.latexes[-1].SetTextSize(0.038)
        self.latexes[-1].DrawLatex(-50, 50, "{}[{}] {}".format(peaktype.upper(),index,tb) )

        c.Update()
        # return canvas to save to static png file
        if not directory_name:
            return c

        c.Write()
        # Save from garbage collection
        return 0

    def polygon(self, x0, y0, r, N):
        n = np.arange(N)
        x = r * np.cos(2*np.pi*n/N) + x0
        y = r * np.sin(2*np.pi*n/N) + y0
        return (x,y)

    def drawPMT(self, x, y, r=1):
        return self.polygon(x0=x,y0=y,r=r,N=20)



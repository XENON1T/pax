"""
Event display using ROOT framework
"""
import ROOT  # sigh
import os
import numpy as np
import pytz
import datetime

from itertools import islice
from pax import plugin, units, dsputils


def epoch_to_human_time(timestamp):
        # Unfortunately the python datetime, explicitly choose UTC timezone (default is local)
        tz = pytz.timezone('UTC')
        return datetime.datetime.fromtimestamp(timestamp / units.s, tz=tz).strftime("%Y/%m/%d, %H:%M:%S")


class ROOTWaveformDisplay(plugin.OutputPlugin):

    def startup(self):
        self.output_dir = self.config['output_name']
        self.full_plot = False
#        print("ROOTPLOT")
#        print(self.config['rootplot_full'])
        if self.config['rootplot_full']:
            self.full_plot = True
        self.samples_to_us = self.config['sample_duration'] / units.us

        if self.output_dir is None:
            raise RuntimeError("You must supply an output directory for ROOT display")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        ROOT.gROOT.SetBatch(True)
        self.peak_colors = {"lone_hit": 40, "s1": 4, "s2": 2, "unknown": 29, "noise": 30}

        self.pmt_locations = np.array([[self.config['pmts'][ch]['position']['x'],
                                        self.config['pmts'][ch]['position']['y']]
                                       for ch in range(self.config['n_channels'])])
        self.hitpattern_limit_low = 1e-1
        self.hitpattern_limit_high = 1e4
        self.pmts = {array: self.config['channels_%s' % array] for array in ('top', 'bottom')}

        # ROOT does some weird stuff with memory. Don't ask. Don't think. Just obey.
        self.plines = []
        self.latexes = []

    def write_event(self, event):

        # Super simple. Just make a sum waveform and dump it
        namestring = '%s_%s' % (event.dataset_name, event.event_number)
        outfile = os.path.join(self.output_dir, "event_" + namestring + ".root")
        self.outfile = ROOT.TFile(outfile, 'RECREATE')

        # Borrow liberally from xerawdp
        titlestring = 'Event %s from %s Recorded at %s UTC, %09d ns' % (
            event.event_number, event.dataset_name,
            epoch_to_human_time(event.start_time / units.ns),
            (event.start_time/units.ns) % (units.s))
        win = ROOT.TCanvas("canvas", titlestring, 1200, 1000)
        # win.Divide(1,2);

        sum_pad = ROOT.TPad("sum_waveform_pad", "sum_waveform_pad", 0.0, 0.4, 1.0, 1.0, 0)
        detail_pad_0 = ROOT.TPad("detail_0_pad", "detail_0_pad", 0.0, 0.0, 0.25, 0.4, 0)
        detail_pad_1 = ROOT.TPad("detail_1_pad", "detail_1_pad", 0.25, 0.0, 0.5, 0.4, 0)
        detail_pad_2 = ROOT.TPad("detail_2_pad", "detail_2_pad", 0.5, 0.0, 0.75, 0.4, 0)
        detail_pad_3 = ROOT.TPad("detail_3_pad", "detail_3_pad", 0.75, 0.0, 1.0, 0.4, 0)
        # hitplot_pad = ROOT.TPad("hit_plot_pad", "hit_plot_pad", 0.0, 0.0, 1.0, 0.3, 0)

        sum_pad.SetRightMargin(0.01)
        sum_pad.SetLeftMargin(0.05)

        sum_pad.Draw()
        detail_pad_0.Draw()
        detail_pad_1.Draw()
        detail_pad_2.Draw()
        detail_pad_3.Draw()
        # hitplot_pad.Draw()

        # Plot Sum Waveform
        sum_pad.cd()
        sum_waveform_x = np.arange(0, event.length()-1,
                                   dtype=np.float32)

        for w in self.config['waveforms_to_plot']:
            waveform = event.get_sum_waveform(w['internal_name'])
            h = ROOT.TH1D("g", titlestring, len(sum_waveform_x), 0,
                          self.samples_to_us*(len(sum_waveform_x)-1))
            for i, s in enumerate(waveform.samples):
                h.SetBinContent(i, s)
            h.SetStats(0)
            h.GetXaxis().SetTitle("Time [#mus]")
            h.GetYaxis().SetTitleOffset(0.3)
            h.GetYaxis().SetTitleSize(0.05)
            h.GetXaxis().SetTitleOffset(0.8)
            h.GetXaxis().SetTitleSize(0.05)
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
                boxes[len(boxes)-1].Draw("same")

        # Labels, want them sorted
        for i, peak in enumerate(event.s2s()):
            if i > 1:
                break
            label = "s2["+str(s2)+"]: " + "{:.2f}".format(peak.area)
            if len(peak.contributing_channels) < 5:
                label += str(peak.contributing_channels)
            else:
                label += "["+str(len(peak.contributing_channels))+" channels]"
            s2 += 1
            text.append(ROOT.TText(peak.left*self.samples_to_us, peak.height, label))
            text[len(text)-1].SetTextColor(self.peak_colors[peak.type])
            text[len(text)-1].SetTextSize(0.03)
            text[len(text)-1].Draw("same")
        for i, peak in enumerate(event.s1s()):
            if i > 1:
                break
            label = "s1["+str(s1)+"]: " + "{:.2f}".format(peak.area)
            if len(peak.contributing_channels) < 5:
                label += str(peak.contributing_channels)
            else:
                label += "[" + str(len(peak.contributing_channels)) + " channels]"
            s1 += 1
            text.append(ROOT.TText(peak.left*self.samples_to_us, peak.height, label))
            text[len(text)-1].SetTextColor(self.peak_colors[peak.type])
            text[len(text)-1].SetTextSize(0.02)
            text[len(text)-1].Draw("same")

        # Polymarkers don't like being overwritten I guess, so make a container
        pms = {}
        for peaktype, plist in peaks.items():
            pms[peaktype] = ROOT.TPolyMarker()
            pms[peaktype].SetMarkerStyle(23)
            if peaktype in self.peak_colors:
                pms[peaktype].SetMarkerColor(self.peak_colors[peaktype])
            else:
                pms[peaktype].SetMarkerColor(0)
            pms[peaktype].SetMarkerSize(1.1)
            pms[peaktype].SetPolyMarker(len(plist['x']), np.array(plist['x']), np.array(plist['y']))
            pms[peaktype].Draw("same")

        # Make the 2D displays
        latex = ROOT.TLatex()
        latex.SetTextAlign(12)
        latex.SetTextSize(.06)
        latex.SetTextColor(1)
        detail_pad_0.cd()
        latex.DrawLatex(0.3, 0.97, "s1[0] top")
        detail_pad_1.cd()
        latex.DrawLatex(0.3, 0.97, "s1[0] bottom")
        detail_pad_2.cd()
        latex.DrawLatex(0.3, 0.97, "s2[0] top")
        detail_pad_3.cd()
        latex.DrawLatex(0.3, 0.97, "s2[0] bottom")

        self.Draw2DDisplay(event, 's1', 'top', 0, detail_pad_0)
        self.Draw2DDisplay(event, 's1', 'bottom', 0, detail_pad_1)
        self.Draw2DDisplay(event, 's2', 'top', 0, detail_pad_2)
        self.Draw2DDisplay(event, 's2', 'bottom', 0, detail_pad_3)

        win.Update()
        self.outfile.cd()
        win.Write()

        if self.full_plot:
            # Now want all pulses written out
            # peak_indices = {}
            hist_objs = []  # suppress warnings
            for i, peak in enumerate(event.S2s()):
                if i > 2:
                    break
                hist_objs.append(self.WriteHits(peak, event, i))
            for i, peak in enumerate(event.S1s()):
                if i > 2:
                    break
                hist_objs.append(self.WriteHits(peak, event, i))
            '''
            for peak in event.peaks:
                if peak.type == 's1' or peak.type == 's2':
                    continue
                if peak.type not in peak_indices:
                    peak_indices[peak.type] = -1
                peak_indices[peak.type] += 1
                hist_objs.append(self.WriteHits(peak, event,
                                                peak_indices[peak.type]))
            '''
        self.outfile.Close()

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
            leftbound = peak.left
            rightbound = peak.right

            # Needs an initial scan to find histogram range
            '''
            for pulseid in pulselist:
                if leftbound == -1 or event.pulses[pulseid].left < leftbound:
                    leftbound = event.pulses[pulseid].left
                if rightbound == -1 or event.pulses[pulseid].right > rightbound:
                    rightbound = event.pulses[pulseid].right
            '''
            # Make and book the histo. Put into hists so doesn't get overwritten
            histname = "%s_%i_channel_%i" % (peak.type, index, channel)
            histtitle = "Channel %i in %s[%i]" % (channel, peak.type, index)
            c = ROOT.TCanvas(histname, "")
            h = ROOT.TH1F(histname, histtitle, int(rightbound-leftbound),
                          float(leftbound), float(rightbound))

            # Now put the bin values in the histogram
            wf_ch = np.zeros(rightbound - leftbound)
            for pulseid in pulselist:
                pulse = event.pulses[pulseid]
                w = (self.config['digitizer_reference_baseline'] + pulse.baseline -
                     pulse.raw_data.astype(np.float64))
                '''
                for i, sample in enumerate(w):
                    h.SetBinContent(int(i+pulse.left-leftbound), sample)
                '''
                for sample in range(leftbound, rightbound):
                    if(pulse.left <= sample and pulse.right >= sample):
                        wf_ch[sample - leftbound] = w[sample - pulse.left]
            for j in range(int(rightbound - leftbound)):
                h.SetBinContent(j, wf_ch[j])

            h.SetStats(0)
            h.GetXaxis().SetTitle("Time [samples]")
            h.GetYaxis().SetTitleOffset(0.8)
            h.GetYaxis().SetTitleSize(0.05)
            h.GetXaxis().SetTitleOffset(0.8)
            h.GetXaxis().SetTitleSize(0.05)
            h.GetYaxis().SetTitle("ADC Reading (baseline corrected)")

            c.cd()
            h.Draw()

            plist = {"x": [], "y": []}
            for i, hitdict in enumerate(hitlist[channel]):

                baseline = ROOT.TLine(hitdict['left'], pulse.baseline,
                                      hitdict['right'], pulse.baseline)
                leftline = ROOT.TLine(hitdict['left'], 0, hitdict['left'], hitdict['height'])
                rightline = ROOT.TLine(hitdict['right'], 0, hitdict['right'], hitdict['height'])

                plist['x'].append(hitdict['center'])
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

                label = "hit " + str(i) + "({:.2f} p.e.)".format(hitdict['area'])
                text = ROOT.TText(hitdict['center'], hitdict['height'], label)
                text.SetTextColor(2)
                text.SetTextSize(0.03)
                text.Draw("same")
                c.Update()
                hists.append({"lline": leftline, "text": text,
                              "rline": rightline, "bline": baseline})
            c.cd()
            polymarker = ROOT.TPolyMarker()
            polymarker.SetMarkerStyle(23)
            polymarker.SetMarkerColor(2)
            polymarker.SetMarkerSize(1.1)
            polymarker.SetPolyMarker(len(plist['x']), np.array(plist['x']),
                                     np.array(plist['y']))
            polymarker.Draw("same")
            c.Update()
            hists.append({"poly": polymarker, "hist": h, "c": c})

            directory.cd()
            c.Write()
        return hists

    def Draw2DDisplay(self, event, peaktype, tb, index, pad):

        pad.cd()
        if peaktype == 's1':
            try:
                thepeak = next(islice(event.s1s(), index, index+1))
            except:
                return -1
        if peaktype == 's2':
            try:
                thepeak = next(islice(event.s2s(), index, index+1))
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
            x1 = ((x1 + 50) / 115)+.02
            y1 = ((y1 + 50) / 115)+.02

            # xx = [x1-w, x1+w, x1+w, x1-w, x1-w]
            # yy = [y1+w+yoff, y1+w+yoff, y1-w+yoff, y1-w+yoff, y1+w+yoff]

            self.plines.append(ROOT.TEllipse(x1, y1+yoff, w, w))
            self.plines[len(self.plines)-1].SetFillColor(col)
            self.plines[len(self.plines)-1].SetLineColor(1)
            self.plines[len(self.plines)-1].SetLineWidth(1)
            self.plines[len(self.plines)-1].Draw("f")
            self.plines[len(self.plines)-1].Draw("")

            self.latexes.append(ROOT.TLatex())
            self.latexes[len(self.latexes)-1].SetTextAlign(12)
            self.latexes[len(self.latexes)-1].SetTextSize(0.035)
            if col == 1:
                self.latexes[len(self.latexes)-1].SetTextColor(0)
            self.latexes[len(self.latexes)-1].DrawLatex(x1-(2*w/3), y1+yoff, str(pmt['id']))
            pad.Update()
        # Save from garbage collection
        return 0

    def GetColor(self, pmt, maxhit, minhit):
        if pmt['hit'] > maxhit:
            return 2
        if pmt['hit'] < minhit:
            return 0
        color = (int)((pmt['hit']-minhit)/(maxhit-minhit)*50)+51
        if color > 100:
            color = 2
        if color == 0:
            color = 51
        return color

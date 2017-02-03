"""
Event display using ROOT framework
"""
import ROOT #sigh
import os
import numpy as np
from pax import plugin


class ROOTWaveformDisplay(plugin.OutputPlugin):
    
    def startup(self):
        self.output_dir = self.config['output_name']
        if self.output_dir is None:
            raise RuntimeError("You must supply an output directory for ROOT display")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        
    def write_event(self, event):
        
        # Super simple. Just make a sum waveform and dump it
        name = "test.root"
        outfile = os.path.join(self.output_dir, name)
        self.outfile = ROOT.TFile(outfile, 'RECREATE')
        c = ROOT.TCanvas("display", "display")
        c.cd()
        # Get the data
        sum_waveform_x = np.arange(0, event.length()-1, 
                                   dtype=np.float32)
        graphs = []
        mg = ROOT.TMultiGraph()
        for w in self.config['waveforms_to_plot']:
            waveform = event.get_sum_waveform(w['internal_name'])
            g = ROOT.TGraph(len(sum_waveform_x), sum_waveform_x,
                            waveform.samples[0:event.length()])
            mg.Add(g)
        mg.Draw("AL")
        c.Update()
        self.outfile.cd()
        c.Write()
        #self.outfile.Write()
        self.outfile.Close()


            

from pax import plugin as p

import ROOT

class WriteToROOTFile(p.OutputPlugin):

    def __init__(self, config):
        p.OutputPlugin.__init__(self, config)

        self.log.debug("Writing ROOT file to %s" % config['rootfile'])
        self.tfile = ROOT.TFile(config['rootfile'], "Recreate")
        self.t1 = ROOT.TTree("T1", "T1")
        self.S2s = ROOT.vector('float')()
        self.S2s_x_simple = ROOT.vector('float')()
        self.S2s_y_simple = ROOT.vector('float')()
        self.t1.Branch('S2s', self.S2s)
        self.t1.Branch('S2s_x_simple', self.S2s_x_simple)
        self.t1.Branch('S2s_y_simple', self.S2s_y_simple)

    def __del__(self):
        self.log.debug("Closing %s" % self.config['rootfile'])
        self.tfile.Write()
        self.tfile.Close()

    def write_event(self, event):
        self.log.debug('Writing event to ROOT file')
        self.S2s.clear()
        self.S2s_x_simple.clear()
        self.S2s_y_simple.clear()
        for i in range(0,len(event['peaks'])):#p in event['peaks']:
            p = event['peaks'][i]
            if p['peak_type'] == 'large_s2' or p['peak_type'] == 'small_s2':
                self.S2s.push_back(p['top_and_bottom']['area'])
                self.S2s_x_simple.push_back(p['rec']['PosSimple']['x'])
                self.S2s_y_simple.push_back(p['rec']['PosSimple']['y'])
                self.t1.Fill()

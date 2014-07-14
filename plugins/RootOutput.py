from pax import plugin as p
try:
    import ROOT
except: 
    print("ROOT not found.")
    exit()

__author__='coderre'

class WriteToROOTFile(p.OutputPlugin):
    def __init__(self,config):
        p.OutputPlugin.__init__(self,config)
        
        self.log.debug("Writing ROOT file to %s" % config['rootfile'])
        self.tfile = ROOT.TFile(config['rootfile'],"Recreate")
        self.t1 = ROOT.TTree("T1","T1")
        self.S2s = ROOT.vector('float')()
        self.t1.Branch('S2s', self.S2s)

    def __del__(self):
        self.log.debug("Closing %s" % self.config['rootfile'])
        self.tfile.Write()
        self.tfile.Close()

    def WriteEvent(self,event):
        self.log.debug('Writing event to ROOT file')
        self.S2s.clear()
#        print(event['peaks']['summed'])
        for p in event['peaks']['summed']:
            self.S2s.push_back(p['summed']['area'])
            self.t1.Fill()
            
#for s2 in event['peaks']['summed']:
        #    self.S2s.push_back(s2['area'])
        #self.t1.Fill()

import numpy as np
import ROOT

from pax import plugin

overall_header = """
#include "TObject.h"
#include "TClonesArray.h"
#include "TRefArray.h"
#include "TRef.h"
#include "TH1.h"
#include "TBits.h"
#include "TMath.h"

# Are these actually necessary?
#include <vector>
#include <string>
#include <iostream>
"""


class_template = """
class {class_name} : public TObject {

private:
   {data_attributes}

public:
   ClassDef(Event,1)  //Event structure
};
"""



class ROOTClass(plugin.OutputPlugin):

    def startup(self):
        # TODO: dataset_name requires long string fields, Peak.type and Peak.detector would rather have short ones
        # Maybe the latter two should become enums?
        self.config.setdefault('string_field_length', 32)
        self.config.setdefault('fields_to_ignore', ('all_hits', 'raw_data', 'sum_waveforms',
                                                    # Temporarily ignore the long fields, see below...
                                                    'start_time', 'stop_time',
                                                    # Temporarily ignore the peak reference fields
                                                    's1', 's2',
                                                    ))

        # This line makes sure all TTree objects are NOT owned
        # by python, avoiding segfaults when garbage collecting
        ROOT.TTree.__init__._creates = False

        # TODO: add overwrite check
        self.f = ROOT.TFile(self.config['output_name'], "RECREATE")


    def write_event(self, event):
        self.t = ROOT.TTree('T', 'event tree')
        self.log.debug("Creating event tree")
        self.branch_buffers = {}
        for name, value in event.get_fields_data():

            if type(value) == int:
                self.branch_buffers[name] = np.zeros(1, np.int32)
                self.t.Branch(name,
                              self.branch_buffers[name],
                                                 name + '/I')
            elif type(value) == float:
                self.branch_buffers[name] = np.zeros(1, np.float32)
                self.t.Branch(name,
                              self.branch_buffers[name],
                                                 name + '/F')
            if name == 'peaks':
                for name2, value2 in value[0].get_fields_data():
                    if isinstance(value2, float):
                        peak_string += "float " + name2 + ";\n"
                    elif isinstance(value2, int):
                        peak_string += "int " + name2 + ";\n"
                    elif isinstance(value2, str):
                        peak_string += "string " + name2 + ";\n"


        self.log.debug(("C++ class",
                        self.cpp_string % peak_string))

        if self.output_name.endswith('.root'):
            f_class = open(self.output_name[:-5] + '.C', 'w')
            f_class.write(self.cpp_string % peak_string)
            f_class.close()
        ROOT.gROOT.ProcessLine(self.cpp_string % peak_string)
        self.peaks = ROOT.vector('Peak')()
        self.t.Branch('peaks', self.peaks)

    def write_event(self, event):
        if self.t == None:
            self.setup_tree(event)

        # self.peaks.clear()

        for name, value in event.get_fields_data():
            if isinstance(value, (int, float)):
                self.branch_buffers[name][0] = value

            if name == 'peaks':
                for peak_pax in value:
                    peak_root = ROOT.Peak()
                    for name2, value2 in peak_pax.get_fields_data():
                        if isinstance(value2, (int, float, str)):
                            setattr(peak_root, name2, value2)

                    self.peaks.push_back(peak_root)

        self.t.Fill()
        self.log.debug('Writing event')

    def shutdown(self):
        self.t.Write()
        self.f.Close()
        pass

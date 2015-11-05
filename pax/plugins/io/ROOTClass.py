import os
import re
import numpy as np
import ROOT

import pax      # For version number
from pax import plugin, datastructure

import rootpy.stl as stl
from rootpy.userdata import BINARY_PATH
# import rootpy.compiled as C

overall_header = """
#include "TFile.h"
#include "TTree.h"
#include "TObject.h"
#include "TString.h"
#include <vector>
"""

class_template = """
{child_classes_code}

{ifndefs}
class {class_name} : public TObject {{

public:
{data_attributes}
    ClassDef({class_name}, {class_version});
}};

#endif

"""


def load_event_class(filename):
    """Read a C++ root class definition, generating dictionaries for vectors of classes"""
    # Find the classes defined in the file
    classnames = []
    with open(filename, 'r') as classfile:
        for line in classfile.readlines():
            m = re.match(r'class (\w*) ', line)
            if m:
                classnames.append(m.group(1))

    # Load the file in ROOT
    ROOT.gROOT.ProcessLine('.L %s+' % filename)
    # C.register_file(filename,classnames)

    # Build the required dictionaries for the vectors of classes
    for name in classnames:
        stl.generate("vector<%s>" % name, "%s;<vector>" % filename, True)
        # ROOT.gInterpreter.GenerateDictionary("vector<%s>" % name, filename)


class WriteROOTClass(plugin.OutputPlugin):

    def startup(self):
        # TODO: dataset_name requires long string fields, Peak.type and Peak.detector would rather have short ones
        # Maybe the latter two should become enums?
        self.config.setdefault('buffer_size', 16000)
        self.config.setdefault('fields_to_ignore', ('all_hits', 'raw_data', 'sum_waveforms',
                                                    'hits', 'pulses'))
        self._custom_types = []

        # TODO: add overwrite check
        # cleanup()
        self.f = ROOT.TFile(self.config['output_name'] + '.root', "RECREATE")
        self.event_tree = None

    def write_event(self, event):

        if not self.event_tree:
            if self.config['exclude_compilation_from_timer']:
                self.processor.timer.punch()

            # Construct the event tree
            self.event_tree = ROOT.TTree(self.config['tree_name'],
                                         'Tree with %s events from pax' % self.config['tpc_name'])

            # Write the event class C++ definition
            # Do this here, since it requires an instance (for length of arrays)

            full_name = os.path.join(BINARY_PATH, 'modules', 'pax_event_class.cpp')

            # TODO: This fails if the first event doesn't have a peak!!
            with open(full_name, mode='w') as outfile:
                outfile.write(overall_header)
                outfile.write(self._build_model_class(event))
            load_event_class(full_name)
            self.log.debug("Event class loaded, creating event")
            self.root_event = ROOT.Event()
            # TODO: setting the splitlevel to 0 or 99 seems to have no effect??
            self.event_tree.Branch('events', 'Event', self.root_event, self.config['buffer_size'], 99)

            if self.config['exclude_compilation_from_timer']:
                self.processor.timer.punch()

        # Last collection of each data model type seen
        self.last_collection = {}
        self.set_root_object_attrs(event, self.root_event)
        self.event_tree.Fill()

    def set_root_object_attrs(self, python_object, root_object):
        """Set attribute values of the root object based on data_model instance python object
        Returns nothing: modifies root_objetc in place
        """
        obj_name = python_object.__class__.__name__
        fields_to_ignore = self.config['fields_to_ignore']
        list_field_info = python_object.get_list_field_info()

        for field_name, field_value in python_object.get_fields_data():
            if field_name in fields_to_ignore:
                continue

            elif isinstance(field_value, list):
                # Collection field -- recursively initialize collection elements
                element_name = list_field_info[field_name].__name__
                list_of_elements = getattr(python_object, field_name)

                root_vector = getattr(root_object, field_name)
                root_vector.clear()
                for element_python_object in list_of_elements:
                    element_root_object = getattr(ROOT, element_name)()
                    self.set_root_object_attrs(element_python_object, element_root_object)
                    root_vector.push_back(element_root_object)
                self.last_collection[element_name] = list_of_elements

            elif isinstance(field_value, np.ndarray):
                # Unfortunately we can't store numpy arrays directly into ROOT's ROOT.PyXXXBuffer.
                # Doing so will not give an error, but the data will be mangled!
                # Instead we'd have to use python's old array module...
                # For now we'll simply store the elements one-by-one
                # This isn't very efficient, but it seems the speed is still good.
                root_field = getattr(root_object, field_name)
                for i, x in enumerate(field_value):
                    root_field[i] = x

            else:
                # Everything else apparently just works magically:
                setattr(root_object, field_name, field_value)

        # Add values to user-defined fields
        for field_name, field_type, field_code in self.config['extra_fields'].get(obj_name, []):
            field = getattr(root_object, field_name)
            exec(field_code, dict(root_object=root_object, python_object=python_object, field=field, self=self))

    def _get_index(self, py_object):
        """Return index of py_object in last collection of models of corresponding type seen in event"""
        return self.last_collection[py_object.__class__.__name__].index(py_object)

    def write_to_disk(self):
        self.event_tree.Write()

    def shutdown(self):
        self.write_to_disk()
        self.f.Close()

    def _build_model_class(self, model):
        """Return ROOT C++ class definition corresponding to instance of data_model.Model
        """
        model_name = model.__class__.__name__
        self.log.debug('Building ROOT class for %s' % model_name)

        type_mapping = {'float':    'Float_t',
                        'float64':  'Double_t',
                        'float32':  'Float_t',
                        'int':      'Int_t',
                        'int16':    'Short_t',
                        'int32':    'Int_t',
                        'int64':    'Long64_t',
                        'bool':     'Bool_t',
                        'bool_':    'Bool_t',
                        'long':     'Long64_t',
                        'str':      'TString'}

        list_field_info = model.get_list_field_info()
        class_attributes = ''
        child_classes_code = ''
        for field_name, field_value in sorted(model.get_fields_data()):
            if field_name in self.config['fields_to_ignore']:
                continue

            # Collections (e.g. event.peaks)
            elif field_name in list_field_info:
                element_model_name = list_field_info[field_name].__name__
                self.log.debug("List column %s encountered. Type is %s" % (field_name, element_model_name))
                if element_model_name not in self._custom_types:
                    self._custom_types.append(element_model_name)
                    if not len(field_value):
                        self.log.warning("Don't have a %s instance to use: making default one..." % element_model_name)
                        source = list_field_info[field_name]()
                    else:
                        source = field_value[0]
                    child_classes_code += '\n' + self._build_model_class(source)
                class_attributes += '\tvector <%s>  %s;\n' % (element_model_name, field_name)

            # Numpy array (assumed fixed-length, 1-d)
            elif isinstance(field_value, np.ndarray):
                class_attributes += '\t%s  %s[%d];\n' % (type_mapping[field_value.dtype.type.__name__],
                                                         field_name, len(field_value))

            # Everything else (int, float, bool)
            else:
                class_attributes += '\t%s  %s;\n' % (type_mapping[type(field_value).__name__],
                                                     field_name)

        # Add any user-defined extra fields
        for field_name, field_type, field_code in self.config['extra_fields'].get(model_name, []):
            class_attributes += '\t%s %s;\n' % (field_type, field_name)

        define = "#ifndef %s" % (model_name.upper() + "\n") + "#define %s " % (model_name.upper() + "\n")

        return class_template.format(ifndefs=define, class_name=model_name,
                                     data_attributes=class_attributes,
                                     child_classes_code=child_classes_code,
                                     class_version=pax.__version__.replace('.', ''))


class ReadROOTClass(plugin.InputPlugin):

    def startup(self):
        # Make sure to store the file as an attribute, else it will go out of scope -> garbage collected -> you die
        full_name = os.path.join(BINARY_PATH, 'modules', 'pax_event_class.cpp')
        if not os.path.exists(full_name):
            raise ValueError("You must provide pax_event_class.cpp in the current directory to read the ROOT format.\n"
                             "Looking for a nice project? Please fix this!")
        load_event_class(full_name)
        if not os.path.exists(self.config['input_name']):
            raise ValueError("Input file %s does not exist" % self.config['input_name'])
        self.f = ROOT.TFile(self.config['input_name'])
        self.t = self.f.Get(self.config['tree_name'])
        self.number_of_events = self.t.GetEntries()
        # TODO: read in event numbers, so we can select events!

    def get_events(self):
        for event_i in range(self.number_of_events):
            self.t.GetEntry(event_i)
            root_event = self.t.events
            event = datastructure.Event(n_channels=root_event.n_channels,
                                        start_time=root_event.start_time,
                                        sample_duration=root_event.sample_duration,
                                        stop_time=root_event.stop_time)
            self.set_python_object_attrs(root_event, event, self.config['fields_to_ignore'])
            yield event

    def set_python_object_attrs(self, root_object, py_object, fields_to_ignore):
        """Sets attribute values of py_object to corresponding values in ROOT_object
        Returns nothing (modifies py_object in place)
        """
        for field_name, default_value in py_object.get_fields_data():

            if field_name in fields_to_ignore:
                continue

            try:
                root_value = getattr(root_object, field_name)
            except AttributeError:
                # Value not present in root object (e.g. event.all_hits)
                self.log.debug("%s not in root object?" % field_name)
                continue

            if isinstance(default_value, list):
                child_class_name = py_object.get_list_field_info()[field_name].__name__
                result = []
                for child_i in range(len(root_value)):
                    child_py_object = getattr(datastructure, child_class_name)()
                    self.set_python_object_attrs(root_value[child_i], child_py_object, fields_to_ignore)
                    result.append(child_py_object)

            elif isinstance(default_value, np.ndarray):
                try:
                    if not len(root_value):
                        # Empty! no point to assign the value. Errors for structured array.
                        continue
                except TypeError:
                    self.log.warning("Strange error in numpy array field %s, type from ROOT object is %s, "
                                     "which is not iterable!" % (field_name, type(root_value)))
                    continue
                # Use list() for same reason as described above in WriteROOTClass:
                # Something is wrong with letting the root buffers interact with numpy arrays directly
                result = np.array(list(root_value), dtype=default_value.dtype)

            else:
                result = root_value

            setattr(py_object, field_name, result)

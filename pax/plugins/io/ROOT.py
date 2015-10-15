import os
from copy import copy
import re
import numpy as np
import ROOT

import pax      # For version number
from pax import plugin, data_model

overall_header = """
#include "TFile.h"
#include "TTree.h"
#include "TObject.h"
#include "TRefArray.h"
#include "TRef.h"

#include <vector>
#include <string>
#include <iostream>
"""

class_template = """
{child_classes_code}

class {class_name} : public TObject {{

public:
{data_attributes}
    ClassDef({class_name}, {class_version});
}};
"""


class WriteROOTClass(plugin.OutputPlugin):

    def startup(self):
        # TODO: dataset_name requires long string fields, Peak.type and Peak.detector would rather have short ones
        # Maybe the latter two should become enums?
        self.config.setdefault('string_field_length', 32)
        self.config.setdefault('buffer_size', 16000)
        self.config.setdefault('fields_to_ignore', ('all_hits', 'raw_data', 'sum_waveforms',
                                                    'hits', 'pulses'))
        self._custom_types = []

        # TODO: add overwrite check
        self.cleanup()
        self.f = ROOT.TFile(self.config['output_name'] + '.root', "RECREATE")
        self.event_tree = None

    def cleanup(self):
        """Clean any C++ crap (AutoDict, pax_event_class) in the current directory
        TODO: obviously this is a temp hack, all the stuff should go into some subdir, which we can then delete
        """
        for f in os.listdir('.'):
            if re.search(r'pax_event_class', f) or re.search(r'AutoDict', f):
                os.remove(os.path.join('.', f))

    def write_event(self, event):

        if not self.event_tree:
            if self.config['exclude_compilation_from_timer']:
                self.processor.timer.punch()

            # Construct the event tree
            # Branchref() is needed to enable convenient readout of TRef and TRefArray
            # TODO: somehow it works with or without it??
            self.event_tree = ROOT.TTree('T1', 'Tree with %s events from pax' % self.config['tpc_name'])
            self.event_tree.BranchRef()

            # Write the event class C++ definition
            # Do this here, since it requires an instance (for length of arrays)
            # TODO: This fails if the first event doesn't have a peak!!
            with open('pax_event_class.cpp', mode='w') as outfile:
                outfile.write(overall_header)
                outfile.write(self._build_model_class(event))
            ROOT.gROOT.ProcessLine('.L pax_event_class.cpp+')

            # Build dictionaries for the custom vector types
            for vtype in self._custom_types + ['TRef']:
                self.log.debug("Generating dictionary for %s" % vtype)
                ROOT.gInterpreter.GenerateDictionary("vector<%s*>" % vtype, "pax_event_class.cpp")
            self.log.debug("Event class loaded, creating event")
            self.root_event = ROOT.Event()
            # TODO: setting the splitlevel to 0 or 99 seems to have no effect??
            self.event_tree.Branch('events', 'Event', self.root_event, self.config['buffer_size'], 99)

            if self.config['exclude_compilation_from_timer']:
                self.processor.timer.punch()

        # Last collection of each data model type seen
        self.last_collection = {}
        self.set_values(event, self.root_event)
        self.event_tree.Fill()

    def set_values(self, python_object, root_object):
        """Set attribute values of the root object based on data_model instance python object"""
        obj_name = python_object.__class__.__name__
        fields_to_ignore = copy(self.config['fields_to_ignore'])

        # Handle collections first (so references will work afterwards)
        # TODO: add sort method to make sure event.peaks is done before peak.interactions
        # sorting with reverse=True is just a dirty hack ;-)
        list_field_info = python_object.get_list_field_info()
        list_field_names = sorted(list(list_field_info.keys()), reverse=True)
        for field_name in list_field_names:
            if field_name in fields_to_ignore:
                continue
            element_name = list_field_info[field_name].__name__
            list_of_elements = getattr(python_object, field_name)

            # Recursively initialize collection elements
            root_vector = getattr(root_object, field_name)
            root_vector.clear()
            for element_python_object in list_of_elements:
                element_root_object = getattr(ROOT, element_name)()
                self.set_values(element_python_object, element_root_object)
                root_vector.push_back(element_root_object)
            self.last_collection[element_name] = (list_of_elements, root_vector)

            fields_to_ignore.append(field_name)

        for field_name, field_value in python_object.get_fields_data():
            if field_name in fields_to_ignore:
                continue

            # References (e.g. interaction.s1)
            if isinstance(field_value, data_model.Model):
                tref = getattr(root_object, field_name)
                tref.__assign__(self._get_root_obj(field_value))

            # Everything else (float, int, bool, string, numpy array)
            else:
                setattr(root_object, field_name, field_value)

        # Add values to user-defined fields
        for field_name, field_type, field_code in self.config['extra_fields'].get(obj_name, []):
            field = getattr(root_object, field_name)
            exec(field_code, dict(root_object=root_object,
                                  python_object=python_object,
                                  field=field,
                                  self=self))

    def _get_root_obj(self, py_object):
        """Return root object corresponding to py_object
        from last collection of models of corresponding type seen in event"""
        element_name = py_object.__class__.__name__
        py_list, root_vector = self.last_collection[element_name]
        # Look up element in py_list, set TRef to corresponding entry in root_vector
        # Normal python assignment just changes what the python name points to and won't work.
        # TRef apparently has an __assign__ method we need to use.
        # This isn't a python magic method, as far as I can tell... so why name it with __'s????
        return root_vector[py_list.index(py_object)]

    def _make_tref(self, py_object):
        """Makes a new TRef object pointing to self._get_root_obj(py_object)"""
        tref = ROOT.TRef()
        tref.__assign__(self._get_root_obj(py_object))
        return tref

    def write_to_disk(self):
        self.event_tree.Write()

    def shutdown(self):
        self.write_to_disk()
        self.f.Close()
        # self.cleanup()

    def _build_model_class(self, model):
        """Return ROOT C++ class definition corresponding to instance of data_model.Model
        """
        model_name = model.__class__.__name__
        self.log.debug('Building ROOT class for %s' % model_name)

        type_mapping = {'float':    'float',
                        'float64':  'double',
                        'float32':  'float',
                        'int':      'int',
                        'int16':    'short int',
                        'int32':    'int',
                        'int64':    'long int',
                        'bool':     'bool',
                        'bool_':    'bool',
                        'long':     'long long'}

        list_field_info = model.get_list_field_info()
        class_attributes = ''
        child_classes_code = ''
        for field_name, field_value in model.get_fields_data():
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
                # TODO: do we need to add "//->" ??
                class_attributes += '\tvector <%s*>  %s;\n' % (element_model_name, field_name)

            # References (e.g. interaction.s1)
            elif isinstance(field_value, data_model.Model):
                class_attributes += '\tTRef %s;\n' % field_name

            # Numpy array (assumed fixed-length, 1-d)
            elif isinstance(field_value, np.ndarray):
                class_attributes += '\t%s  %s[%d];\n' % (type_mapping[field_value.dtype.type.__name__],
                                                         field_name,
                                                         len(field_value))

            # Strings (configurable minimum length)
            elif isinstance(field_value, str):
                class_attributes += '\tchar  %s[%d];\n' % (field_name, self.config['string_field_length'])

            # Everything else (int, float, bool)
            else:
                class_attributes += '\t%s  %s;\n' % (type_mapping[type(field_value).__name__],
                                                     field_name)

        # Add any user-defined extra fields
        for field_name, field_type, field_code in self.config['extra_fields'].get(model_name, []):
            # Hack for fields where you need a * before name
            if not field_type.endswith('*'):
                field_type += ' '
            class_attributes += '\t%s%s;\n' % (field_type, field_name)

        return class_template.format(class_name=model_name,
                                     data_attributes=class_attributes,
                                     child_classes_code=child_classes_code,
                                     class_version=pax.__version__.replace('.', ''))

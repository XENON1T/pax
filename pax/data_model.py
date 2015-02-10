"""
Python skeleton for data structure
Extends python object to do a few tricks
"""
import json

import numpy as np

from pax.utils import Memoize


class Model(object):
    """Data modelling base class -- use for subclassing.
    Features:
      - set attribute values by passing dict or kwargs to init
      - get_data_fields(): iterates over the user-specified attributes
                          (that are not methods, properties, _internals)
      - safely declare attributes defaulting to empty lists ([])
        in the class declaration. some_field = (SomeClass,) in class declaration
        means you promise to fill the list  with SomeClass instances.
        TODO: this isn't actually checked, even in StrictModel
      - dump as dictionary and JSON
    """

    def __init__(self, kwargs_dict=None, **kwargs):

        # Initialize the list fields
        # super() is needed to bypass type checking in StrictModel
        for field_name in self._get_list_field_names():
            super().__setattr__(field_name, [])

        # Initialize all attributes from kwargs and kwargs_dict
        kwargs.update(kwargs_dict or {})
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod        # Use only in initialization (or if attributes are fixed, as for StrictModel)
    @Memoize            # Caching decorator, improves performance if a model is initialized often
    def _get_list_field_names(cls):
        """Get the field names of all list fields in this class
        """
        list_field_names = []
        for k, v in cls.__dict__.items():
            if isinstance(v, tuple) and len(v) == 1 and type(v[0]) == type:
                list_field_names.append(k)
        return list_field_names

    def __str__(self):
        return str(self.__dict__)

    def get_fields_data(self):
        """Iterator over (key, value) tuples of all user-specified fields
        Returns keys in lexical order
        """
        # TODO: increase performance by pre-sorting keys?
        # self.__dict__.items() does not return default values set in class declaration
        # Hence we need something more complicated
        class_dict = self.__class__.__dict__
        self_dict = self.__dict__
        for field_name in sorted(class_dict.keys()):
            if field_name in self_dict:
                # The instance has a value for this field: return it
                yield (field_name, self_dict[field_name])
            else:
                # ... it doesnt. Should we return its value?
                if field_name.startswith('_'):
                    continue    # No, is internal
                value_in_class = class_dict[field_name]
                if callable(value_in_class):
                    continue    # No, is a method
                if isinstance(value_in_class, property):
                    continue    # No, is a property
                # Yes, yield the class-level value
                yield (field_name, value_in_class)

    def to_dict(self, json_compatible=False):
        result = {}
        for k, v in self.get_fields_data():
            if isinstance(v, list):
                result[k] = [el.to_dict(json_compatible) for el in v]
            elif isinstance(v, np.ndarray) and json_compatible:
                # For JSON compatibility, numpy arrays must be converted to lists
                result[k] = v.tolist()
            else:
                result[k] = v
        return result

    def to_json(self):
        return json.dumps(self.to_dict(json_compatible=True))


casting_allowed_for = {
    int:    ['int16', 'int32', 'int64'],
    float:  ['int', 'float32', 'float64', 'int16', 'int32', 'int64'],
}


class StrictModel(Model):
    """Model which enforces additional restrictions:
      - can't add new attributes: have to be fixed at class declaration
      - attributes can't change type or numpy dtype once set.
    """

    def __setattr__(self, key, value):

        # Get the old attr.
        # #Will raise AttributeError if doesn't exists, which is what we want
        old_val = getattr(self, key)
        old_type = type(old_val)
        new_type = type(value)

        # Check for attempted type change
        if old_type != new_type:

            # Are we allowed to cast the type?
            if old_type in casting_allowed_for \
                    and value.__class__.__name__ in casting_allowed_for[old_type]:
                value = old_type(value)

            else:
                raise TypeError('Attribute %s of class %s should be a %s, not a %s. '
                                % (key, self.__class__.__name__, old_type, new_type))

        # Check for attempted dtype change
        if isinstance(old_val, np.ndarray):
            if old_val.dtype != value.dtype:
                raise TypeError('Attribute %s of class %s should have numpy dtype %s, not %s' % (
                    key, self.__class__.__name__, old_val.dtype, value.dtype))

        super().__setattr__(key, value)

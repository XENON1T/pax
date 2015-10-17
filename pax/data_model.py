"""
Python skeleton for data structure
Extends python object to do a few tricks
"""
import json
import bson
import six
import numpy as np

from pax.utils import Memoize


class Model(object):
    """Data modelling base class -- use for subclassing.
    Features:
      - set attribute values by passing dict or kwargs to init
      - get_data_fields(): iterates over the user-specified attributes
                          (that are not methods, properties, _internals)
      - safely declare attributes defaulting to empty lists ([])
        in the class declaration. some_field = ListField(SomeType,) in class declaration
        means you promise to fill the list with SomeType instances.
        NB: This is checked on initialization (even for Model) not for appending (even for StrictModel)!
            (we'd have to mess with / override list for that)
      - recursive initializiation of subclasses
      - dump as dictionary and JSON
    """

    def __init__(self, kwargs_dict=None, **kwargs):
        # Initialize the collection fields to empty lists
        # object.__setattr__ is needed to bypass type checking in StrictModel
        list_field_info = self.get_list_field_info()
        for field_name in list_field_info:
            object.__setattr__(self, field_name, [])

        # Initialize all attributes from kwargs and kwargs_dict
        kwargs.update(kwargs_dict or {})
        for k, v in kwargs.items():
            if k in list_field_info:
                # User gave a value to initialize a list field. Hopefully an iterable!
                # Let's check if the types are correct
                desired_type = list_field_info[k]
                temp_list = []
                for el in v:
                    if isinstance(el, desired_type):
                        # Good, pass through unmolested
                        temp_list.append(el)
                    elif isinstance(el, dict):
                        # Dicts are fine too, we can use them to init the desired type
                        temp_list.append(desired_type(**el))
                    else:
                        raise ValueError("Attempt to initialize list field %s with type %s, "
                                         "but you promised type %s in class declaration." % (k,
                                                                                             type(el),
                                                                                             desired_type))
                # This has to be a list of dictionaries
                # suitable to be passed to __init__ of the list field's element type
                setattr(self, k, temp_list)
            else:
                default_value = getattr(self, k)
                if type(default_value) == np.ndarray:
                    if isinstance(v, np.ndarray):
                        pass
                    elif isinstance(v, bytes):
                        # Numpy arrays can be also initialized from a 'string' of bytes...
                        v = np.fromstring(v, dtype=default_value.dtype)
                    elif hasattr(v, '__iter__'):
                        # ... or an iterable
                        v = np.array(v, dtype=default_value.dtype)
                    else:
                        raise ValueError("Can't initialize field %s: "
                                         "don't know how to make a numpy array from a %s" % (k, type(v)))
                elif isinstance(default_value, Model):
                    v = default_value.__class__(**v)

                setattr(self, k, v)

    @classmethod        # Use only in initialization (or if attributes are fixed, as for StrictModel)
    @Memoize            # Caching decorator, improves performance if a model is initialized often
    def get_list_field_info(cls):
        """Return dict with fielname => type of elements in collection fields in this class
        """
        list_field_info = {}
        for k, v in cls.__dict__.items():
            if isinstance(v, ListField):
                list_field_info[k] = v.element_type
        return list_field_info

    def __str__(self):
        return str(self.__dict__)

    def get_fields_data(self):
        """Iterator over (key, value) tuples of all user-specified fields
        Returns keys in lexical order
        """
        # TODO: increase performance by pre-sorting keys?
        # self.__dict__.items() does not return default values set in class declaration
        # Hence we need something more complicated

        if type(self) == type:
            # Called as a class method
            class_dict = self.__dict__
            self_dict = {}
        else:
            # Called as instance method
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
                if isinstance(value_in_class, (property, classmethod)):
                    continue    # No, is a property or classmethod
                # Yes, yield the class-level value
                yield (field_name, value_in_class)

    def to_dict(self, convert_numpy_arrays_to=None, fields_to_ignore=None):
        result = {}
        if fields_to_ignore is None:
            fields_to_ignore = tuple()
        for k, v in self.get_fields_data():
            if k in fields_to_ignore:
                continue
            if isinstance(v, Model):
                result[k] = v.to_dict(convert_numpy_arrays_to=convert_numpy_arrays_to,
                                      fields_to_ignore=fields_to_ignore)
            elif isinstance(v, list):
                result[k] = [el.to_dict(convert_numpy_arrays_to=convert_numpy_arrays_to,
                                        fields_to_ignore=fields_to_ignore) for el in v]
            elif isinstance(v, np.ndarray) and convert_numpy_arrays_to is not None:
                if convert_numpy_arrays_to == 'list':
                    result[k] = v.tolist()
                elif convert_numpy_arrays_to == 'bytes':
                    result[k] = bson.Binary(v.tostring())
                else:
                    raise ValueError('convert_numpy_arrays_to must be "list" or "bytes"')
            else:
                result[k] = v
        return result

    def to_json(self, fields_to_ignore=None):
        return json.dumps(self.to_dict(convert_numpy_arrays_to='list',
                                       fields_to_ignore=fields_to_ignore))

    @classmethod
    def get_dtype(cls):
        type_mapping = {'int':    np.int64,
                        'float':  np.float64,
                        'bool':   np.bool_}
        dtype = []
        for field_name, default_value in cls.get_fields_data(cls):
            value_type = default_value.__class__.__name__
            if value_type in type_mapping:
                dtype.append((field_name, type_mapping[value_type]))
        return np.dtype(dtype)

    def to_bson(self, fields_to_ignore=None):
        return bson.BSON.encode(self.to_dict(convert_numpy_arrays_to='bytes',
                                             fields_to_ignore=fields_to_ignore))

    @classmethod
    def from_json(cls, x):
        return cls(**json.loads(x))

    @classmethod
    def from_bson(cls, x):
        if six.PY2:
            # Hack for python 2: may work in py3 too, but it's definitely not the standard way!
            reader = bson.decode_file_iter(six.BytesIO(x))
            event_dict = next(reader)
        else:
            event_dict = bson.BSON.decode(x)
        return cls(**event_dict)


casting_allowed_for = {
    'int':    ['int16', 'int32', 'int64', 'Int64', 'Int32', 'long'],
    'float':  ['int', 'int16', 'int32', 'int64', 'Int64', 'Int32', 'float32', 'float64', 'long'],
    'bool':   ['int', 'int16', 'int32', 'int64', 'Int64', 'Int32', 'long'],
    'long':   ['int', 'int16', 'int32', 'int64', 'Int64', 'Int32'],
    'str':    ['unicode']
}


class ListField(object):
    def __init__(self, element_type):
        if not issubclass(element_type, Model):
            raise ValueError("Model collections must specify a type")
        self.element_type = element_type


class StrictModel(Model):
    """Model which enforces additional restrictions:
      - can't add new attributes: have to be fixed at class declaration
      - attributes can't change type or numpy dtype once set.
    """

    def __setattr__(self, key, value):

        # Get the old attr.
        # Will raise AttributeError if doesn't exists, which is what we want
        old_val = getattr(self, key)
        old_type = type(old_val)
        new_type = type(value)

        # Check for attempted type change
        if old_type != new_type:
            old_class_name = old_val.__class__.__name__
            new_class_name = value.__class__.__name__

            # Are we allowed to cast the type?
            if old_class_name in casting_allowed_for and new_class_name in casting_allowed_for[old_class_name]:
                value = old_type(value)

            else:
                raise TypeError('Attribute %s of class %s should be a %s, not a %s. '
                                % (key,
                                   self.__class__.__name__,
                                   old_class_name,
                                   new_class_name))

        # Check for attempted dtype change
        if old_type == np.ndarray:
            if old_val.dtype != value.dtype:
                raise TypeError('Attribute %s of class %s should have numpy dtype %s, not %s' % (
                    key, self.__class__.__name__, old_val.dtype, value.dtype))

        Model.__setattr__(self, key, value)

"""
Python skeleton for data structure
Extends python object to do a few tricks
"""
import numpy as np


class Model(object):
    """Data modelling base class -- use for subclassing.
    Features:
      - set attribute values by passing dict or kwargs to init
      - get_data_fields(): iterates over the user-specified attributes (that are not methods, properties, _internals)
      - safely declare attributes defaulting to empty lists ([]) in the class declaration
        some_field = (SomeClass,) in class declaration means you promise to fill the list with SomeClass instances
        TODO: this isn't actually checked, even in StrictModel
    """

    def __init__(self, kwargs_dict=None, **kwargs):

        # Do we already know what the list fields of this class are?
        if not hasattr(self, '_list_fields'):
            # We need to figure out the list fields in this class
            list_fields = []   # list of (field_name, promised_type) tuples
            for attr in dir(self):
                val = getattr(self, attr)
                # list fields are specified as (ClassName,) 1-tuples, with ClassName the class of stuff in the list
                if isinstance(val, tuple) and len(val) == 1 and type(val[0]) == type:
                    list_fields.append((attr, val[0]))

            # Conver to tuple, so we're not storing a mutable as a class attribute
            list_fields = tuple(list_fields)

            # Store _list_fields in this object
            # Need to use object.__setattr__ to override type checking
            object.__setattr__(self, '_list_fields', list_fields)

            # This took a few CPU cycles, so store as a class attribute too,
            # next time we init this object we won't have to go through this
            # (note this calls the ordinary python object's setattr, 'class' is just a normal object)
            # maybe downcast to tuple of 2-tuples..
            setattr(self.__class__, '_list_fields', list_fields)

            print("Figured out _list_fields for %s: %s" % (self.__class__.__name__, list_fields))

        # Give this instance a new list in the list fields
        # Could write a list extension that checks types... too lazy right now
        for field_name, wrapped_class in self._list_fields:
            object.__setattr__(self, field_name, [])

        # Initialize the kwargs as attrs
        if kwargs_dict:
            kwargs.update(kwargs_dict)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError('Invalid argument %s to %s.__init__' % (k, self.__class__.__name__))
            setattr(self, k, v)

    def get_data_fields(self, except_for=()):
        """Iterate over (name, data) tuples for all fields in the model
        """
        # Loop over all attrs of self
        for field_name in dir(self):

            # Filter out internal stuff
            if field_name.startswith('_'):
                continue

            # Filter out properties, see stackoverflow.com/questions/17735520
            try:
                value_in_class = getattr(self.__class__, field_name)
                if isinstance(value_in_class, property):
                    continue
            except AttributeError:
                # Apparently it wasn't a class attribute, so it won't be a property
                continue



            # Ignore fields we want to ignore
            if field_name in except_for:
                continue

            field_value = getattr(self, field_name)

            # Filter out methods
            if callable(field_value):
                continue

            yield (field_name, field_value)


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

        # Get the old attr. Will raise AttributeError if doesn't exists, which is what we want
        old_val = getattr(self, key)
        old_type = type(old_val)
        new_type = type(value)

        # Check for attempted type change

        if old_type != new_type:

            # Check if we are allowed to cast the type
            if old_type in casting_allowed_for and value.__class__.__name__ in casting_allowed_for[old_type]:
                value = old_type(value)

            else:
                raise TypeError('Attribute %s of class %s should be a %s, not a %s. '
                                % (key, self.__class__.__name__, old_type, new_type))

        # Check for attempted dtype change
        if isinstance(old_val, np.ndarray):
            if old_val.dtype != value.dtype:
                raise TypeError('Attribute %s of class %s should have dtype %s, not %s' % (
                    key, self.__class__.__name__, old_val.dtype, value.dtype))

        super().__setattr__(key, value)

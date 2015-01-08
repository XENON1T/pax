"""
Python skeleton for data structure
Extends python object to do a few tricks
"""
import numpy as np


class Model(object):
    """Extended python object with few new features:/restrictions
      - can set attrs by passing dict or kwargs to init
      - some_field = (SomeClass,) in class declaration allows a list of SomeClass
        To get promised type for some_field, do self.get_list_fields()['some_field'] -> SomeClass
    """

    def __init__(self, kwargs_dict=None, **kwargs):

        # Do we already know what the list fields of this class are?
        # print("initializing %s..." % self.__class__.__name__)

        if not hasattr(self, '_list_fields'):
            # We need to figure out the list fields in this class
            #print("Don't know list fields yet")
            list_fields = {}
            for attr in dir(self):
                val = getattr(self, attr)
                # list fields are specified as (ClassName,) 1-tuples, with ClassName the class of stuff in the list
                if isinstance(val, tuple) and len(val) == 1 and type(val[0]) == type:
                    list_fields[attr] = val[0]

            # Store _list_fields in this object
            # Need to use object.__setattr__ to override type checking
            object.__setattr__(self, '_list_fields', list_fields)

            # This took a few CPU cycles, so store as a class attribute too,
            # next time we init this object we won't have to go through this
            # (note this calls the ordinary python object's setattr, 'class' is just a normal object)
            # TODO: somewhat dangerous to store dict as class attr.. (mutable)
            # maybe downcast to tuple of 2-tuples..
            setattr(self.__class__, '_list_fields', list_fields)

            # print("Determined list fields to be %s" % list_fields)

        # print("List fields are %s" % self._list_fields)
        # Give this instance a new list in this field
        # Could write a list extension that checks types... too lazy right now
        for field_name, wrapped_class in self._list_fields.items():
            object.__setattr__(self, field_name, [])

        # Initialize the kwargs as attrs
        if kwargs_dict:
            kwargs.update(kwargs_dict)
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise ValueError('Invalid argument %s to %s.__init__' % (k, self.__class__.__name__))
            setattr(self, k, v)

    def get_list_fields(self):
        return self._list_fields



casting_allowed_for = {
    int:    ['int16', 'int32', 'int64'],
    float:  ['int', 'float32', 'float64', 'int16', 'int32', 'int64'],
}


class StrictModel(Model):
    """Model which enforces additional restrictions:
      - can't add new attributes: have to be set at class level
      - attributes are not allowed to change type once set
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

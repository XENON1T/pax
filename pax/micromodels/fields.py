"""Fork of micromodels

https://github.com/j4mie/micromodels

Add numpy support and quite a few ways the code operates.
"""

import numpy as np
import logging
try:
    from tables import Int64Col, Float64Col, StringCol
except:
    print("Pytables is not installed, you can't use the HDF5 I/O!!")
    def dummy(x=None): return None
    Int64Col = Float64Col = StringCol = dummy


class BaseField(object):

    """Base class for all field types.

    The ``source`` parameter sets the key that will be retrieved from the source
    data. If ``source`` is not specified, the field instance will use its own
    name as the key to retrieve the value from the source data.

    """

    def __init__(self, default=0, source=None, hdf5_type=None):
        self._default = default
        self.source = source
        self._hdf5_type = hdf5_type

    def populate(self, data):
        """Set the value or values wrapped by this field"""

        self.data = data

    def to_python(self):
        '''After being populated, this method casts the source data into a
        Python object. The default behavior is to simply return the source
        value. Subclasses should override this method.

        '''
        return self.data

    def to_serial(self, data):
        '''Used to serialize forms back into JSON or other formats.

        This method is essentially the opposite of
        :meth:`~micromodels.fields.BaseField.to_python`. A string, boolean,
        number, dictionary, list, or tuple must be returned. Subclasses should
        override this method.

        '''
        return data

    def hdf5_type(self):
        """Used to say what HDF5 type this corresponds to
        """
        return self._hdf5_type


class NumpyArrayField(BaseField):

    """Field to represent a numpy array"""

    def __init__(self, dtype, **kwargs):
        self._dtype = dtype

        if 'default' not in kwargs:
            kwargs['default'] = np.array([], dtype=dtype)

        BaseField.__init__(self, **kwargs)

    def to_python(self):
        if isinstance(self.data, np.ndarray):
            if self.data.dtype != self._dtype:
                logging.warning("Casting.  Wrong data type; must be %s, not %s." % (str(self._dtype),
                                                                                    str(self.data.dtype)))
                self.data = self.data.astype(self._dtype, copy=False)
            return self.data
        elif isinstance(self.data, set):
            logging.warning("Converting set to list then numpy array.")
            return np.array(list(self.data),
                            dtype=self._dtype)
        elif isinstance(self.data, (list, tuple)):
            logging.warning("Converting list/tuple to numpy array.")
            return np.array(self.data, dtype=self._dtype)
        else:
            raise TypeError("Data must be array: not %s, %s" % (str(type(self.data)),
                                                                str(self.data)))

    def to_serial(self, model_instances):
        return self.data.tolist()


class StringField(BaseField):

    """Field to represent a simple Unicode string value."""

    def __init__(self, hdf5_type=StringCol(32), **kwargs):
        BaseField.__init__(self, hdf5_type=hdf5_type, **kwargs)

    def to_python(self):
        """Convert the data supplied using the :meth:`populate` method to a
        Unicode string.

        """
        if self.data is None:
            return ''
        return str(self.data)


class IntegerField(BaseField):

    """Field to represent an integer value"""

    def __init__(self, hdf5_type=Int64Col(), **kwargs):
        BaseField.__init__(self, hdf5_type=hdf5_type, **kwargs)

    def to_python(self):
        """Convert the data supplied to the :meth:`populate` method to an
        integer.

        """
        if self.data is None:
            return 0
        return int(self.data)


class FloatField(BaseField):

    """Field to represent a floating point value"""

    def __init__(self, hdf5_type=Float64Col(), **kwargs):
        BaseField.__init__(self, hdf5_type=hdf5_type, **kwargs)

    def to_python(self):
        """Convert the data supplied to the :meth:`populate` method to a
        float.

        """
        if self.data is None:
            return 0.0
        return float(self.data)


class BooleanField(BaseField):

    """Field to represent a boolean"""

    def to_python(self):
        """The string ``'True'`` (case insensitive) will be converted
        to ``True``, as will any positive integers.

        """
        if isinstance(self.data, str):
            return self.data.strip().lower() == 'true'
        if isinstance(self.data, int):
            return self.data > 0
        return bool(self.data)


class WrappedObjectField(BaseField):

    """Superclass for any fields that wrap an object"""

    def __init__(self, wrapped_class, related_name=None, **kwargs):
        self._wrapped_class = wrapped_class
        self._related_name = related_name
        self._related_obj = None

        BaseField.__init__(self, **kwargs)


class ModelField(WrappedObjectField):

    """Field containing a model instance

    Use this field when you wish to nest one object inside another.
    It takes a single required argument, which is the nested class.
    For example, given the following dictionary::

        some_data = {
            'first_item': 'Some value',
            'second_item': {
                'nested_item': 'Some nested value',
            },
        }

    You could build the following classes
    (note that you have to define the inner nested models first)::

        class MyNestedModel(micromodels.Model):
            nested_item = micromodels.CharField()

        class MyMainModel(micromodels.Model):
            first_item = micromodels.CharField()
            second_item = micromodels.ModelField(MyNestedModel)

    Then you can access the data as follows::

        >>> m = MyMainModel(some_data)
        >>> m.first_item
        u'Some value'
        >>> m.second_item.__class__.__name__
        'MyNestedModel'
        >>> m.second_item.nested_item
        u'Some nested value'

    """

    def to_python(self):
        if isinstance(self.data, self._wrapped_class):
            obj = self.data
        else:
            obj = self._wrapped_class.from_dict(self.data or {})

        # Set the related object to the related field
        if self._related_name is not None:
            setattr(obj, self._related_name, self._related_obj)

        return obj

    def to_serial(self, model_instance):
        return model_instance.to_dict(serial=True)


class ModelCollectionField(WrappedObjectField):

    """Field containing a list of model instances.

    Use this field when your source data dictionary contains a list of
    dictionaries. It takes a single required argument, which is the name of the
    nested class that each item in the list should be converted to.
    For example::

        some_data = {
            'list': [
                {'value': 'First value'},
                {'value': 'Second value'},
                {'value': 'Third value'},
            ]
        }

        class MyNestedModel(micromodels.Model):
            value = micromodels.CharField()

        class MyMainModel(micromodels.Model):
            list = micromodels.ModelCollectionField(MyNestedModel)

        >>> m = MyMainModel(some_data)
        >>> len(m.list)
        3
        >>> m.list[0].__class__.__name__
        'MyNestedModel'
        >>> m.list[0].value
        u'First value'
        >>> [item.value for item in m.list]
        [u'First value', u'Second value', u'Third value']

    """

    def to_python(self):
        object_list = []
        for item in self.data:
            if isinstance(item, self._wrapped_class):
                obj = item
            else:
                obj = self._wrapped_class.from_dict(item)
                if self._related_name is not None:
                    setattr(obj, self._related_name, self._related_obj)
            object_list.append(obj)

        return object_list

    def to_serial(self, model_instances):
        return [instance.to_dict(serial=True) for instance in model_instances]

    def append(self, value):
        self.list = self.to_python().append(value)

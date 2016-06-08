"""Tools for working with numpy structured arrays
Extends existing functionality in numpy.lib.recfunctions
"""

import numpy as np
from numpy.lib import recfunctions
from collections import OrderedDict

rename_fields = recfunctions.rename_fields


def append_fields(base, names, data, dtypes=None, fill_value=-1,
                  usemask=False,   # Different from recfunctions default
                  asrecarray=False):
    """Append fields to numpy structured array
    If fields already exists in data, will overwrite
    """
    if isinstance(names, (tuple, list)):
        # Add multiple fields at once
        if dtypes is None:
            dtypes = [d.dtype for d in data]
        # Convert to numpy arrays so we can use boolean index arrays
        names = np.array(names)
        data = np.array(data)
        dtypes = np.array(dtypes)
        not_yet_in_data = True ^ np.in1d(names, base.dtype.names)
        # Append the fields that were not in the data
        base = recfunctions.append_fields(base,
                                          names[not_yet_in_data].tolist(),
                                          data[not_yet_in_data].tolist(),
                                          dtypes[not_yet_in_data].tolist(),
                                          fill_value, usemask, asrecarray)
        # Overwrite the fields that are already in the data
        for i in np.where(True ^ not_yet_in_data)[0]:
            base[names[i]] = data[i]
        return base
    else:
        # Add single field
        if names in base.dtype.names:
            # Field already exists: overwrite data
            base[names] = data
            return base
        else:
            return recfunctions.append_fields(base, names, data, dtypes,
                                              fill_value, usemask, asrecarray)


def drop_fields(arr, *args, **kwargs):
    """Drop fields from numpy structured array
    Gives error if fields don't exist
    """
    return recfunctions.drop_fields(arr, usemask=False, *args, **kwargs)


def drop_fields_if_exist(arr, fields):
    return drop_fields(arr, [f for f in fields if f in arr.dtype.names])


def fields_view(arr, fields):
    """View one or several columns from a numpy record array"""
    # Single field is easy:
    if isinstance(fields, str):
        return arr[fields]
    for f in fields:
        if f not in arr.dtype.names:
            raise ValueError("Field %s is not in the array..." % f)
    # Don't know how to do it for multiple fields, make a copy for now... (probably?)
    return drop_fields(arr, [f for f in arr.dtype.names if f not in fields])
    # The solution in
    # http://stackoverflow.com/questions/15182381/how-to-return-a-view-of-several-columns-in-numpy-structured-array
    # doesn't work in combination with filter_on_fields...
    # dtype2 = np.dtype({name:arr.dtype.fields[name] for name in columns})
    # return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)


def filter_on_fields(to_filter, for_filter, filter_fields, filter_fields_2=None, return_selection=False):
    """Returns entries of to_filter whose combination of the filter_fields values are present in for_filter.
    filter_fields_2: names of filter_fields in for_filter (if different than in to_filter)
    If return_selection, will instead return boolean selection array for to_filter
    """
    a = np.array(fields_view(to_filter, filter_fields))
    if filter_fields_2 is None:
        filter_fields_2 = filter_fields
    b = np.array(fields_view(for_filter, filter_fields_2))
    # Rename the fields, if needed
    # If only one field is selected, this won't be needed (and would return None instead of working)
    if not isinstance(filter_fields, str) and len(filter_fields) > 1:
        b = recfunctions.rename_fields(b, dict(zip(filter_fields_2, filter_fields)))
    selection = np.in1d(a, b)
    if return_selection:
        return selection
    else:
        return to_filter[selection]


def group_by(x, group_by_fields='Event', return_group_indices=False):
    """Splits x into LIST of arrays, each array with rows that have same group_by_fields values.
    Gotchas:
        Assumes x is sorted by group_by_fields (works in either order, reversed or not)
        Does NOT put in empty lists if indices skip a value! (e.g. events without peaks)
    If return_indices=True, returns list of arrays with indices of group elements in x instead
    """

    # Support single index and list of indices
    try:
        group_by_fields[0]
    except TypeError:
        group_by_fields = tuple(group_by_fields)

    # Define array we'll split
    if return_group_indices:
        to_return = np.arange(len(x))
    else:
        to_return = x

    if not len(x):
        return []

    # Indices to determine split points from
    indices = fields_view(x, group_by_fields)

    # Should we split at all?
    if indices[0] == indices[-1]:
        return [to_return]
    else:
        # Split where indices change value
        split_points = np.where((np.roll(indices, 1) != indices))[0]
        # 0 shouldn't be a split_point, will be in it due to roll (and indices[0] != indices[-1]), so remove it
        split_points = split_points[1:]
        return np.split(to_return, split_points)


def dict_group_by(x, group_by_fields='Event', return_group_indices=False):
    """Same as group_by, but returns OrderedDict of value -> group,
    where value is the value (or tuple of values) of group_by_fields in each subgroup
    Gotcha: assumes x is sorted by group_by_fields (works in either order, reversed or not)
    See also group_by
    """
    groups = group_by(x, group_by_fields, return_group_indices)
    return OrderedDict([(fields_view(gr[0:1], group_by_fields)[0], gr) for gr in groups])


def fields_data(arr, ignore_fields=None):
    if ignore_fields is None:
        ignore_fields = []
    """Returns list of arrays of data for each single field in arr"""
    return [arr[fn] for fn in arr.dtype.names if fn not in ignore_fields]

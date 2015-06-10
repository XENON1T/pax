import numpy as np
from numpy.lib import recfunctions


# def merge_dicts(x, y):
#     '''Given two dicts, merge them into a new dict as a shallow copy.
#     Stolen from http://stackoverflow.com/questions/38987/
#     how-can-i-merge-two-python-dictionaries-in-a-single-expression
#     '''
#     z = x.copy()
#     z.update(y)
#     return z

def append_fields(arr, *args, **kwargs):
    """Append fields to numpy structured array"""
    return recfunctions.append_fields(arr, usemask=False, *args, **kwargs)


def fields_view(arr, columns):
    """ View several columns from a numpy record array
    Stolen from http://stackoverflow.com/questions/15182381/how-to-return-a-view-of-several-columns-in-numpy-structured-array
    """
    # Single columns is easy:
    if isinstance(columns, str):
        return arr[columns]
    # Several
    dtype2 = np.dtype({name: arr.dtype.fields[name] for name in columns})
    return np.ndarray(arr.shape, dtype2, arr, 0, arr.strides)


def filter_on_fields(to_filter, for_filter, filter_fields):
    """Returns entries of to_filter whose combination of the filter_fields values
    are present in for_filter.
    Find better name
    """
    return to_filter[np.in1d(
        fields_view(to_filter, filter_fields),
        fields_view(for_filter, filter_fields)
    )]


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
    except TypeErrror:
        group_by_fields = tuple(group_by_fields)

    # Define array we'll split
    if return_group_indices:
        to_return = np.arange(len(x))
    else:
        to_return = x

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


def fields_data(arr, ignore_fields=None):
    if ignore_fields is None:
        ignore_fields = []
    """Returns list of arrays of data for each single field in arr"""
    return [arr[fn] for fn in arr.dtype.names if fn not in ignore_fields]

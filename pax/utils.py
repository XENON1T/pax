"""Helper routines needed in pax

Please only put stuff here that you *really* can't find any other place for!
e.g. a list clustering routine that isn't in some standard, library but several plugins depend on it
"""

import re
import inspect
import logging
import time
import os
import glob

log = logging.getLogger('pax_utils')


##
# Utilities for finding files inside pax.
##

# Store the directory of pax (i.e. this file's directory) as PAX_DIR
PAX_DIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))


def data_file_name(filename):
    """Returns filename if a file exists there, else returns PAX_DIR/data/filename"""
    if os.path.exists(filename):
        return filename
    new_filename = os.path.join(PAX_DIR, 'data', filename)
    if os.path.exists(new_filename):
        return new_filename
    else:
        raise ValueError('File name or path %s not found!' % filename)


def get_named_configuration_options():
    """ Return the names of all working named configurations
    """
    config_files = []
    for filename in glob.glob(os.path.join(PAX_DIR, 'config', '*.ini')):
        filename = os.path.basename(filename)
        m = re.match(r'(\w+)\.ini', filename)
        if m is None:
            print("Weird file in config dir: %s" % filename)
        filename = m.group(1)
        # Config files starting with '_' won't appear in the usage list (they won't work by themselves)
        if filename[0] == '_':
            continue
        config_files.append(filename)
    return config_files


# Caching decorator
# Stolen from http://avinashv.net/2008/04/python-decorators-syntactic-sugar/
class Memoize:

    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
            return self.memoized[args]


class Timer:
    """Simple stopwatch timer
    punch() returns ms since timer creation or last punch
    """
    last_t = 0

    def __init__(self):
        self.punch()

    def punch(self):
        now = time.time()
        result = (now - self.last_t) * 1000
        self.last_t = now
        return result

import inspect
import time
import logging
import os
from glob import glob
from copy import deepcopy

import numba
import numpy as np
import h5py

import pax          # For version number
from pax.utils import PAX_DIR
from pax.datastructure import TriggerSignal

times_dtype = np.dtype([('time', np.int64),
                        ('pmt', np.int32),
                        ('area', np.float64)
                        ])


class TriggerPlugin(object):

    def __init__(self, trigger, config):
        self.trigger = trigger
        self.config = config
        self.pmt_data = self.trigger.pmt_data
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)
        self.log.info("Logging started for %s" % self.name)
        self.end_of_run_info = {'config': {k: v for k, v in self.config.items() if k not in self.trigger.config}}
        self.startup()

    def startup(self):
        self.log.debug("%s does not define a startup." % self.name)

    def shutdown(self):
        self.log.debug("%s does not define a shutdown." % self.name)


class TriggerData(object):
    times = np.array([], dtype=times_dtype)
    signals = np.array([], dtype=TriggerSignal.get_dtype())
    event_ranges = np.zeros((0, 2), dtype=np.int64)
    trigger_times = np.array([], dtype=np.int64)
    signals_by_event = []
    last_data = False
    last_time_searched = 0


class Trigger(object):
    """XENON1T trigger main class
    This is responsible for running data by each of the trigger modules.
    """

    def __init__(self, pax_config):
        self.log = logging.getLogger('Trigger')
        self.config = pax_config['Trigger']
        self.pmt_data = pax_config['DEFAULT']['pmts']

        # Build a (module, channel) ->  lookup matrix
        # I whish numba had some kind of dictionary / hashtable support...
        # but this will work as long as the module serial numbers are small :-)
        # I will asssume always and everywhere the pmt position numbers start at 0 and increase by 1 continuously!
        # Initialize the matrix to n_channels, which is one above the last PMT
        # This will ensure we do not crash on data in 'ghost' channels (not plugged in,
        # do report data in self-triggered mode)
        pmt_data = self.pmt_data
        n_channels = len(pmt_data)
        max_module = max([q['digitizer']['module'] for q in pmt_data])
        max_channel = max([q['digitizer']['channel'] for q in pmt_data])
        self.pmt_lookup = n_channels * np.ones((max_module + 1, max_channel + 1), dtype=np.int)
        for q in pmt_data:
            module = q['digitizer']['module']
            channel = q['digitizer']['channel']
            self.pmt_lookup[module][channel] = q['pmt_position']

        # Create file & datasets for extra trigger data
        dir = os.path.dirname(self.config['trigger_data_filename'])
        if dir and not os.path.exists(dir):
            os.makedirs(dir)
        self.dark_monitor_data_file = h5py.File(self.config['trigger_data_filename'], mode='w')

        self.end_of_run_info = dict(times_read=0,
                                    signals_found=0,
                                    events_built=0,
                                    start_timestamp=time.time(),
                                    pax_version=pax.__version__,
                                    config=self.config)

        # Build a dictionary mapping trigger plugin names to classes
        tm_classes = {}
        for module_filename in glob(os.path.join(PAX_DIR + '/trigger_plugins/*.py')):
            module_name = os.path.splitext(os.path.basename(module_filename))[0]
            if module_name.startswith('_'):
                continue

            # Import the module, after which we can do pax.trigger_plugins.module_name
            __import__('pax.trigger_plugins.%s' % module_name, globals=globals())

            # Now get all the trigger plugins defined in each module
            for tp_name, tp_class in inspect.getmembers(getattr(pax.trigger_plugins, module_name),
                                                        lambda x: type(x) == type and issubclass(x, TriggerPlugin)):
                if tp_name == 'TriggerPlugin':
                    continue
                if tp_name in tm_classes:
                    raise ValueError("Two TriggerModule's named %s!" % tp_name)
                tm_classes[tp_name] = tp_class

        # Initialize the trigger plugins specified in the configuration
        self.plugins = []
        for tp_name in self.config['trigger_plugins']:
            if tp_name not in tm_classes:
                raise ValueError("Don't know a trigger plugin called %s!" % tp_name)
            conf = deepcopy(self.config)
            conf.update(pax_config.get('Trigger.%s' % tp_name, {}))
            self.plugins.append(tm_classes[tp_name](trigger=self, config=conf))

    def run(self, last_time_searched, start_times=tuple(), channels=None, modules=None, areas=None, last_data=False):
        """Run on the specified data, yields ((start time, stop time), signals in event, event type identifier)
        """
        data = TriggerData()
        data.last_time_searched = last_time_searched
        data.last_data = last_data

        # Enter the times into data
        data.times = np.zeros(len(start_times), dtype=times_dtype)
        data.times['time'] = start_times
        if channels is not None and modules is not None:
            # Convert the channel/module specs into pmt numbers.
            get_pmt_numbers(channels, modules, pmts_buffer=data.times['pmt'], pmt_lookup=self.pmt_lookup)
        else:
            # If PMT numbers not specified, pretend everything is from a 'ghost' pmt at # = n_channels
            data.times['pmt'] = len(self.pmt_data)
        if areas is not None:
            data.times['area'] = areas

        # Hand over to each of the trigger plugins in turn.
        # Notice we won't even try to sort the event ranges from different triggers by time: some triggers (like the
        # main trigger) can hold events back until next time we give them data, so this would be a thorny problem.
        for plugin in self.plugins:
            self.log.debug("Passing data to plugin %s" % plugin.name)
            plugin.process(data)

        # Update the end of run info
        self.end_of_run_info['times_read'] += len(data.times)
        self.end_of_run_info['signals_found'] += len(data.signals)
        self.end_of_run_info['events_built'] += len(data.event_ranges)

        for event_i, (start, stop) in enumerate(data.event_ranges):
            yield (start, stop), data.signals_by_event[event_i]

    def shutdown(self):
        """Shut down trigger, return dictionary with end-of-run information."""
        self.log.debug("Shutting down the trigger")
        for p in self.plugins:
            p.shutdown()

        # Close the trigger data file.
        # Note this must be done after shutting down the plugins, they may add something on shutdown as well.
        if hasattr(self, 'f'):
            self.f.close()

        # Add end-of-run info for the runs database
        self.end_of_run_info.update(dict(end_trigger_processing_timestamp=time.time()))
        for p in self.plugins:
            self.end_of_run_info[p.name] = p.end_of_run_info

        return self.end_of_run_info


@numba.jit(nopython=True)
def get_pmt_numbers(channels, modules, pmts_buffer, pmt_lookup):
    """Fills pmts_buffer with pmt numbers corresponding to channels, modules according to pmt_lookup matrix:
     - pmt_lookup: lookup matrix for pmt numbers. First index is digitizer module, second is digitizer channel.
    Modifies pmts_buffer in-place.
    """
    for i in range(len(channels)):
        pmts_buffer[i] = pmt_lookup[modules[i], channels[i]]


def h5py_append(dataset, records):
    """Append records to h5py dataset, resizing axis=0 as needed
    This probably goes completely against the philosophy of hdf5, and is super convenient
    """
    if len(dataset.shape) != len(records.shape):
        # Hack for appending a single record
        want_shape = list(dataset.shape)
        want_shape[0] = -1
        records = records.reshape(tuple(want_shape))
    orig_size = dataset.shape[0]
    dataset.resize(orig_size + records.shape[0], axis=0)
    dataset[orig_size:] = records

import errno
from collections import defaultdict
from copy import deepcopy
from glob import glob
import inspect
import logging
import os
import pickle
import time
import zipfile
import zlib

import numpy as np

import pax          # For version number
from pax.utils import PAX_DIR
from pax.datastructure import TriggerSignal
from pax.exceptions import InvalidConfigurationError


pulse_dtype = np.dtype([('time', np.int64),
                        ('pmt', np.int32),
                        ('area', np.float64),
                        ])


class TriggerData(object):
    """Carries all data from one "batch" between trigger modules"""

    def __init__(self, **kwargs):
        self.pulses = None                                          # Pulses array, see pulse_dtype above
        self.last_data = False                                      # Is this the last batch of data?
        self.last_time_searched = 0                                 # Last time searched while querying this batch
        self.signals = np.array([], dtype=TriggerSignal.get_dtype())
        self.event_ranges = np.zeros((0, 2), dtype=np.int64)        # Event (left, right) time ranges (in ns)
        self.signals_by_event = []                                  # Signals to save with each event
        self.batch_info_doc = dict()                                # Status info about batch, saved by monitor

        # Deleted in early stages
        self.input_data = dict()        # times, modules, channels, etc. raw from input

        for k, v in kwargs.items():
            setattr(self, k, v)


class TriggerPlugin(object):
    """Base class for trigger plugins"""

    def __init__(self, trigger, config):
        self.trigger = trigger
        self.config = config
        self.name = self.__class__.__name__
        self.log = logging.getLogger(self.name)
        self.log.debug("Logging started for %s" % self.name)
        self.end_of_run_info = {}
        self.startup()

    def startup(self):
        self.log.debug("%s does not define a startup." % self.name)

    def shutdown(self):
        self.log.debug("%s does not define a shutdown." % self.name)


class Trigger(object):
    """XENON1T trigger main class
    This is responsible for
        * constructing a TriggerData object from a batch of pulse start times, their module/channel info, etc.
        * running this by each of the trigger modules
        * collecting the end-of-run info from all plugins into one dictionary
        * managing the trigger monitor stream (insert docs to MongoDB, dump to file, ...)
    """

    def __init__(self, pax_config, trigger_monitor_collection=None):
        """
            trigger_monitor_collection: None or a mongodb collection to write trigger monitor info to
        """
        self.log = logging.getLogger('Trigger')
        self.pax_config = pax_config
        self.config = pax_config['Trigger']

        # Configuration sanity check
        if self.config['event_separation'] < self.config['left_extension'] + self.config['right_extension']:
            raise InvalidConfigurationError("event_separation must not be smaller "
                                            "than left_extension + right_extension")

        ##
        # Initialize trigger monitor stuff
        ##
        self.trigger_monitor_collection = trigger_monitor_collection
        if trigger_monitor_collection is None:
            self.log.info("No trigger monitor collection provided: won't write trigger monitor data to MongoDB")
        self.monitor_cache = []         # Cache of (data_type, doc), with doc document to insert into db / write to zip.
        self.data_type_counter = defaultdict(float)    # Counts how often a document of each data type has been inserted

        # Create a zipfile to store the trigger monitor data, if config says so
        # (the data is additionaly stored in a MongoDB, if trigger_monitor_collection was passed)
        trigger_monitor_file_path = self.config.get('trigger_monitor_file_path', None)
        if trigger_monitor_file_path is not None:
            base_dir = os.path.dirname(trigger_monitor_file_path)
            try:
                os.makedirs(base_dir)
            except OSError as e:
                if e.errno == errno.EEXIST:
                    pass
                else:
                    raise
            self.trigger_monitor_file = zipfile.ZipFile(trigger_monitor_file_path, mode='w')
        else:
            self.log.info("Not trigger monitor file path provided: won't write trigger monitor data to Zipfile")
            self.trigger_monitor_file = None

        self.end_of_run_info = defaultdict(float)
        self.end_of_run_info.update(pulses_read=0,
                                    signals_found=0,
                                    trigger_monitor_data_format_version=2,
                                    events_built=0,
                                    start_timestamp=time.time(),
                                    pax_version=pax.__version__,
                                    config=self.config)

        ##
        # Load the trigger plugins
        ##
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

            # Trigger plugins inherit settings from [Trigger], but overriding is explicitly disallowed
            # This is because trigger settings go into the run doc, and we want to always find a particular setting
            # (e.g. the prescale, whether some dangerous option like skip_ahead was used, etc) in the same position.
            conf = deepcopy(self.config)
            plugin_conf = pax_config.get('Trigger.%s' % tp_name, {})
            for k, v in plugin_conf.items():
                if k in conf:
                    raise ValueError("Config override attempted for option %s, not allowed in trigger" % k)
                else:
                    conf[k] = v

            # Store the plugin-specific settings in the run doc
            self.end_of_run_info['config'][tp_name] = plugin_conf

            # Initialize the plugin with the config
            self.plugins.append(tm_classes[tp_name](trigger=self, config=conf))

        self.previous_last_time_searched = 0

    def run(self, last_time_searched, start_times=tuple(), channels=None, modules=None, areas=None, last_data=False):
        """Run on the specified data, yields ((start time, stop time), signals in event, event type identifier)
        """
        data = TriggerData(last_time_searched=last_time_searched, last_data=last_data)
        data.input_data = dict(start_times=start_times, channels=channels, modules=modules, areas=areas)

        # Hand over to each of the trigger plugins in turn.
        for plugin in self.plugins:
            self.log.debug("Passing data to plugin %s" % plugin.name)
            plugin.process(data)
        self.log.info("Trigger found %d event ranges, %d signals in %d pulses." % (
            len(data.event_ranges), len(data.signals), len(start_times)))

        # Update and save the batch info doc
        pulses_read = len(start_times)
        signals_found = len(data.signals)
        events_built = len(data.event_ranges)
        if events_built:
            total_event_duration = np.sum(data.event_ranges[:, 1] - data.event_ranges[:, 0])
        else:
            total_event_duration = 0
        data.batch_info_doc.update(dict(last_time_searched=data.last_time_searched,
                                        timestamp=time.time(),
                                        pulses_read=pulses_read,
                                        signals_found=signals_found,
                                        events_built=events_built,
                                        total_event_duration=total_event_duration,
                                        batch_duration=last_time_searched - self.previous_last_time_searched,
                                        is_last_data=data.last_data))
        self.save_monitor_data('batch_info', data.batch_info_doc)

        # Update the end of run info
        self.end_of_run_info['pulses_read'] += pulses_read
        self.end_of_run_info['signals_found'] += signals_found
        self.end_of_run_info['events_built'] += events_built
        self.end_of_run_info['total_event_duration'] += total_event_duration

        # Store any documents in the monitor cache to disk / database
        # We don't want to do break the trigger logic every time some plugin calls save_monitor_data, so this happens
        # only at the end of each batch
        if len(self.monitor_cache):
            if self.trigger_monitor_collection is not None:
                self.log.debug("Inserting %d trigger monitor documents into MongoDB" % len(self.monitor_cache))
                result = self.trigger_monitor_collection.insert_many([d for _, d in self.monitor_cache])
                self.log.debug("Inserted docs ids: %s" % result.inserted_ids)
            if self.trigger_monitor_file is not None:
                for data_type, d in self.monitor_cache:
                    self.trigger_monitor_file.writestr("%s=%012d" % (data_type, self.data_type_counter[data_type]),
                                                       zlib.compress(pickle.dumps(d)))
                    self.data_type_counter[data_type] += 1
            self.monitor_cache = []

        # Yield the events to the processor
        for event_i, (start, stop) in enumerate(data.event_ranges):
            if data.signals_by_event:
                yield (start, stop), data.signals_by_event[event_i]
            else:
                yield (start, stop), []

    def shutdown(self):
        """Shut down trigger, return dictionary with end-of-run information."""
        self.log.debug("Shutting down the trigger")
        for p in self.plugins:
            p.shutdown()

        # Close the trigger data file.
        # Note this must be done after shutting down the plugins, they may add something on shutdown as well.
        if self.trigger_monitor_file is not None:
            self.trigger_monitor_file.close()

        # Add end-of-run info for the runs database
        self.end_of_run_info.update(dict(end_trigger_processing_timestamp=time.time()))
        for p in self.plugins:
            self.end_of_run_info[p.name] = p.end_of_run_info

        return self.end_of_run_info

    def save_monitor_data(self, data_type, data, metadata=None):
        """Store trigger monitor data document in cache. It will be written to disk/db at the end of the batch.
          data_type: string indicating what kind of data this is (e.g. count_of_lone_pulses).
          data: either
            a dictionary with things bson.BSON.encode() will not crash on, or
            a numpy array. It will be converted to a list, to ensure it is queryable by the DAQ website.
          metadata: more data. Just convenience so you can pass numpy array as data, then something else as well.
        """
        if isinstance(data, np.ndarray):
            data = {'data': data.tolist()}
        data['data_type'] = data_type
        if metadata is not None:
            data.update(metadata)
        self.monitor_cache.append((data_type, data))

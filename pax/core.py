"""The backbone of pax - the Processor class
"""
import glob
import logging
import six
import itertools
import os
import psutil
import sys
import time
import multiprocessing
import heapq
if six.PY2:
    import imp
else:
    import importlib
try:
    import queue
except ImportError:
    import Queue as queue   # flake8: noqa

from prettytable import PrettyTable     # Timing report
from tqdm import tqdm                   # Progress bar

import pax      # Needed for pax.__version__
from pax.configuration import load_configuration, fix_sections_from_mongo, combine_configs
from pax.exceptions import InvalidConfigurationError
from pax import simulation, utils
from pax.MongoDB_ClientMaker import ClientMaker

# For diagnosing suspected memory leaks, uncomment this code
# and similar code in process_event
# import gc
# import objgraph

# Multiprocess status codes
MP_STATUS = dict(normal=0,
                 shutdown=1,
                 crashing=2,
                 input_in_progress=3,
                 input_done=4,
                 processing_done=5)


class Processor:

    def __init__(self, config_names=(), config_paths=(), config_string=None, config_dict=None, just_testing=False):
        """Setup pax using configuration data from three sources:
          - config_names: List of named configurations to load (sequentially)
          - config_paths: List of config file paths to load (sequentially)
          - config_string: A config string (overrides all config from files)
          - config_dict: A final config ovveride dict: {'section' : {'setting' : 17, ...

        Files from config_paths will be loaded after config_names, and can thus override their settings.
        Configuration files can inherit from others using parent_configuration and parent_configuration_files.
        Each value in the ini files (and the config_string) is eval()'ed in a context with physical unit variables:
            4 * 2 -> 8
            4 * cm**(2)/s -> correctly interpreted as a physical value with units of cm^2/s

        The config_dict's settings are not evaluated.

        If config['pax']['look_for_config_in_runs_db'], will try to connect to the runs db and fetch configuration
        for this particular run. The run id is fetched either by number (config['DEFAULT']['run_number']

        Setting just_testing disables some warnings about not specifying any plugins or plugin groups in the config.
        Use only if, for some reason, you don't want to load a full configuration file.

        .. note::
          Although the log level can be specified in the configuration, it is an application wide
          setting that will not be modified once set. New instances of the Processor class will have
          the same log level as the first, regardless of their configuration.  See #78.
        """
        self.config = load_configuration(config_names, config_paths, config_string, config_dict)
        self.log = self.setup_logging()

        # Do we need to get further configuration from the runs database?
        if self.config['pax'].get('look_for_config_in_runs_db'):
            self.log.debug("Looking for extra configuration in runs db")
            # Connect to MongoDB
            run_collection = ClientMaker(self.config['MongoDB']).get_client('run')['run'].get_collection('runs_new')

            # Get the run document, either by explicitly specified run number, or by the input name
            # The last option is a bit of a hack... if you don't like it, think of some way to always pass
            # the run number explicitly. By the way, let's hope nobody tries to reprocess run 0..
            if self.config.get('DEFAULT', {}).get('run_number', 0) > 0:
                run_number = self.config['DEFAULT']['run_number']
                run_doc = run_collection.find_one({'number': run_number})
                if not run_doc:
                    raise InvalidConfigurationError("Unable to find run number %d!" % run_number)

            elif 'input_name' in self.config['pax']:
                run_name = os.path.basename(self.config['pax']['input_name'])

                if run_name.endswith("_MV"):
                    run_doc = run_collection.find_one({'name': run_name[:-3],
                                                   'detector' : 'muon_veto'})
                else:
                    run_doc = run_collection.find_one({'name': run_name,
                                                   'detector' : 'tpc'})

                if not run_doc:
                    raise InvalidConfigurationError("Unable to find a run named %s!" % run_name)

            else:
                raise InvalidConfigurationError("Cannot get configuration from runs db: give run_number or input_name!")

            self.log.debug("Found run document for run %d, %s" % (run_doc['number'], run_doc['name']))

            # The run doc settings act as (but do not override) config_dict
            mongo_conf = fix_sections_from_mongo(run_doc.get('processor', {}))
            config_dict = combine_configs(mongo_conf, override=config_dict)

            # Add run number and run name to the config_dict
            config_dict.setdefault('DEFAULT', {})
            config_dict['DEFAULT']['run_number'] = run_doc['number']
            config_dict['DEFAULT']['run_name'] = run_doc['name']

            self.config = combine_configs(self.config, override=config_dict)

        pc = self.config['pax']
        self.worker_id = pc.get('_worker_id', 'master')

        if self.worker_id != 'master':
            self.log.debug("I'm worker %s" % self.worker_id)
            # I'm a child processor
            self.multiprocessing = True
            self.status = pc['status']
            self.input_queue = pc['input_queue']
            self.output_queue = pc.get('output_queue', None)
            # Remove multiprocessing objects from config datastructure,
            # so the configuration can still be serialized to JSON later
            for k in ['input_queue', 'output_queue', 'status']:
                pc[k] = None

        else:
            # I'm the main processor
            self.log.info("This is PAX version %s, running with configuration for %s." % (
                pax.__version__, self.config['DEFAULT'].get('tpc_name', 'UNSPECIFIED TPC NAME')))

            n_cpus = pc.get('n_cpus', 1)
            if n_cpus == 'all':
                n_cpus = multiprocessing.cpu_count()
            n_cpus = int(n_cpus)   # On the command line it gets passed as a string, since 'all' is possible
            if n_cpus == 1:
                self.multiprocessing = False
            else:
                # Setup multiprocessing
                self.multiprocessing = True
                self.manager = multiprocessing.Manager()
                self.status = self.manager.Value('i', MP_STATUS['normal'])
                self.input_queue = self.manager.Queue()
                self.output_queue = self.manager.Queue()
                self.last_status_update = time.time()

                # Start worker processes
                self.processing_workers = []
                from copy import deepcopy
                for worker_number in range(n_cpus):
                    c = deepcopy(self.config)
                    c['pax'].update(dict(plugin_group_names=[q for q in pc['plugin_group_names']
                                                             if q not in ('input', 'output')],
                                         input_queue=self.input_queue,
                                         output_queue=self.output_queue,
                                         status=self.status,
                                         _worker_id='processing_%d' % worker_number))
                    self.processing_workers.append(multiprocessing.Process(target=Processor,
                                                                           kwargs=dict(config_dict=c)))

                c = deepcopy(self.config)
                c['pax'].update(dict(plugin_group_names=['output'],
                                     _worker_id='output',
                                     status=self.status,
                                     input_queue=self.output_queue))
                self.output_worker = multiprocessing.Process(target=Processor,
                                                             kwargs=dict(config_dict=c))

                # Start my child processes
                for w in self.processing_workers + [self.output_worker]:
                    w.start()

                # I will just focus on input
                pc['plugin_group_names'] = ['input']

        # Start up the simulator
        # Must be done explicitly here, as plugins can rely on its presence in startup
        if 'WaveformSimulator' in self.config:
            wvsim_config = {}
            wvsim_config.update(self.config['DEFAULT'])
            wvsim_config.update(self.config['WaveformSimulator'])
            self.simulator = simulation.Simulator(wvsim_config)
        elif not just_testing:
                self.log.warning('You did not specify any configuration for the waveform simulator!\n' +
                                 'If you attempt to load the waveform simulator, pax will crash!')

        # Get the list of plugins from the configuration
        # plugin_names is a dict with group names as keys, and the plugins we have to initialize per group as values
        plugin_names = {}
        if 'plugin_group_names' not in pc:
            if not just_testing:
                self.log.warning('You did not specify any plugin groups to load: are you testing me?')
            pc['plugin_group_names'] = []

        if not self.multiprocessing or self.worker_id not in ('master', 'output'):
            # Standalone or child processor (not output)
            # Make plugin group names for the encoder and decoder plugins
            # By having this code here, we ensure they are always just after/before input/output,
            # no matter what plugin group names the user is using
            if pc.get('decoder_plugin') is not None:
                decoder_pos = 0
                if len(pc['plugin_group_names']) and pc['plugin_group_names'][0] == 'input':
                    decoder_pos += 1
                pc['plugin_group_names'].insert(decoder_pos, 'decoder_plugin')

            if pc.get('encoder_plugin') is not None:
                encoder_pos = len(pc['plugin_group_names'])
                if len(pc['plugin_group_names']) and pc['plugin_group_names'][-1] == 'output':
                    encoder_pos -= 1
                pc['plugin_group_names'].insert(encoder_pos, 'encoder_plugin')

        for plugin_group_name in pc['plugin_group_names']:
            if plugin_group_name not in pc:
                raise InvalidConfigurationError('Plugin group list %s missing' % plugin_group_name)

            plugin_names[plugin_group_name] = pc[plugin_group_name]

            if not isinstance(plugin_names[plugin_group_name], (str, list)):
                raise InvalidConfigurationError("Plugin group list %s should be a string, not %s"  % (
                    plugin_group_name, type(plugin_names)))

            if not isinstance(plugin_names[plugin_group_name], list):
                plugin_names[plugin_group_name] = [plugin_names[plugin_group_name]]

            # Ensure each plugin has a configuration
            for plugin_name in plugin_names[plugin_group_name]:
                self.config[plugin_name] = self.config.get(plugin_name, {})

        # Separate input and actions (which for now includes output).
        # For the plugin groups which are action plugins, get all names, flatten them
        action_plugin_names = list(itertools.chain(*[plugin_names[g]
                                                     for g in pc['plugin_group_names']
                                                     if g != 'input']))

        # Hand out input & output override instructions
        if 'input_name' in pc and 'input' in pc['plugin_group_names']:
            self.log.debug('User-defined input override: %s' % pc['input_name'])
            self.config[plugin_names['input'][0]]['input_name'] = pc['input_name']

        if 'output_name' in pc and 'output' in pc['plugin_group_names']:
            self.log.debug('User-defined output override: %s' % pc['output_name'])
            for o in plugin_names['output']:
                self.config[o]['output_name'] = pc['output_name']

        self.plugin_search_paths = self.get_plugin_search_paths(pc.get('plugin_paths', None))

        self.log.debug("Search path for plugins is %s" % str(self.plugin_search_paths))

        # Load input plugin & setup the get_events generator
        if 'input' in pc['plugin_group_names']:
            if len(plugin_names['input']) != 1:
                raise InvalidConfigurationError("There should be one input plugin listed, not %s" %
                                                len(plugin_names['input']))

            self.input_plugin = self.instantiate_plugin(plugin_names['input'][0])
            self.number_of_events = self.input_plugin.number_of_events
            self.stop_after = pc.get('stop_after', float('inf'))

            # Parse the event numbers file, if one is given
            if pc.get('event_numbers_file', None) is not None:
                with open(pc['event_numbers_file'], mode='r') as f:
                    pc['events_to_process'] = [int(line.rstrip()) for line in f]

            if pc.get('events_to_process', None) is not None:
                # The user specified which events to process:
                self.number_of_events = len(pc['events_to_process'])

                def get_events():
                    for event_number in pc['events_to_process']:
                        yield self.input_plugin.get_single_event(event_number)
                self.get_events = get_events
            else:
                # Let the input plugin decide which events to process:
                self.get_events = self.input_plugin.get_events

            self.number_of_events = min(self.number_of_events, self.stop_after)

        else:
            # During multiprocessing or testing there is often no input plugin events are added manually
            self.input_plugin = None
            self.log.debug("No input plugin specified: how are you planning to get any events?")

        # Load the action plugins
        if len(action_plugin_names) > 0:
            self.action_plugins = [self.instantiate_plugin(x) for x in action_plugin_names]

        # During tests of input plugins there is often no action plugin
        else:
            self.action_plugins = []
            self.log.debug("No action plugins specified: this will be a pretty boring processing run...")

        self.timer = utils.Timer()

        # For worker processed, we have to call run from init: nobody else wil...
        self.max_queue_blocks = self.config['pax'].get('max_queue_blocks', 100)
        self.block_size = self.config['pax'].get('event_block_size', 10)
        if self.worker_id != 'master':
            self.run()

    def setup_logging(self):
        """Sets up logging. Must have loaded config first."""

        pc = self.config['pax']

        # Setup logging
        log_spec = pc.get('logging_level', 'INFO').upper()
        numeric_level = getattr(logging, log_spec, None)
        if not isinstance(numeric_level, int):
            raise InvalidConfigurationError('Invalid log level: %s' % log_spec)

        logging.basicConfig(level=numeric_level,
                            format='%(name)s %(processName)-10s L%(lineno)s %(levelname)s %(message)s')

        logger = logging.getLogger('processor')
        logger.debug('Logging initialized with level %s' % log_spec)

        return logger

    @staticmethod
    def get_plugin_search_paths(extra_paths=None):
        """Returns paths where we should search for plugins
        Search for plugins in  ., ./plugins, utils.PAX_DIR/plugins, any directories in config['plugin_paths']
        Search in all subdirs of the above, except for __pycache__ dirs
        """
        plugin_search_paths = ['./plugins', os.path.join(utils.PAX_DIR, 'plugins')]
        if extra_paths is not None:
            plugin_search_paths += extra_paths

        # Look in all subdirectories
        for entry in plugin_search_paths:
            plugin_search_paths.extend(glob.glob(os.path.join(entry, '*/')))

        # Don't look in __pychache__ folders
        plugin_search_paths = [path for path in plugin_search_paths if '__pycache__' not in path]
        return plugin_search_paths

    def instantiate_plugin(self, name):
        """Take plugin class name and build class from it

        The python default module locations are also searched... I think.. so don't name your module 'glob'...
        """

        self.log.debug('Instantiating %s' % name)
        name_module, name_class = name.split('.')

        # Find and load the module which includes the plugin
        if six.PY2:
            file, pathname, description = imp.find_module(name_module, self.plugin_search_paths)
            if file is None:
                raise InvalidConfigurationError('Plugin %s not found.' % name)
            plugin_module = imp.load_module(name_module, file, pathname, description)
        else:
            # imp has been deprecated in favor of importlib.
            # Moreover, the above code gives non-closed file warnings in py3, so although it works,
            # we really don't want to use it.
            spec = importlib.machinery.PathFinder.find_spec(name_module, self.plugin_search_paths)
            if spec is None:
                raise InvalidConfigurationError('Plugin %s not found.' % name)
            plugin_module = spec.loader.load_module()

        this_plugin_config = {}

        this_plugin_config.update(self.config['DEFAULT'])          # First load the default settings
        if name_module in self.config:
            this_plugin_config.update(self.config[name_module])    # Then override with module-level settings
        if name in self.config:
            this_plugin_config.update(self.config[name])           # Then override with plugin-level settings

        # Let each plugin access its own config, and the processor instance as well
        # -- needed to e.g. access self.simulator in the simulator plugins or self.config for dumping the config file
        # TODO: Is this wise? If s there another way?
        instance = getattr(plugin_module, name_class)(this_plugin_config, processor=self)

        self.log.debug('Instantiated %s succesfully' % name)

        return instance

    def get_plugin_by_name(self, name):
        """Return plugin by class name. Use for testing."""
        plugins_by_name = {p.__class__.__name__: p for p in self.action_plugins}
        if self.input_plugin is not None:
            plugins_by_name[self.input_plugin.__class__.__name__] = self.input_plugin
        if name in plugins_by_name:
            return plugins_by_name[name]
        else:
            raise ValueError("No plugin named %s has been initialized." % name)

    def get_metadata(self):
        return dict(run_number=self.config['DEFAULT']['run_number'],
                    tpc=self.config['DEFAULT']['tpc_name'],
                    file_builder_name='pax',
                    file_builder_version=pax.__version__,
                    timestamp=time.time(),
                    configuration=self.config)

    def process_event(self, event):
        """Process one event with all action plugins. Returns processed event."""
        total_plugins = len(self.action_plugins)

        for j, plugin in enumerate(self.action_plugins):
            self.log.debug("%s (step %d/%d)" % (plugin.__class__.__name__, j, total_plugins))
            event = plugin.process_event(event)
            plugin.total_time_taken += self.timer.punch()

        # Uncomment to diagnose memory leaks
        # gc.collect()  # don't care about stuff that would be garbage collected properly
        # objgraph.show_growth(limit=5)
        return event

    def run(self, clean_shutdown=True):
        """Run the processor over all events, then shuts down the plugins (unless clean_shutdown=False)

        If clean_shutdown=False, will not shutdown plugin classes
            (they still shut down if the Processor class is deleted)
            Use only if for some arcane reason you want to run a single instance more than once.
            If you do, you get in trouble if you start a new Processor instance that tries to write to the same files.

        """
        if self.worker_id != 'master':
            # I'm a child processor
            self.check_crash()
            if self.worker_id == 'output':
                block_id = -1
                block_heap = []

            while True:
                # Check if we can end before we fetch event blocks:
                # the last block may get added while we are fetching events
                can_end = self.status.value == MP_STATUS['processing_done' if self.worker_id == 'output'
                                                         else 'input_done']

                if self.worker_id == 'output':
                    # The output worker has an additional complication: blocks must be written in proper order
                    self.check_crash()

                    try:
                        # If we don't have the block we want yet, keep fetching event blocks from the queue .
                        # Note: if one block takes much longer than the others, this would slurp all blocks
                        # while waiting for the difficult one to come through, potentially exhausting your RAM...
                        while not (len(block_heap) and block_heap[0][0] == block_id + 1):
                            self.check_crash()
                            heapq.heappush(block_heap, self.input_queue.get(block=True, timeout=1))
                            self.log.debug("Output just got a block, heap is now %d blocks long" % len(block_heap))
                            self.log.debug("Earliest block: %d, looking for block %s" % (block_heap[0][0],
                                                                                         block_id + 1))

                    except queue.Empty:
                        if can_end and not len(block_heap):
                            # We're done!
                            break
                        # Queue is empty, but either we're not in the processing_done status, or we're waiting for a
                        # block which hasn't arrived on the queue yet. Wait, then try the queue again.
                        time.sleep(1)
                        continue

                    # Pop the next event block from the heap
                    block_id, event_block = heapq.heappop(block_heap)

                else:
                    # Ordinary worker
                    self.check_crash()
                    try:
                        block_id, event_block = self.input_queue.get(block=True, timeout=1)
                    except queue.Empty:
                        if can_end:
                            # We're done!
                            break
                        # Queue is empty: wait for more data, then check can_end again
                        self.log.debug("%s found empty input queue, can_end is %s, status is %s" % (
                            self.worker_id, can_end, self.status.value))
                        time.sleep(1)
                        continue

                try:
                    self.log.debug("%s now processing block %d" % (self.worker_id, block_id))
                    for i, event in enumerate(event_block):
                        self.check_crash()
                        event_block[i] = self.process_event(event)
                except Exception:
                    # Crash occurred during processing: notify everyone else, then die
                    self.status.value = MP_STATUS['crashing']
                    raise

                if self.worker_id != 'output':
                    # Push the result to the output queue
                    self.output_queue.put((block_id, event_block))
                    # If the output worker has trouble catching up, sleep for a bit
                    while self.output_queue.qsize() >= self.max_queue_blocks:
                        self.check_crash()
                        time.sleep(1)

        else:
            # I'm a master or standalone processor
            if self.input_plugin is None:
                # You're allowed to specify no input plugin, which is useful for testing. (You may want to feed events
                # in by hand). If you do this, you can't use the run method. In case somebody ever tries:
                raise InvalidConfigurationError("You just tried to run a Processor without specifying input plugin.")

            if self.input_plugin.has_shut_down:
                raise RuntimeError("Attempt to run a Processor twice!")

            if self.multiprocessing:
                # I'm a master processor
                block_id = 0
                event_block = []
                for i, event in enumerate(self.get_events()):
                    event_block.append(event)
                    self.master_heartbeat()
                    if len(event_block) >= self.block_size:
                        self.log.debug("Created event block %d" % block_id)
                        self.input_queue.put((block_id, event_block))
                        block_id += 1
                        event_block = []
                    # If the processing workers have trouble catching up, sleep for a bit
                    while self.input_queue.qsize() >= self.max_queue_blocks:
                        self.master_heartbeat()
                        time.sleep(1)
                    if i >= self.stop_after:
                        self.log.info("Read in user-defined limit of %d events." % i)
                        break
                self.input_queue.put((block_id, event_block))
                self.master_heartbeat()
                self.status.value = MP_STATUS['input_done']

                # Wait for child processes to die or crash
                # Note we never use join/wait -- so we never wait for something that may not happen
                while True:
                    self.master_heartbeat()
                    if all([not w.is_alive() for w in self.processing_workers]):
                        self.status.value = MP_STATUS['processing_done']
                    if not self.output_worker.is_alive():
                        break
                    time.sleep(1)
                self.log.info("Pax is done, goodbye!")

            else:
                # I'm a standalone processor
                i = 0  # in case loop does not run
                self.timer.punch()
                for i, event in enumerate(tqdm(self.get_events(),
                                               desc='Event',
                                               total=self.number_of_events)):
                    self.input_plugin.total_time_taken += self.timer.punch()
                    if i >= self.stop_after:
                        self.log.info("User-defined limit of %d events reached." % i)
                        break
                    self.process_event(event)
                    self.log.debug("Event %d (%d processed)" % (event.event_number, i))
                else:   # If no break occurred:
                    self.log.info("All events from input source have been processed.")

                if self.config['pax']['print_timing_report']:
                    self.make_timing_report(i + 1)

        # Shutdown all plugins now -- don't wait until this Processor instance gets deleted
        if clean_shutdown:
            self.shutdown()

    def master_heartbeat(self):
        self.check_crash()
        self.update_status()

    def check_crash(self):
        if self.status.value == MP_STATUS['crashing']:
            if self.worker_id == 'master':
                self.log.fatal("Crash detected, giving worker processes five seconds to die in peace")
                time.sleep(5)
                self.log.fatal("That's it, farewell cruel world!")
                raise RuntimeError("Terminated pax multiprocessing due to crash in one of the workers.")
            exit('')

    def update_status(self):
        # Don't update the status too often: this is expensive
        if time.time() <= self.last_status_update + 1:
            return
        self.last_status_update = time.time()

        def get_mem_usage(pid):
            """Return memory usage in MB for process with PID pid.
            Returns 0 if process does not exist (anymore).
            Maintains a cache to make sure this is not polled more than once per second
            """
            try:
                return psutil.Process(pid).memory_info().rss / 1e6
            except psutil.NoSuchProcess:
                return 0

        sys.stdout.write('\rStatus: %s. Processing queue: %d events. Output queue: %s events. '
                         'RAM usage: %0.1f (master) %0.1f (workers) %0.1f (output)' % (
            [k for k, v in MP_STATUS.items() if v == self.status.value][0],
            self.input_queue.qsize() * self.block_size,
            self.output_queue.qsize() * self.block_size,
            get_mem_usage(os.getpid()),
            sum([get_mem_usage(worker.pid) if worker.pid is not None else 0
                 for worker in self.processing_workers]),
            (get_mem_usage(self.output_worker.pid) if self.output_worker.pid is not None else 0),
        ))
        sys.stdout.flush()

    def make_timing_report(self, events_actually_processed):
        all_plugins = [self.input_plugin] + self.action_plugins
        timing_report = PrettyTable(['Plugin',
                                     '%',
                                     '/event (ms)',
                                     '#/s',
                                     'Total (s)'])
        timing_report.align = "r"
        timing_report.align["Plugin"] = "l"
        total_time = sum([plugin.total_time_taken for plugin in all_plugins])

        for plugin in all_plugins:
            t = plugin.total_time_taken

            if t > 0:
                time_per_event_ms = round(t / events_actually_processed, 1)

                event_rate_hz = round(1000 * events_actually_processed / t, 1)
                if event_rate_hz > 100:
                    event_rate_hz = ''

                timing_report.add_row([plugin.__class__.__name__,
                                       round(100 * t / total_time, 1),
                                       time_per_event_ms,
                                       event_rate_hz,
                                       round(t / 1000, 1)])
            else:
                timing_report.add_row([plugin.__class__.__name__,
                                       0,
                                       0,
                                       'n/a',
                                       round(t / 1000, 1)])

        if total_time > 0:
            timing_report.add_row(['TOTAL',
                                   round(100., 1),
                                   round(total_time / events_actually_processed, 1),
                                   round(1000 * events_actually_processed / total_time, 1),
                                   round(total_time / 1000, 1)])
        else:
            timing_report.add_row(['TOTAL',
                                   round(100., 1),
                                   0,
                                   'n/a',
                                   round(total_time / 1000, 1)])
        self.log.info("Timing report:\n" + str(timing_report))

    def shutdown(self):
        """Call shutdown on all plugins"""
        self.log.debug("Shutting down all plugins...")
        if self.input_plugin is not None:
            self.log.debug("Shutting down %s..." % self.input_plugin.name)
            self.input_plugin.shutdown()
            self.input_plugin.has_shut_down = True
        for ap in self.action_plugins:
            self.log.debug("Shutting down %s..." % ap.name)
            ap.shutdown()
            ap.has_shut_down = True

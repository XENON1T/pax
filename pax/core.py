"""The backbone of pax

"""
import glob
import itertools
import logging
import inspect
from configparser import ConfigParser, ExtendedInterpolation

import re
import os
from io import StringIO
import importlib
from tqdm import tqdm     # Progress bar
from prettytable import PrettyTable     # Timing report

import pax
from pax import units

# Store the directory of pax (i.e. this file's directory) as PAX_DIR
PAX_DIR = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))


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
    """ Return the names of all named configurations
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


##
# Processor class
##
class Processor:

    fallback_configuration = 'XENON100'    # Configuration to use when none is specified

    def __init__(self, config_names=(), config_paths=(), config_string=None, config_dict={}, just_testing=False):
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

        :return: nested dictionary of evaluated configuration values, use as: config[section][key].

        Setting just_testing disables some warnings about not specifying any plugins or plugin groups in the config.
        Use only if, for some reason, you don't want to load a full configuration file.
        """
        self.config_files_read = []
        self.configp = None    # load_configuration will use this to store the ConfigParser
        self.config = self.load_configuration(config_names, config_paths, config_string, config_dict)
        self.log = self.setup_logging()
        pc = self.config['pax']

        self.log.info("This is PAX version %s, running with configuration for %s." % (
            pax.__version__, self.config['DEFAULT'].get('tpc_name', 'UNSPECIFIED TPC NAME')))

        # Get the list of plugins from the configuration file
        # plugin_names[group] is a list of all plugins we have to initialize in the group 'group'
        plugin_names = {}
        if not 'plugin_group_names' in pc:
            if not just_testing:
                self.log.warning('You did not specify any plugin groups to load: are you testing me?')
            pc['plugin_group_names'] = []

        for plugin_group_name in pc['plugin_group_names']:

            if not plugin_group_name in pc:
                raise ValueError('Invalid configuration: plugin group list %s missing' % plugin_group_name)

            plugin_names[plugin_group_name] = pc[plugin_group_name]

            if not isinstance(plugin_names[plugin_group_name], (str, list)):
                raise ValueError("Invalid configuration: plugin group list %s should be a string, not %s" %
                                 (plugin_group_name, type(plugin_names)))

            if not isinstance(plugin_names[plugin_group_name], list):
                plugin_names[plugin_group_name] = [plugin_names[plugin_group_name]]

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
            # Hmmz, there can be several output plugins, some don't have a configuration...
            for o in plugin_names['output']:
                if not o in self.config:
                    self.config[o] = {}
                self.config[o]['output_name'] = pc['output_name']

        self.plugin_search_paths = self.get_plugin_search_paths(pc.get('plugin_paths', None))

        self.log.debug("Search path for plugins is %s" % str(self.plugin_search_paths))

        # Load input plugin & setup the get_events generator
        if 'input' in pc['plugin_group_names']:

            if len(plugin_names['input']) != 1:
                raise ValueError("Invalid configuration: there should be one input plugin listed, not %s" %
                                 len(plugin_names['input']))

            self.input_plugin = self.instantiate_plugin(plugin_names['input'][0])

            self.stop_after = float('inf')
            if 'stop_after' in pc and pc['stop_after'] is not None:
                self.stop_after = pc['stop_after']

            self.total_number_events = min(self.input_plugin.number_of_events, self.stop_after)

            # How should the events be generated?
            if 'events_to_process' in pc and pc['events_to_process'] is not None:
                # The user specified which events to process:
                self.total_number_events = min(len(pc['events_to_process']), self.stop_after)

                def get_events():
                    for event_number in pc['events_to_process']:
                        yield self.input_plugin.get_single_event(event_number)

            else:
                # Let the input plugin decide which events to process:
                get_events = self.input_plugin.get_events

            self.get_events = get_events

        # During tests there is often no input plugin, events are added manually
        else:
            self.input_plugin = None
            if not just_testing:
                self.log.warning("No input plugin specified: how are you planning to get any events?")

        # Load the action plugins
        if len(action_plugin_names) > 0:
            self.action_plugins = [self.instantiate_plugin(x) for x in action_plugin_names]

        # During tests of input plugins there is often no action plugin
        else:
            self.action_plugins = []
            if not just_testing:
                self.log.warning("No action plugins specified: this will be a pretty boring processing run...")

    def load_configuration(self, config_names, config_paths, config_string, config_dict):
        """Load a configuration -- see init's docstring"""
        self.config_files_read = []      # Need to clean this here so tests can re-load the config

        # Support for string arguments
        if isinstance(config_names, str):
            config_names = [config_names]
        if isinstance(config_paths, str):
            config_paths = [config_paths]

        self.configp = ConfigParser(inline_comment_prefixes='#',
                                    interpolation=ExtendedInterpolation(),
                                    strict=True)

        # Allow for case-sensitive configuration keys
        self.configp.optionxform = str

        # Make a list of all config paths / file objects to load
        config_files = []
        for config_name in config_names:
            config_files.append(os.path.join(PAX_DIR, 'config', config_name + '.ini'))
        for config_path in config_paths:
            config_files.append(config_path)
        if config_string is not None:
            config_files.append(StringIO(config_string))
        if len(config_files) == 0 and config_dict == {}:
            # Load the fallback configuration
            # Have to use print, logging is not yet setup...
            print("WARNING: no configuration specified: loading %s config!" % self.fallback_configuration)
            config_files.append(os.path.join(PAX_DIR, 'config', self.fallback_configuration + '.ini'))

        # Loads the files into configparser, also takes care of inheritance.
        for config_file_thing in config_files:
            self._load_file_into_configparser(config_file_thing)

        # Get a dict with all variables from the units submodule
        units_variables = {name: getattr(units, name) for name in dir(units)}
        # Evaluate the values in the ini file
        evaled_config = {}
        for section_name, section_dict in self.configp.items():
            evaled_config[section_name] = {}
            for key, value in section_dict.items():
                # Eval value in a context where all units are defined
                evaled_config[section_name][key] = eval(value, units_variables)

        # Apply the config_dict
        for section_name in config_dict.keys():
            if section_name in evaled_config:
                evaled_config[section_name].update(config_dict[section_name])
            else:
                evaled_config[section_name] = config_dict[section_name]

        return evaled_config

    def _load_file_into_configparser(self, config_file):
        """Loads a configuration file into our config parser, with support for inheritance.

        :param config_file: path or file object of configuration file to read
        :return: None
        """

        # This code has some commented print statements, because it can't yet use logging:
        # The loglevel is specified in the configuration, which isn't loaded at this point

        if isinstance(config_file, str):
            #print("Loading %s" % config_file)
            if not os.path.isfile(config_file):
                raise ValueError("Configuration file %s does not exist!" % config_file)
            if config_file in self.config_files_read:
                # This file has already been loaded: don't load it again
                # If we did, it would cause problems with inheritance diamonds
                #print("Skipping config file %s: don't load it a second time" % config_file)
                return
            self.configp.read(config_file)
            self.config_files_read.append(config_file)
        else:
            #print("Loading config from file object")
            self.configp.read_file(config_file)

        # Determine the path(s) of the parent config file(s)
        parent_file_paths = []

        if 'parent_configuration' in self.configp['pax']:
            # This file inherits from other config file(s) in the 'config' directory
            parent_files = eval(self.configp['pax']['parent_configuration'])
            if not isinstance(parent_files, list):
                parent_files = [parent_files]
            parent_file_paths.extend([
                os.path.join(PAX_DIR, 'config', pf + '.ini')
                for pf in parent_files])

        if 'parent_configuration_file' in self.configp['pax']:
            # This file inherits from user-defined config file(s)
            parent_files = eval(self.configp['pax']['parent_configuration_file'])
            if not isinstance(parent_files, list):
                parent_files = [parent_files]
            parent_file_paths.extend(parent_files)

        if len(parent_file_paths) == 0:
            # This file has no parents...
            return

        # Unfortunately, configparser can only override settings, not set missing ones.
        # We have no choice but to load the parent file(s), then reload the original one again.
        # By doing this in a recursing function, multi-level inheritance is supported.
        for pfp in parent_file_paths:
            self._load_file_into_configparser(pfp)
        #print("Reloading %s for override" % config_file)
        if isinstance(config_file, str):
            self.configp.read(config_file)
        else:
            self.configp.read_file(config_file)

    def setup_logging(self):
        """Sets up logging. Must have loaded config first."""
        pc = self.config['pax']
        # Setup logging
        try:
            log_spec = pc['logging_level'].upper()
        except KeyError:
            log_spec = 'INFO'
        numeric_level = getattr(logging, log_spec, None)
        if not isinstance(numeric_level, int):
            raise ValueError('Invalid log level: %s' % log_spec)

        logging.basicConfig(level=numeric_level,
                            format='%(name)s L%(lineno)s %(levelname)s %(message)s')
        return logging.getLogger('processor')

    @staticmethod
    def get_plugin_search_paths(extra_paths=None):
        """Returns paths where we should search for plugins
        Search for plugins in  ./plugins, PAX_DIR/plugins, any directories in config['plugin_paths']
        Search in all subdirs of the above, except for __pycache__ dirs
        """
        plugin_search_paths = ['./plugins', os.path.join(PAX_DIR, 'plugins')]
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
        # The traditional and easy way to do this (with imp) has been deprecated... for some reason...
        # importlib also recently deprecated its 'loader' API in favor of some new 'spec' API... for some reason...
        # There is no easy example of this in the python docs, hopefully this will change soon.
        spec = importlib.machinery.PathFinder.find_spec(name_module, self.plugin_search_paths)
        if spec is None:
            raise ValueError('Invalid configuration: plugin %s not found.' % name)
        plugin_module = spec.loader.load_module()

        # First load the default settings
        this_plugin_config = self.config['DEFAULT']
        # Then override with module-level settings
        if name_module in self.config:
            this_plugin_config.update(self.config[name_module])
        # Then override with plugin-level settings
        if name in self.config:
            this_plugin_config.update(self.config[name])

        instance = getattr(plugin_module, name_class)(this_plugin_config)

        self.log.debug('Instantiated %s succesfully' % name)

        return instance

    def get_plugin_by_name(self, name):
        """Return plugin by class name. Use for testing."""
        plugins_by_name = {p.__class__.__name__: p for p in self.action_plugins}
        if self.input_plugin is not None:
            plugins_by_name[self.input_plugin.__name__] = self.input_plugin
        if name in plugins_by_name:
            return plugins_by_name[name]
        else:
            raise ValueError("No plugin named %s has been initialized." % name)

    def process_event(self, event):
        """Process one event with all action plugins. Returns processed event."""
        total_plugins = len(self.action_plugins)

        for j, plugin in enumerate(self.action_plugins):
            self.log.debug("%s (step %d/%d)" % (plugin.__class__.__name__, j, total_plugins))
            event = plugin.process_event(event)

        return event

    def run(self):
        """Run the processor over all events
        """

        # This is the actual event loop.  'tqdm' is a progress bar.
        for i, event in enumerate(tqdm(self.get_events(),
                                       desc='Event',
                                       total=self.total_number_events)):
            if i >= self.stop_after:
                self.log.info("User-defined limit of %d events reached." % i)
                break

            _ = self.process_event(event)

            self.log.debug("Event %d (%d processed)" % (event.event_number, i))

        else:   # If no break occurred:
            self.log.info("All events from input source have been processed.")

        if self.config['pax']['print_timing_report']:
            all_plugins = [self.input_plugin] + self.action_plugins
            timing_report = PrettyTable(['Plugin', '%', 'Per event (ms)', 'Total (s)'])
            timing_report.align = "r"
            timing_report.align["Plugin"] = "l"
            total_time = sum([plugin.total_time_taken for plugin in all_plugins])
            for plugin in all_plugins:
                t = plugin.total_time_taken
                timing_report.add_row([
                    plugin.__class__.__name__,
                    round(100*t/total_time, 1),
                    round(t/self.total_number_events, 1),
                    round(t/1000, 1)])
            timing_report.add_row([
                'TOTAL',
                round(100., 1),
                round(total_time/self.total_number_events, 1),
                round(total_time/1000, 1)])
            self.log.info("Timing report:\n"+str(timing_report))
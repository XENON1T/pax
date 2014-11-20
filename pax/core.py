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
from tqdm import tqdm # Progress bar

import pax
from pax import units


FALLBACK_CONFIGURATION = 'XENON100' # Configuration to use when none is specified


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


##
# Configuration handling
##
CONFIG_FILES_READ = []

def get_named_configuration_options():
    """ Return the names of all named configurations
    """
    config_files =[]
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

def init_configuration(config_names=(), config_paths=(), config_string=None):
    """ Get pax configuration from three configuration sources:
      - config_names: List of named configurations to load (sequentially)
      - config_paths: List of config file paths to load (sequentially)
      - config_string: A final config override string
    Files from config_paths will be loaded after config_names, and can thus override their settings.
    The config_string is loaded last of all.

    Configuration files can inherit from others using parent_configuration and parent_configuration_files.

    Each value in the ini file is eval()'ed in a context with physical unit variables:
        4 * 2 -> 8
        4 * cm**(2)/s -> correctly interpreted as a physical value with units of cm^2/s

    :return: nested dictionary of evaluated configuration values, use as: config[section][key].
    Will return None if no configuration sources are specified at all.
    """
    CONFIG_FILES_READ = []      # Need to clean this here so tests can re-load the config

    # Support for string arguments
    if isinstance(config_names, str):
        config_names = [config_names]
    if isinstance(config_paths, str):
        config_paths = [config_paths]


    config = ConfigParser(inline_comment_prefixes='#',
                          interpolation=ExtendedInterpolation(),
                          strict=True)

    # Allow for case-sensitive configuration keys
    config.optionxform = str

    # Make a list of all config paths / file objects to load
    config_files = []
    for config_name in config_names:
        config_files.append(os.path.join(PAX_DIR, 'config', config_name + '.ini'))
    for config_path in config_paths:
        config_files.append(config_path)
    if config_string is not None:
        config_files.append(StringIO(config_string))
    if len(config_files) == 0:
        return None # The processor will load a default

    # Loads the files into configparser, also takes care of inheritance.
    for config_file_thing in config_files:
        load_file_into_configparser(config, config_file_thing)


    # Get a dict with all variables from the units submodule
    units_variables = {name: getattr(units, name) for name in dir(units)}
    # Evaluate the values in the ini file
    evaled_config = {}
    for section_name, section_dict in config.items():
        evaled_config[section_name] = {}
        for key, value in section_dict.items():
            # Eval value in a context where all units are defined
            evaled_config[section_name][key] = eval(value, units_variables)

    return evaled_config


def load_file_into_configparser(config, config_file):
    """Loads a configuration file into our config parser, with support for inheritance.

    :param config: configparser instance
    :param config_file: path or file object of configuration file to read
    :return: None
    """

    # This code has some commented print statements, because it can't yet use logging:
    # The loglevel is specified in the configuration, which isn't loaded at this point

    if isinstance(config_file, str):
        #print("Loading %s" % config_file)
        if not os.path.isfile(config_file):
            raise ValueError("Configuration file %s does not exist!" % config_file)
        if config_file in CONFIG_FILES_READ:
            # This file has already been loaded: don't load it again
            # If we did, it would cause problems with inheritance diamonds
            #print("Skipping config file %s: don't load it a second time" % config_file)
            return
        config.read(config_file)
        CONFIG_FILES_READ.append(config_file)
    else:
        #print("Loading config from file object")
        config.read_file(config_file)

    # Determine the path(s) of the parent config file(s)
    parent_file_paths = []
    if 'parent_configuration' in config['pax']:
        # This file inherits from other config file(s) in the 'config' directory
        parent_files = eval(config['pax']['parent_configuration'])
        if not isinstance(parent_files, list):
            parent_files = [parent_files]
        parent_file_paths.extend([
            os.path.join(PAX_DIR, 'config', pf + '.ini')
            for pf in parent_files
        ])
    if 'parent_configuration_file' in config['pax']:
        # This file inherits from user-defined config file(s)
        parent_files = eval(config['pax']['parent_configuration_file'])
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
        load_file_into_configparser(config, pfp)
    #print("Reloading %s for override" % config_file)
    if isinstance(config_file, str):
        config.read(config_file)
    else:
        config.read_file(config_file)


##
# Plugin handling
##

def make_plugin_search_paths(config=None):

    plugin_search_paths = ['./plugins', os.path.join(PAX_DIR, 'plugins')]

    if config is not None:
        plugin_search_paths += config['pax']['plugin_paths']

    # Look in all subdirectories
    for entry in plugin_search_paths:
        plugin_search_paths.extend(glob.glob(os.path.join(entry, '*/')))

    # Don't look in __pychache__ folders
    plugin_search_paths = [path for path in plugin_search_paths if '__pycache__' not in path]

    return plugin_search_paths


def instantiate_plugin(name, plugin_search_paths=None, config=None, log=logging, for_testing=False):
    """Take plugin class name and build class from it

    The python default module locations are also searched... I think.. so don't name your module 'glob'...
    """

    # Shortcut for tests
    if for_testing:
        if plugin_search_paths is None:
            plugin_search_paths = make_plugin_search_paths(None)
        if config is None:
            config = init_configuration(config_names=FALLBACK_CONFIGURATION)
    else:
        assert config is not None and plugin_search_paths is not None


    log.debug('Instantiating %s' % name)
    name_module, name_class = name.split('.')

    # Find and load the module which includes the plugin
    # The traditional and easy way to do this (with imp) has been deprecated... for some reason...
    # importlib also recently deprecated its 'loader' API in favor of some new 'spec' API... for some reason...
    # There is no easy example of this in the python docs, hopefully this will change soon.
    spec = importlib.machinery.PathFinder.find_spec(name_module, plugin_search_paths)
    if spec is None:
        raise ValueError('Invalid configuration: plugin %s not found.' % name)
    plugin_module = spec.loader.load_module()

    # First load the default settings
    this_plugin_config = config['DEFAULT']
    # Then override with module-level settings
    if name_module in config:
        this_plugin_config.update(config[name_module])
    # Then override with plugin-level settings
    if name in config:
        this_plugin_config.update(config[name])

    instance = getattr(plugin_module, name_class)(this_plugin_config)

    log.debug('Instantiated %s succesfully' % name)

    return instance


##
# Event processing
##

def processor(config):
    """Run the processor according to the configuration dictionary

    :param config: dictionary with configuration values
    :return: None
    """

    # Setup logging
    log_spec = config['pax']['logging_level'].upper()
    numeric_level = getattr(logging, log_spec, None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_spec)

    logging.basicConfig(level=numeric_level,
                        format='%(name)s L%(lineno)s %(levelname)s %(message)s')
    log = logging.getLogger('processor')


    # Complain if we didn't get a configuration, then load a default
    if config is None:
        log.warning("No configuration specified: loading %s config!" % FALLBACK_CONFIGURATION)
        config = init_configuration(config_names=[FALLBACK_CONFIGURATION])
    log.info("This is PAX version %s, running with configuration for %s." % (
        pax.__version__, config['DEFAULT']['tpc_name'])
    )


    # Get the list of plugins from the configuration file
    plugin_groups = ('input', 'dsp', 'transform', 'my_postprocessing', 'output')
    # plugin_names[group] is a list of all plugins we have to initialize in the group 'group'
    plugin_names = {}

    for plugin_group_name in plugin_groups:

        if not plugin_group_name in config['pax']:
            raise ValueError('Invalid configuration: plugin group list %s missing' % plugin_group_name)

        plugin_names[plugin_group_name] = config['pax'][plugin_group_name]

        if not isinstance(plugin_names[plugin_group_name], (str, list)):
            raise ValueError("Invalid configuration: plugin group list %s should be a string, not %s" %
                             (plugin_group_name, type(plugin_names)))

        if not isinstance(plugin_names[plugin_group_name], list):
            plugin_names[plugin_group_name] = [plugin_names[plugin_group_name]]

    if len(plugin_names['input']) != 1:
        raise ValueError("Invalid configuration: there should be one input plugin listed, not %s" %
                         len(plugin_names['input']))


    # Separate input and actions (which for now includes output).
    input_plugin_name = plugin_names['input'][0]

    # For the plugin groups which are action plugins, get all names, flatten them
    action_plugin_names = itertools.chain(*[plugin_names[g]
                                            for g in plugin_groups
                                            if g != 'input'])

    # Hand out input & output override instructions
    if 'input_name' in config['pax']:
        log.debug('User-defined input override: %s' % config['pax']['input_name'])
        config[input_plugin_name]['input_name'] = config['pax']['input_name']

    if 'output_name' in config['pax']:
        log.debug('User-defined output override: %s' % config['pax']['output_name'])
        # Hmmz, there can be several output plugins, some don't have a configuration...
        for o in plugin_names['output']:
            if not o in config:
                config[o] = {}
            config[o]['output_name'] = config['pax']['output_name']

    # Construct the plugin search path
    plugin_search_paths = make_plugin_search_paths(config)
    log.debug("Search path for plugins is %s" % str(plugin_search_paths))

    # Load all the plugins
    input_plugin =    instantiate_plugin(input_plugin_name, plugin_search_paths,
                                         config, log)
    action_plugins = [instantiate_plugin(x, plugin_search_paths, config,
                                         log) for x in action_plugin_names]

    total_number_events = input_plugin.number_events()

    # How should the events be generated?
    if 'events_to_process' in config['pax'] and \
                    config['pax']['events_to_process'] is not None:
        # The user specified which events to process:
        total_number_events = len(config['pax']['events_to_process'])
        def get_events():
            for event_number in config['pax']['events_to_process']:
                yield input_plugin.get_single_event(event_number)
    else:
        # Let the input plugin decide which events to process:
        get_events = input_plugin.get_events

    # This is the actual event loop.  'tqdm' is a progress bar.
    for i, event in enumerate(tqdm(get_events(),
                                   desc='Event',
                                   total=total_number_events)):
        if 'stop_after' in config['pax'] and \
                        config['pax']['stop_after'] is not None:
                if i >= config['pax']['stop_after']:
                    log.info("User-defined limit of %d events reached." % i)
                    break

        log.debug("Event %d (%d processed)" % (event.event_number, i))

        for j, plugin in enumerate(action_plugins):
            log.debug("Step %d with %s", j, plugin.__class__.__name__)
            event = plugin.process_event(event)

    else:
        log.info("All events from input source have been processed.")
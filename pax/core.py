"""The backbone of pax

"""
import logging
import inspect
from configparser import ConfigParser, ExtendedInterpolation
import glob

import re
import os
from io import StringIO
from pluginbase import PluginBase

import pax
from pax import units


# Store the directory of pax (i.e. this file's directory) as pax_dir
pax_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
def data_file_name(filename):
    """Returns filename if a file exists there, else returns pax_dir/data/filename"""
    if os.path.exists(filename):
        return filename
    new_filename = os.path.join(pax_dir, 'data', filename)
    if os.path.exists(new_filename):
        return new_filename
    else:
        raise ValueError('File name or path %s not found!' % filename)

##
# Configuration handling
##
global config_files_read
config_files_read = []

def get_named_configuration_options():
    """ Return the names of all named configurations
    """
    config_files =[]
    for filename in glob.glob(os.path.join(pax_dir, 'config', '*.ini')):
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
    global config_files_read
    config_files_read = []      # Need to clean this here so tests can re-load the config

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
        config_files.append(os.path.join(pax_dir, 'config', config_name + '.ini'))
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
    if isinstance(config_file, str):
        # print("Loading %s" % config_file)
        if not os.path.isfile(config_file):
            raise ValueError("Configuration file %s does not exist!" % config_file)
        global config_files_read
        if config_file in config_files_read:
            # This file has already been loaded: don't load it again
            # If we did, it would cause problems with inheritance diamonds
            # print("Skipping config file %s: don't load it a second time" % config_file)
            return
        config.read(config_file)
        config_files_read.append(config_file)
    else:
        # print("Loading config from file object")
        config.read_file(config_file)

    # Determine the path(s) of the parent config file(s)
    parent_file_paths = []
    if 'parent_configuration' in config['pax']:
        # This file inherits from other config file(s) in the 'config' directory
        global pax_dir
        parent_files = eval(config['pax']['parent_configuration'])
        if not isinstance(parent_files, list):
            parent_files = [parent_files]
        parent_file_paths.extend([
            os.path.join(pax_dir, 'config', pf + '.ini')
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
    if isinstance(config_file, str):
        config.read(config_file)
    else:
        config.read_file(config_file)


##
# Plugin handling
##

def instantiate_plugin(name, plugin_source, config, log=logging):
    """Take plugin class name and build class from it"""
    log.debug('Instantiating %s' % name)
    name_module, name_class = name.split('.')
    try:
        plugin_module = plugin_source.load_plugin(name_module)
    except ImportError as e:
        log.fatal("Failed to load plugin %s" % name_module)
        log.exception(e)
        raise

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


def get_plugin_source(config, log=logging, ident='blah'):
    # Setup plugins (which involves finding the plugin directory).
    plugin_base = PluginBase(package='pax.plugins')
    searchpath = ['./plugins'] + config['pax']['plugin_paths']
    # Find the absolute path, then director, then find plugin directory
    global pax_dir
    searchpath += [os.path.join(pax_dir, 'plugins')]
    # Want to look in all subdirectories of plugin directories
    for entry in searchpath:
        searchpath.extend(glob.glob(os.path.join(entry, '*/')))

    searchpath = [path for path in searchpath if '__pycache__' not in path]

    log.debug("Search path for plugins is %s" % str(searchpath))
    plugin_source = plugin_base.make_plugin_source(searchpath=searchpath,
                                                   identifier=ident)
    log.debug("Found the following plugins:")
    for plugin_name in plugin_source.list_plugins():
        log.debug("\tFound %s" % plugin_name)

    return plugin_source


def get_actions(config, input, list_of_names, output):
    """Get the class names associated with action

    An action is either input, transform, postprocessing, or output.  This grabs
    all the classes that need to be instantiated and used for processing events.

    Note that output is a list.
    """
    config = config['pax']

    input = config[input]
    if not isinstance(input, str):
        raise ValueError("Input class name must be string")

    actions = []

    for name in list_of_names:
        value = config[name]

        if not isinstance(value, (str, list)):
            raise ValueError("Misformatted ini file on key %s" % name)

        if not isinstance(value, list):
            value = [value]

        actions += value

    if not isinstance(output, (str, list)):
        raise ValueError("Misformatted ini file on key %s" % output)

    output = config[output]

    if not isinstance(output, list):
        output = [output]

    return input, actions, output


##
# Event processing
##

def process_single_event(actions, event, log):
    for j, block in enumerate(actions):
        log.debug("Step %d with %s", j, block.__class__.__name__)
        event = block.process_event(event)


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
        log.warning("No configuration specified: loading Xenon100 config!")
        config = init_configuration(config_names=['XENON100'])
    log.info("This is PAX version %s, running with configuration for %s." % (
        pax.__version__, config['DEFAULT']['tpc_name'])
    )

    input, actions, output = get_actions(config,
                                         'input',
                                         ['dsp', 'transform', 'my_postprocessing'],
                                         'output')
    actions += output   # Append output to actions... for now

    # Hand out input & output override instructions
    if 'input_name' in config['pax']:
        log.debug('User-defined input override: %s' % config['pax']['input_name'])
        config[input]['input_name'] = config['pax']['input_name']
    if 'output_name' in config['pax']:
        log.debug('User-defined output override: %s' % config['pax']['output_name'])
        # Hmmz, there can be several output plugins, some don't have a configuration...
        for o in output:
            if not o in config:
                config[o] = {}
            config[o]['output_name'] = config['pax']['output_name']

    # Gather information about plugins
    plugin_source = get_plugin_source(config, log)

    if plugin_source == None:
        raise RuntimeError("No plugin source found")

    input =    instantiate_plugin(input, plugin_source, config, log)
    actions = [instantiate_plugin(x,     plugin_source, config, log) for x in actions]

    # How should the events be generated?
    if 'events_to_process' in config['pax'] and config['pax']['events_to_process'] is not None:
        # The user specified which events to process:
        def get_events():
            for event_number in config['pax']['events_to_process']:
                yield input.get_single_event(event_number)
    else:
        # Let the input plugin decide which events to process:
        get_events = input.get_events

    # This is the actual event loop
    for i, event in enumerate(get_events()):
        if 'stop_after' in config['pax'] and config['pax']['stop_after'] is not None:
            if i >= config['pax']['stop_after']:
                log.info("User-defined limit of %d events reached: processing stopped." % i)
                break

        log.info("Event %d (%d processed)" % (event.event_number, i))

        process_single_event(actions, event, log)

    else:
        log.info("All events from input source have been processed.")

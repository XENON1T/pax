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


# Store the directory of pax (i.e. parent dir of this file's directory) as pax_dir
pax_dir = os.path.join(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))

##
# Configuration handling
##

def get_named_configuration_options():
    """ Return the names of all named configurations
    """
    config_files =[]
    for filename in glob.glob(os.path.join(pax_dir,
                                           'config',
                                           '*.ini')):
        filename = os.path.basename(filename)
        m = re.match(r'(\w+)\.ini', filename)
        if m is None:
            print("Weird file in config dir: %s" % filename)
        filename = m.group(1)
        # Config files starting with '_' won't work by themselves
        if filename[0] == '_':
            continue
        config_files.append(filename)
    return config_files

def parse_configuration_string(config_string):
    with StringIO(config_string) as config_fake_file:
        config = parse_configuration_file(config_fake_file)
    return config

def parse_named_configuration(config_name):
    """ Get pax configuration from a pre-cooked configuration in 'config'
    :param config_name: name of the config (without .ini)
    :return: output of parse_configuration_file
    """
    config = parse_configuration_file(os.path.join(pax_dir,
                                                   'config',
                                                   config_name + '.ini'))
    return config

def parse_configuration_file(file_object):
    """ Get pax configuration from a configuration file

    Configuration files can inherit from each other using parent_configuration.

    Each value in the ini file is eval()'ed in a context with physical unit variables:
        4 * 2 -> 8
        4 * cm**(2)/s -> correctly interpreted as a physical value with units of cm^2/s

    :param file_object: Path to the ini file to read, or opened file object to read
    :return: nested dictionary of evaluated configuration values, use as: config[section][key].
    """
    config = ConfigParser(inline_comment_prefixes='#',
                          interpolation=ExtendedInterpolation(),
                          strict=True)

    # Allow for case-sensitive configuration keys
    config.optionxform = str

    # Loads the file into configparser, also takes care of inheritance.
    load_file_into_configparser(config, file_object)

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
        if not os.path.isfile(config_file):
            raise ValueError("Configuration file %s does not exist!" % config_file)
        config.read(config_file)
    else:
        config.readfp(config_file)
    # Determine the path of the parent config file
    if 'final_ancestor' in config['pax'] and config['pax'] ['final_ancestor']:
        # We've reached the root of the inheritance line, there is no base file
        return
    elif 'parent_configuration' in config['pax'] :
        # This file inherits from another config file in the 'config' directory
        global pax_dir
        # The [1:-1] removes the quotes around the value... could do another eval, but lazy
        parent_file_path = os.path.join(pax_dir,
                                        'config',
                                        config['pax']['parent_configuration'][1:-1] + '.ini')
    elif 'parent_configuration_file' in config['pax']:
        # This file inherits from a user-defined config file
            parent_file_path = config['pax']['parent_configuration_file'][1:-1]
    else:
        raise RuntimeError('Missing inheritance instructions for config file %s!' %
                           str(config_file))
    # Unfortunately, configparser can only override settings, not set missing ones.
    # We have no choice but to load the parent file, then reload the original one again.
    # By doing this in a recursing function, multi-level inheritance is supported.
    load_file_into_configparser(config, parent_file_path)
    if isinstance(config_file, str):
        config.read(config_file)
    else:
        config.readfp(config_file)

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

    if name in config:
        this_plugin_config = config[name]
    else:
        log.debug('Plugin %s has no configuration!' % name)
        this_plugin_config = config['DEFAULT']

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


def processor(config, log_spec, events_to_process=None, stop_after=None, input_spec=None):
    """Run the processor according to the configuration dictionary

    :param config: dictionary with configuration values
    :param log_spec: loglevel specification (string)
    :return: None
    """

    # Setup logging
    numeric_level = getattr(logging, log_spec.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % log_spec.log.upper())

    logging.basicConfig(level=numeric_level,
                        format='%(name)s L%(lineno)s %(levelname)s %(message)s')
    log = logging.getLogger('processor')

    if config is None:
        log.warning("No configuration specified: loading Xenon100 config!")
        config = parse_named_configuration('XENON100')
    log.info("This is PAX version %s, running with configuration for %s." % (
        pax.__version__, config['DEFAULT']['tpc_name'])
    )

    input, actions, output = get_actions(config,
                                         'input',
                                         ['dsp', 'transform',
                                          'my_postprocessing'],
                                         'output')
    actions += output   # Append output to actions... for now

    # Set the input specification to config['DEFAULT']['input_specification']
    config[input]['input_specification'] = input_spec

    # Gather information about plugins
    plugin_source = get_plugin_source(config, log)

    if plugin_source == None:
        raise RuntimeError("No plugin source found")

    input =    instantiate_plugin(input, plugin_source, config, log)
    actions = [instantiate_plugin(x,     plugin_source, config, log) for x in actions]

    # How should the events be generated?
    if events_to_process is not None:
        # The user specified which events to process:
        def get_events():
            for event_number in events_to_process:
                yield input.get_single_event(event_number)
    else:
        # Let the input plugin decide which events to process:
        get_events = input.get_events

    # This is the actual event loop
    for i, event in enumerate(get_events()):
        if stop_after is not None:
            if i >= stop_after:
                log.info("User-specified limit of %d events reached: processing stopped." % i)
                break

        log.info("Event %d (%d processed)" % (event.event_number, i))

        process_single_event(actions, event, log)
    else:
        log.info("All events processed.")

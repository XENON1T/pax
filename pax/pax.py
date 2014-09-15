"""The core event processing for pax

"""
import logging
import inspect
import argparse
import configparser
import glob
import multiprocessing

from itertools import zip_longest

from io import StringIO
import os
from pluginbase import PluginBase

from pax import units


def evaluate_configuration(config):
    """Converts the ini-style configuration into a dictionary

    This includes eval-ing each value (i.e., 4 * 2 would be equal to 8).  It
    also includes the handling of units.  For example, 4 * mm would evaluate
    the units of mm.
    """
    evaled_config = {}

    for key, value in config.items():
        # Eval value with globals = everything from 'units' module...
        # TODO: this needs more explaination
        units_variables = {name: getattr(units, name) for name in dir(units)}
        evaled_config[key] = eval(value, units_variables)

    return evaled_config


def instantiate(name, plugin_source, config_values, log=logging):
    """Take plugin class name and build class from it"""
    log.debug('Instantiating %s' % name)
    name_module, name_class = name.split('.')
    try:
        plugin_module = plugin_source.load_plugin(name_module)
    except ImportError as e:
        log.fatal("Failed to load plugin %s" % name_module)
        log.exception(e)
        raise

    if config_values.has_section(name):
        this_config = config_values[name]
    else:
        this_config = config_values['DEFAULT']

    this_config = evaluate_configuration(this_config)

    instance = getattr(plugin_module, name_class)(this_config)

    log.info('Instantiated: %s' % name)

    return instance


def get_my_dir():
    """Find location of this file, then my_dir is its directory"""
    absolute_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    my_dir = os.path.dirname(absolute_path)
    return my_dir


def get_configuration(config_overload=""):
    my_dir = get_my_dir()

    config = configparser.ConfigParser(inline_comment_prefixes='#',
                                       strict=True)

    # Allow for case-sensitive configuration keys
    config.optionxform = str

    # Load the default configuration
    config.read(os.path.join(my_dir, 'default.ini'))
    config.read_string(config_overload)
    return config


def get_plugin_source(config, log=logging, ident='blah'):
    # Setup plugins (which involves finding the plugin directory.
    plugin_base = PluginBase(package='pax.plugins')
    searchpath = ['./plugins'] + config['DEFAULT']['plugin_paths'].split()
    # Find the absolute path, then director, then find plugin directory
    searchpath += [os.path.join(get_my_dir(), '..', 'plugins')]
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
    config = evaluate_configuration(config['pax'])

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


def process_single_event(actions, event, log):
    for j, block in enumerate(actions):
        log.debug("Step %d with %s", j, block.__class__.__name__)
        event = block.process_event(event)


def processor(config_overload=""):
    # Load configuration
    config = get_configuration(config_overload)

    input, actions, output = get_actions(config,
                                         'input',
                                         ['dsp', 'transform',
                                          'my_postprocessing'],
                                         'output')
    actions += output   # Append output to actions... why are they separate anyway?

    # Deal with command line arguments for the logging level
    parser = argparse.ArgumentParser(description="Process XENON1T data")
    parser.add_argument('--log', default='INFO', help="Set log level")

    # Event control
    input_control_group = parser.add_mutually_exclusive_group()
    input_control_group.add_argument('--single',
                                     type=int,
                                     help="Process a single event.")
    input_control_group.add_argument('-n',
                                     type=int,
                                     help="Stop after 'n' events processed.")

    args = parser.parse_args()

    # Setup logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log.upper())

    logging.basicConfig(level=numeric_level,
                        format='%(name)s L%(lineno)s %(levelname)s %(message)s')
    log = logging.getLogger('processor')

    # Print settings to log
    string_file = StringIO()
    config.write(string_file)
    # log.debug("Dumping INI file")
    # for line in string_file.getvalue().split('\n'):
    #     log.debug(line)

    # Gather information about plugins
    plugin_source = get_plugin_source(config, log)

    input = instantiate(input, plugin_source, config, log)
    actions = [instantiate(x, plugin_source, config, log) for x in actions]

    if args.single is not None:
        event = input.get_single_event(args.single)
        process_single_event(actions, event, log)
    else:
        # This is the *actual* event loop
        for i, event in enumerate(input.get_events()):
            if args.n is not None:
                if i >= args.n:
                    break

            log.info("Event %d" % i)

            process_single_event(actions, event, log)

        log.info("Finished event loop.")

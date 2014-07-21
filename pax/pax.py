import logging
import inspect
from io import StringIO
import argparse
import configparser

import os
import glob
from pluginbase import PluginBase

from pax import units


def evaluate_configuration(config):
    evaled_config = {}
    for key, value in config.items():
        # Eval value with globals = everything from 'units' module...
        evaled_config[key] = eval(value, {
            name: getattr(units, name)
            for name in dir(units)
        })
    return evaled_config


def instantiate(name, plugin_source, config_values, log=logging):
    """take class name and build class from it"""
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

    return getattr(plugin_module, name_class)(this_config)


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


def get_plugin_source(config, log=logging):
    # Setup plugins (which involves finding the plugin directory.
    plugin_base = PluginBase(package='pax.plugins')
    searchpath = ['./plugins'] + config['DEFAULT']['plugin_paths'].split()
    # Find the absolute path, then director, then find plugin directory
    searchpath += [os.path.join(get_my_dir(), '..', 'plugins')]
    # Want to look in all subdirectories of plugin directories
    subdirs = []
    for entry in searchpath:
        searchpath.extend(glob.glob(os.path.join(entry, '*/')))
    searchpath.extend(subdirs)

    log.debug("Search path for plugins is %s" % str(searchpath))
    plugin_source = plugin_base.make_plugin_source(searchpath=searchpath)
    log.info("Found the following plugins:")
    for plugin_name in plugin_source.list_plugins():
        log.info("\tFound %s" % plugin_name)

    return plugin_source


def processor(input, transform, output, config_overload=""):
    # TODO (tunnell): add DSP?
    # Check input types
    # TODO (tunnell): Don't use asserts, but raise ValueError() with
    # informative error
    assert isinstance(input, str)
    assert isinstance(transform, (str, list))
    assert isinstance(output, (str, list))

    # If 'transform' or 'output' aren't lists, turn them into lists
    if not isinstance(transform, list):
        transform = [transform]
    if not isinstance(output, list):
        output = [output]

    # What we do on data...
    actions = transform + output

    # Load configuration
    config = get_configuration(config_overload)

    # Deal with command line arguments for the logging level
    parser = argparse.ArgumentParser(description="Process XENON1T data")
    parser.add_argument('--log', default='INFO', help="Set log level")
    parser.add_argument('-n',
                        type=int,
                        help="Process fixed number of events")
    args = parser.parse_args()

    # Setup logging
    numeric_level = getattr(logging, args.log.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log.upper())

    logging.basicConfig(level=numeric_level,
                        format='%(name)s L%(lineno)s - %(levelname)s %(message)s')
    log = logging.getLogger('processor')

    # Print settings to log
    string_file = StringIO()
    config.write(string_file)
    log.debug("Dumping INI file")
    for line in string_file.getvalue().split('\n'):
        log.debug(line)

    # Gather information about plugins
    plugin_source = get_plugin_source(config, log)

    input = instantiate(input, plugin_source, config, log)
    actions = [instantiate(x, plugin_source, config, log) for x in actions]

    # This is the *actual* event loop
    for i, event in enumerate(input.get_events()):
        if args.n:
            if i >= args.n:
                break

        log.info("Event %d" % i)
        for j, block in enumerate(actions):
            log.debug("Step %d with %s", j, block.__class__.__name__)
            event = block.process_event(event)
    else:
        log.info("Finished event loop.")

import logging
import inspect
import pprint
import configparser

from pax import units

import os
from pluginbase import PluginBase

from pax import units

def EvaluateConfiguration(config):
    evaled_config = {}
    for key, value in config.items():
        #Eval value with globals = everything from 'units' module...
        evaled_config[key] = eval(value, {
            name : getattr(units, name)
            for name in dir(units)
        })
    return evaled_config

def Instantiate(name, plugin_source, config_values, log=logging):
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

    this_config = EvaluateConfiguration(this_config)

    return getattr(plugin_module, name_class)(this_config)


def Processor(input, transform, output):
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

    # Find location of this file
    absolute_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    dir = os.path.dirname(absolute_path)

    interpolation = configparser.ExtendedInterpolation()
    config = configparser.ConfigParser(interpolation=interpolation,
                                       inline_comment_prefixes='#',
                                       strict=True)
    # Allow for case-sensitive configuration keys
    config.optionxform = str

    # Load the default configuration
    config.read(os.path.join(dir, 'default.ini'))

    default_config = EvaluateConfiguration(config['DEFAULT'])

    # Setup logging
    string_level = default_config['loglevel']
    numeric_level = getattr(logging, string_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % string_level)
    FORMAT = '%(asctime)-15s %(name)s L%(lineno)s - %(levelname)s %(message)s'
    logging.basicConfig(level=numeric_level, format=FORMAT)
    log = logging.getLogger('Processor')

    # Print settings to log
    log.debug(pprint.pformat(config, compact=True))

    # Setup plugins (which involves finding the plugin directory.
    plugin_base = PluginBase(package='pax.plugins')
    searchpath = ['./plugins'] + config['DEFAULT']['plugin_paths'].split()


    # Find the absolute path, then director, then find plugin directory
    searchpath += [os.path.join(dir, '..', 'plugins')]
    log.debug("Search path for plugins is %s" % str(searchpath))
    plugin_source = plugin_base.make_plugin_source(searchpath=searchpath)
    log.info("Found the following plugins:")
    for plugin_name in plugin_source.list_plugins():
        log.info("\tFound %s" % plugin_name)

    # Instantiate requested plugins
    input = Instantiate(input, plugin_source, config, log)
    actions = [Instantiate(x, plugin_source, config, log) for x in actions]

    # This is the *actual* event loop
    for i, event in enumerate(input.GetEvents()):
        log.info("Event %d" % i)
        for j, block in enumerate(actions):
            log.debug("Step %d with %s", j, block.__class__.__name__)
            event = block.ProcessEvent(event)

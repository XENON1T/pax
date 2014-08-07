import logging
import inspect
import argparse
import configparser
import glob

from io import StringIO
import os
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
    for entry in searchpath:
        searchpath.extend(glob.glob(os.path.join(entry, '*/')))

    searchpath = [path for path in searchpath if '__pycache__' not in path]

    log.debug("Search path for plugins is %s" % str(searchpath))
    plugin_source = plugin_base.make_plugin_source(searchpath=searchpath)
    log.info("Found the following plugins:")
    for plugin_name in plugin_source.list_plugins():
        log.info("\tFound %s" % plugin_name)

    return plugin_source


def get_actions(config, input, list_of_names):
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

    return input, actions


def processor(config_overload=""):
    # Load configuration
    config = get_configuration(config_overload)

    input, actions = get_actions(config,
                                 'input',
                                 ['dsp', 'transform', 'my_postprocessing', 'output'])

    # Deal with command line arguments for the logging level
    parser = argparse.ArgumentParser(description="Process XENON1T data")
    parser.add_argument('--log', default='INFO', help="Set log level")

    # Event control
    input_control_group = parser.add_mutually_exclusive_group()
    input_control_group.add_argument('--single',
                       type=int,
                       help="Process a single event.")
    input_control_group.add_argument('--range',
                       type=int,
                       nargs=2,
                       help="Process inclusive range of events")
    input_control_group.add_argument('-n',
                       type=int,
                       help="Stop after this number of events has been processed.")

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


    # Does the input plugin support getting individual events?
    if hasattr(input, 'get_event'):
        # What is the range of events we want?
        event_range = []
        if args.single is not None:
            event_range = [args.single, args.single]
        elif args.range is not None:
            event_range = [args.range[0], args.range[1]]
        elif args.n is not None:
            max_number_of_events = (1 + input.last_event - input.first_event)
            if args.n > max_number_of_events:
                raise ValueError("There are only %s events in the file, can't process %s!" % (max_number_of_events, n))
            event_range = [input.first_event, input.first_event + args.n]
        else:
            event_range = [input.first_event, input.last_event]
        # Do the event loop:
        for event_number in range(event_range[0], event_range[1] + 1):    # +1 for funny python indexing
            log.info("Event %d" % event_number)
            process_one_event(input.get_event(event_number), actions, log)

    else:
        # We'll have to read events sequentlly, skipping events until we hit the desired range
        # This is much slower!
        for i, event in enumerate(input.get_events()):
            if args.n is not None:
                if i >= args.n:
                    break
            elif args.single is not None:
                if i < args.single:
                    continue
                elif i > args.single:
                    break
            elif args.range is not None:
                if i < args.range[0]:
                    continue
                elif i > args.range[1]:
                    break
            log.info("Event %d" % i)
            process_one_event(event, actions, log)

    log.info("Finished processing.")

def process_one_event(event, actions, log):

    for j, block in enumerate(actions):
        log.debug("Step %d with %s", j, block.__class__.__name__)
        event = block.process_event(event)
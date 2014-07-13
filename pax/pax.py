import logging
import argparse

from pluginbase import PluginBase
from confiture import Confiture
from confiture.schema import ValidationError
from confiture.parser import ParsingError
from pax.configuration import PaxSchema


def Instantiate(name, plugin_source, config_values):
    """take class name and build class from it"""
    name_module, name_class = name.split('.')
    plugin_module = plugin_source.load_plugin(name_module)
    return getattr(plugin_module, name_class)(config_values)


def Processor(input, transform, output, config_string=""):
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
    list_of_actions = transform + output

    # Handle configuration
    argument_parser = argparse.ArgumentParser()
    schema = PaxSchema()
    schema.populate_argparse(argument_parser)  # Fetch command line args
    argument_parser.parse_args()
    config = Confiture(config_string, schema=schema)
    try:
        pconfig = config.parse()
    except (ValidationError, ParsingError) as err:
        if err.position is not None:
            print(str(err.position))
        print(err)
        raise
    else:
        config_values = pconfig.to_dict()

    # Setup logging
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT)
    log = logging.getLogger('Processor')

    # Setup plugins
    plugin_base = PluginBase(package='pax.plugins')
    plugin_source = plugin_base.make_plugin_source(
        searchpath=['./plugins'] + config_values['plugin_paths'])

    # Instantiate requested plugins
    input = Instantiate(input, plugin_source, config_values)
    list_of_actions = [
        Instantiate(x, plugin_source, config_values) for x in list_of_actions]

    # This is the *actual* event loop
    for event in input.GetEvents():
        for i, block in enumerate(list_of_actions):
            log.warning("Step %d with %s", i, block.__class__.__name__)
            event = block.ProcessEvent(event)

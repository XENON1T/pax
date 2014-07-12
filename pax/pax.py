from pluginbase import PluginBase
import logging
from configglue import glue, schema
from pax import configuration


def Instantiate(name, plugin_source, config_values):
    """take class name and build class from it"""
    name_module, name_class = name.split('.')
    plugin_module = plugin_source.load_plugin(name_module)
    return getattr(plugin_module, name_class)(config_values)


def Processor(input, transform, output):
    if not isinstance(input, str):
        raise ValueError()
    assert isinstance(transform, (str, list))
    if not isinstance(output, str):
        raise ValueError()

    if not isinstance(transform, list):
        transform = [transform]

    a, b ,config_values, d = glue.configglue(configuration.XedReader, ['config.ini'])
    list_of_actions = [input] + transform + [output]
    Flow(list_of_actions, a.values('__main__'))


def Flow(list_of_blocks, config_values):
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT)

    plugin_base = PluginBase(package='pax.plugins')
    plugin_source = plugin_base.make_plugin_source(searchpath=['./plugins'])

    list_of_blocks = [Instantiate(x, plugin_source, config_values) for x in list_of_blocks]

    try:
        while (1):
            event = None
            for i, block in enumerate(list_of_blocks):
                print(block.__class__.__name__)
                event = block.ProcessEvent(event)

    except StopIteration:
        pass

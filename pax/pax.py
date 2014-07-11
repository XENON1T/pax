from pluginbase import PluginBase
import logging

def ConvertNameToClassInstance(name, plugin_source):
    print(name)
    name_module, name_class = name.split('.')
    plugin_module = plugin_source.load_plugin(name_module)
    return getattr(plugin_module, name_class)()

def Processor(input, transform, output):
    list_of_actions = [input, transform, output]
    Flow(list_of_actions)


def Flow(list_of_blocks):
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT)

    plugin_base = PluginBase(package='pax.plugins')
    plugin_source = plugin_base.make_plugin_source(searchpath=['./plugins'])

    list_of_blocks = [ConvertNameToClassInstance(x, plugin_source) for x in list_of_blocks]

    try:
        while (1):
            # Setup
            event = None
            for i, block in enumerate(list_of_blocks):
                print('\t', i)
                print(block.__class__.__name__)
                event = block.ProcessEvent(event)

    except StopIteration:
        pass


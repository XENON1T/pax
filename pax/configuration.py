import os
from configparser import ConfigParser, ExtendedInterpolation

import six

from pax import units, utils


def load_configuration(config_names=(), config_paths=(), config_string=None, config_dict=None):
    """Load pax configuration using configuration data. See the docstring of Processor for more info.
    :return: nested dictionary of evaluated configuration values, use as: config[section][key].
    """
    if config_dict is None:
        config_dict = {}

    # Support for string arguments
    if isinstance(config_names, str):
        config_names = [config_names]
    if isinstance(config_paths, str):
        config_paths = [config_paths]

    configp = ConfigParser(inline_comment_prefixes='#',
                           interpolation=ExtendedInterpolation(),
                           strict=True,
                           default_section='Why_doesnt_configparser_let_me_disable_DEFAULT')

    # Allow for case-sensitive configuration keys
    configp.optionxform = str

    # Make a list of all config paths / file objects we should load
    config_files = []
    for config_name in config_names:
        config_files.append(os.path.join(utils.PAX_DIR, 'config', config_name + '.ini'))
    for config_path in config_paths:
        config_files.append(config_path)
    if config_string is not None:
        config_files.append(six.StringIO(config_string))
    if len(config_files) == 0 and config_dict == {}:
        raise RuntimeError("You did not specify any configuration :-(")

    # Define an interior function for loading config files: supports recursion
    config_files_read = []

    def _load_file_into_configparser(config_file):
        """Loads a configuration file into our config parser, with support for inheritance.

        :param config_file: path or file object of configuration file to read
        :return: None
        """
        if isinstance(config_file, str):
            if not os.path.isfile(config_file):
                raise ValueError("Configuration file %s does not exist!" % config_file)
            if config_file in config_files_read:
                # This file has already been loaded: don't load it again
                # If we did, it would cause problems with inheritance diamonds
                return
            configp.read(config_file)
            config_files_read.append(config_file)
        else:
            config_file.seek(0)
            configp.read_file(config_file)
            # Apparently ConfigParser.read_file doesn't reset the read position?
            # Or maybe it has to do with using StringIO instead of real files?
            # Anyway, we want to read in the file again (for overriding parent instructions), so:
            config_file.seek(0)

        # Determine the path(s) of the parent config file(s)
        parent_file_paths = []

        if 'parent_configuration' in configp['pax']:
            # This file inherits from other config file(s) in the 'config' directory
            parent_files = eval(configp['pax']['parent_configuration'])
            if not isinstance(parent_files, list):
                parent_files = [parent_files]
            parent_file_paths.extend([
                os.path.join(utils.PAX_DIR, 'config', pf + '.ini')
                for pf in parent_files])

        if 'parent_configuration_file' in configp['pax']:
            # This file inherits from user-defined config file(s)
            parent_files = eval(configp['pax']['parent_configuration_file'])
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
            _load_file_into_configparser(pfp)
        if isinstance(config_file, str):
            configp.read(config_file)
        else:
            configp.read_file(config_file)

    # Loads the files into configparser, also takes care of inheritance.
    for config_file_thing in config_files:
        _load_file_into_configparser(config_file_thing)

    # Get a dict with all names visible by the eval:
    #  - all variables from the units submodule
    # NOT 'np', if you add a numpy array to the config, it will no longer be json serializable
    visible_variables = {name: getattr(units, name) for name in dir(units)}

    # Evaluate the values in the ini file
    evaled_config = {}
    for section_name, section_dict in configp.items():
        evaled_config[section_name] = {}
        for key, value in section_dict.items():
            # Eval value in a context where all units are defined
            evaled_config[section_name][key] = eval(value, visible_variables)

    # Apply the config_dict
    evaled_config = combine_configs(evaled_config, config_dict)

    # Make sure [DEFAULT] is at least present
    evaled_config['DEFAULT'] = evaled_config.get('DEFAULT', {})
    if 'Why_doesnt_configparser_let_me_disable_DEFAULT' in evaled_config:
        del evaled_config['Why_doesnt_configparser_let_me_disable_DEFAULT']

    return evaled_config


def combine_configs(config, override):
    """Apply overrides to config, then returns config.
    Config and overrides must be configuration dictionaries, i.e. have at most one level of sections.
    Settings in overrides override settings in config (as you might have guessed).
    """
    for section_name, section_config in override.items():
        config.setdefault(section_name, {})
        if not isinstance(section_config, dict):
            raise ValueError("COnfiguration dictionary should be a dictionary of dictionaries.")
        config[section_name].update(section_config)
    return config


def fix_sections_from_mongo(config):
    """Returns configuration with | replaced with . in section keys.
    Needed because . in field names has special meaning in MongoDB
    """
    return {k.replace('|', '.'): v for k, v in config.items()}

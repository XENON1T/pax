import os
from configparser import ConfigParser, ExtendedInterpolation

import six

from pax import units, utils
from pax.exceptions import InvalidConfigurationError


def load_configuration(config_names=(), config_paths=(), config_string=None, config_dict=None, maybe_call_mongo=False):
    """Load pax configuration using configuration data. See the docstring of Processor for more info.
    :param: maybe_call_mongo: if True, at the end of loading the config (but before applying config_dict)
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

    # Apply the config dict.
    evaled_config = combine_configs(evaled_config, config_dict)

    if maybe_call_mongo and evaled_config['pax'].get('look_for_config_in_runs_db'):
        # Connect to MongoDB. Do import here, since someone may wish to run pax without mongo
        from pax.MongoDB_ClientMaker import ClientMaker         # noqa
        run_collection = ClientMaker(evaled_config['MongoDB']).get_client(
            'run', autoreconnect=True)['run']['runs_new']

        # Get the run document, either by explicitly specified run number, or by the input name
        # The last option is a bit of a hack... if you don't like it, think of some way to always pass
        # the run number explicitly. By the way, let's hope nobody tries to reprocess run 0..
        if evaled_config.get('DEFAULT', {}).get('run_number', 0) > 0:
            run_number = evaled_config['DEFAULT']['run_number']
            run_doc = run_collection.find_one({'number': run_number})
            if not run_doc:
                raise InvalidConfigurationError("Unable to find run number %d!" % run_number)

        elif 'input_name' in evaled_config['pax']:
            run_name = os.path.splitext(os.path.basename(evaled_config['pax']['input_name']))[0]

            if run_name.endswith("_MV"):
                run_doc = run_collection.find_one({'name': run_name[:-3],
                                                   'detector': 'muon_veto'})
            else:
                run_doc = run_collection.find_one({'name': run_name,
                                                   'detector': 'tpc'})

            if not run_doc:
                raise InvalidConfigurationError("Unable to find a run named %s!" % run_name)

        else:
            raise InvalidConfigurationError("Cannot get configuration from runs db: give run_number or input_name!")

        # The run doc settings act as (but do not override) config_dict
        mongo_conf = fix_sections_from_mongo(run_doc.get('processor', {}))
        config_dict = combine_configs(mongo_conf, config_dict)

        # Add run number and run name to the config_dict
        config_dict.setdefault('DEFAULT', {})
        config_dict['DEFAULT']['run_number'] = run_doc['number']
        config_dict['DEFAULT']['run_name'] = run_doc['name']

        evaled_config = combine_configs(evaled_config, config_dict)

    # Make sure [DEFAULT] is at least present
    evaled_config['DEFAULT'] = evaled_config.get('DEFAULT', {})
    if 'Why_doesnt_configparser_let_me_disable_DEFAULT' in evaled_config:
        del evaled_config['Why_doesnt_configparser_let_me_disable_DEFAULT']

    return evaled_config


def combine_configs(*args):
    """Combines a series of configuration dictionaries; later ones have higher priority.
    Each argument must be a pax configuration dictionary, i.e. have at most one level of sections.
    """
    if not len(args):
        return {}
    elif len(args) == 1:
        return args[0]
    elif len(args) == 2:
        config, override = args
    else:
        return combine_configs(combine_configs(*args[:-1]), args[-1])

    for section_name, section_config in override.items():
        config.setdefault(section_name, {})
        if not isinstance(section_config, dict):
            raise ValueError("Configuration dictionary should be a dictionary of dictionaries.")
        config[section_name].update(section_config)
    return config


def fix_sections_from_mongo(config):
    """Returns configuration with | replaced with . in section keys.
    Needed because . in field names has special meaning in MongoDB
    """
    return {k.replace('|', '.'): v for k, v in config.items()}

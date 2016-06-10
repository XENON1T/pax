import logging
import re
import os

import pymongo
try:
    from monary import Monary
except Exception:
    pass   # Let's hope we're not the event builder.


class ClientMaker:
    """Helper class to create MongoDB clients

    On __init__, you can specify options that will be used to format mongodb uri's,
    in particular user, password, host and port.
    """
    def __init__(self, config):
        if 'password' not in config:
            config['password'] = os.environ.get('MONGO_PASSWORD')
            if not config['password']:
                raise ValueError("Please provide the mongo password in the environment variable MONGO_PASSWORD")
        # Select only relevant config options, so we can just pass this to .format later.
        self.config = {k: config[k] for k in ('user', 'password', 'host', 'port')}
        self.log = logging.getLogger('Mongo client maker')

    def get_client(self, database_name=None, uri=None, monary=False, **kwargs):
        """Get a Mongoclient. Returns Mongo database object.
        If you provide a mongodb connection string uri, we will insert user & password into it,
        otherwise one will be built from the configuration settings.
        If database_name=None, will connect to the default database of the uri. database=something
        overrides event the uri's specification of a database.
        """
        # Format of URI we should eventually send to mongo
        full_uri_format = 'mongodb://{user}:{password}@{host}:{port}/{database}'

        if uri is None:
            # We must construct the entire URI from default settings
            uri = full_uri_format.format(database=database_name, **self.config)
        else:
            # A URI was given. We expect it to NOT include user and password:
            uri_pattern = r'mongodb://([^:]+):(\d+)/(\w+)'
            m = re.match(uri_pattern, uri)
            if m:
                # URI was provided, but without user & pass.
                host, port, _database_name = m.groups()
                if database_name is None:
                    database_name = _database_name
                uri = full_uri_format.format(database=database_name, host=host, port=port,
                                             user=self.config['user'], password=self.config['password'])
            else:
                # Some other URI was provided. Maybe works...
                self.log.warning("Unexpected Mongo URI %s, expected format %s. Trying anyway..." % (uri, uri_pattern))

        if monary:
            self.log.debug("Connecting to Mongo via monary using uri %s" % uri)
            client = Monary(uri, **kwargs)
            self.log.debug("Succesfully connected via monary (probably...)")
            return client

        else:
            self.log.debug("Connecting to Mongo using uri %s" % uri)
            client = pymongo.MongoClient(uri, **kwargs)
            client.admin.command('ping')        # raises pymongo.errors.ConnectionFailure on failure
            self.log.debug("Successfully pinged client")
            return client

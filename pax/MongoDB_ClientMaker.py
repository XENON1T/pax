import logging
import time
import re
import os

import pymongo

try:
    from monary import Monary     # noqa
except Exception:
    pass   # Let's hope we're not the event builder.

try:
    from mongo_proxy import MongoProxy      # noqa
except Exception:
    # MongoDB Proxy did not load, falling back to vanilla pymongo
    def dummy(x, **kwargs):
        return x
    MongoProxy = dummy


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

    def get_client(self, database_name=None, uri=None, monary=False, host=None, autoreconnect=False, **kwargs):
        """Get a Mongoclient. Returns Mongo database object.
        If you provide a mongodb connection string uri, we will insert user & password into it,
        otherwise one will be built from the configuration settings.
        If database_name=None, will connect to the default database of the uri. database=something
        overrides event the uri's specification of a database.
        host is special magic for split_hosts
        kwargs will be passed to pymongo.mongoclient/Monary
        """
        # Format of URI we should eventually send to mongo
        full_uri_format = 'mongodb://{user}:{password}@{host}:{port}/{database}'

        if uri is None:
            # We must construct the entire URI from the settings
            uri = full_uri_format.format(database=database_name, **self.config)
        else:
            # A URI was given. We expect it to NOT include user and password:
            result = parse_passwordless_uri(uri)
            _host, port, _database_name = result
            if result is not None:
                if not host:
                    host = _host
                if database_name is None:
                    database_name = _database_name
                uri = full_uri_format.format(database=database_name, host=host, port=port,
                                             user=self.config['user'], password=self.config['password'])
            else:
                # Some other URI was provided. Just try it and hope for the best
                pass

        if monary:
            # Be careful enabling this debug log statement, it's useful but prints the password in the uri
            # self.log.debug("Connecting to Mongo via monary using uri %s" % uri)
            # serverselection option makes the C driver retry if it can't connect;
            # since we often make new monary connections this is useful to protect against brief network hickups.
            client = Monary(uri + '?serverSelectionTryOnce=false&serverSelectionTimeoutMS=60000', **kwargs)
            self.log.debug("Succesfully connected via monary (probably...)")
            return client

        else:
            # Be careful enabling this debug log statement, it's useful but prints the password in the uri
            # self.log.debug("Connecting to Mongo using uri %s" % uri)
            client = pymongo.MongoClient(uri, **kwargs)
            client.admin.command('ping')        # raises pymongo.errors.ConnectionFailure on failure
            self.log.debug("Successfully pinged client")

            if autoreconnect:
                # Wrap the client in a magic object that retries autoreconnect exceptions
                client = MongoProxy(client, disconnect_on_timeout=False, wait_time=180)

            return client


class PersistentRunsDBConnection:
    """Helper class for maitaining a persistent collection to the XENON1T runs database"""

    def __init__(self, clientmaker_config):
        self.log = logging.getLogger(self.__class__.__name__)
        self.clientmaker = ClientMaker(clientmaker_config)
        self._connect()

    def _connect(self):
        self.client = self.clientmaker.get_client('run', autoreconnect=True)
        self.db = self.client['run']
        self.collection = self.db['runs_new']
        self.pipeline_status_collection = self.db['pipeline_status']

    def check(self):
        """Checks that the runs db connection we currently have is alive. If not, we try to re-acquire it forever."""
        while True:
            try:
                self.client.admin.command('ping')
                return

            except Exception as e:
                self.log.fatal("Exception pinging runs db: %s: %s" % (type(e), str(e)))

                try:
                    self._connect()
                except Exception as e:
                    self.log.fatal("Could not re-acquire runs db connection: %s %s. Trying again in ten seconds." % (
                        type(e), str(e)))
                    time.sleep(10)


def parse_passwordless_uri(uri):
    """Return host, port, database_name"""
    uri_pattern = r'mongodb://([^:]+):(\d+)/(\w+)'
    m = re.match(uri_pattern, uri)
    if m:
        # URI was provided, but without user & pass.
        return m.groups()
    else:
        # Some other URI was provided. Just try it and hope for the best
        print("Unexpected Mongo URI %s, expected format %s." % (uri, uri_pattern))
        return None

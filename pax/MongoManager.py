import logging
import pymongo
import re
try:
    from monary import Monary
    MONARY_ENABLED = True
except ImportError:
    MONARY_ENABLED = False


class MongoManager:
    """Create clients for several databases, and shut them down properly
    """

    def __init__(self, config):
        self.config = config
        self.clients = {}
        self.log = logging.getLogger('MongoManager')
        self.monary_enabled = MONARY_ENABLED

    def get_database(self, database_name=None, uri=None, monary=False):
        """Get a connection to the database_name. Returns Mongo database object.
        If you provide a mongodb connection string uri, we will insert user & password into it,
        otherwise one will be built from the configuration settings.
        If database_name=None, will connect to the default database of the uri. database=something
        overrides event the uri's specification of a database.
        If there is already a client connecting to this database, no new mongoclient will be created.
        """
        # Pattern of URI's we expect from database (without user & pass)
        uri_pattern = r'mongodb://([^:]+):(\d+)/(\w+)'

        # Format of URI we should eventually send to mongo
        full_uri_format = 'mongodb://{user}:{password}@{host}:{port}/{database}'

        if uri is None:
            # Construct the entire URI from default settings
            uri = full_uri_format.format(database=database_name, **self.config)
        else:
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
            # Monary clients are not cached
            self.log.debug("Connecting to Mongo via monary using uri %s" % uri)
            return Monary(uri)

        elif database_name not in self.clients:
            self.log.debug("Connecting to Mongo using uri %s" % uri)
            client = pymongo.MongoClient(uri)

            if database_name is None:
                self.log.debug("Grabbing default database")
                database = client.get_default_database()
            else:
                self.log.debug("Grabbing database %s" % database_name)
                database = client[database_name]
            client.admin.command('ping')        # raises pymongo.errors.ConnectionFailure on failure
            self.log.debug("Succesfully pinged database %s" % database_name)

            self.clients[database_name] = dict(client=client, db=database)

        return self.clients[database_name]['db']

    def shutdown(self):
        self.log.debug('Shutting down all MongoDB clients')
        for v in self.clients.values():
            v['client'].close()

import logging
import pymongo
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
        """Get a connection to the database_name. Returns Mongo database access object.
        If you provide a mongodb connection string uri, it will be used, otherwise one will be built
        from the configuration settings.
        If database_name=None, will connect to the default database of the uri.
        If there is already a client connecting to this database, no new mongoclient will be created.
        """
        if uri is None:
            uri = 'mongodb://{user}:{password}@{host}:{port}/{database}'.format(database=database_name,
                                                                                **self.config)
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

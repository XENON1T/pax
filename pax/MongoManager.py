import logging
import pymongo


class MongoManager:
    """Create clients for several databases, and shut them down properly
    """

    def __init__(self, config):
        self.config = config
        self.log = logging.getLogger('MongoManager')

    def get_database(self, database_name=None):
        uri = 'mongodb://{user}:{password}@{host}:{port}'.format(**self.config)

        if database_name not in self.clients:
            self.log.debug("Connecting to Mongo using uri %s" % uri)
            client = pymongo.MongoClient(uri)

            if database_name is None:
                self.log.debug("Grabbing default database")
                database = client.get_default_database()
            else:
                self.log.debug("Grabbing database %s" % database_name)
                database = client[database_name]

            database.authenticate(self.config['username'], self.config['password'])
            client.admin.command('ping')        # raises pymongo.errors.ConnectionFailure on failure

            self.clients[database_name] = client

        return database

    def shutdown(self):
        self.log.debug('Shutting down all MongoDB clients')
        for client in self.clients.values():
            client.close()

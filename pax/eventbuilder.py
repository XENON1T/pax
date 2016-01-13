"""Event builder routines

The event builder is responsible for finding events within the DAQ output
stream and turning this into events that the end user can process.  The data is
fetched untriggered, which means that there is no event structure.  Or rather,
it is just a stream of PMT pulses that need to be assembled into events.  This
is exactly what this code does.  There are two inputs requires:

* Location of runs database
* Trigger settings

This code is meant to run in production, therefore a runs database must be
 provided.  This code will use this database to locate what data needs to be
 processed and handle it appropriately.

This code is also responsible triggering (i.e., deciding which events to
record).  At present there are two stages to the triggering: a pretrigger and
a pax-event trigger.  The pretrigger is a simple sliding window coincidence
trigger that runs over the data to identify possible signals.  The parameters
of this are then, clearly, the size of the window and the coincidence
requirement.  Another parameter is how much data to save around the discovered
peak, which often is an S2.

The second level trigger, which we often do not run, processed the event in pax
and is used to make, e.g., radial cuts.  At present, the code just writes HDF5
files.

It is worth mentioning that is an 'expert modes' with a flag for mega events:
an event of one second is made that can be used for computing the trigger
efficiency.

The entry point to this code is typically via bin/event-builder.
"""

import argparse
import logging
import os
import pymongo
import time
from pax import core, units


def run():
    """Start running the event builder

    Find a dataset to process, then process it with settings from command line.
    """

    # Fetch command line arguments
    args, log = handle_args()

    logging.info("Connection to %s" % args.mongo)
    client = pymongo.MongoClient(args.mongo)

    try:
        client.admin.command('ping')
        log.debug("Connection successful to %s", args.mongo)
    except pymongo.errors.ConnectionFailure:
        log.fatal("Cannot connect to MongoDB at %s" % (args.mongo))
        raise

    authenticate(client)

    status_name = 'detectors.tpc.trigger.status'

    if args.name:
        query = {'name' : args.name}
    else:
        query = {status_name: 'waiting_to_be_processed'}

    log.info("Searching for run")

    db = client.get_default_database()
    log.debug('Fetched databases: %s', db.name)

    collection = db.get_collection('runs')
    log.debug('Got collection: %s', collection.name)

    while 1:
        run_doc = collection.find_one_and_update(query,
                                                 {'$set': {status_name: 'staging'}})

        if run_doc is None:
            if args.impatient:
                log.info("Too impatient to wait for data, exiting...")
                break
            else:
                log.info("No data to process... waiting %d seconds",
                         args.wait)
                time.sleep(args.wait)
        else:
            log.info("Building events for %s", run_doc['name'])

            pax_config = {'output_name': 'raw_%s' % run_doc['name']}

            config_names = 'eventbuilder'
            config_dict = {'DEFAULT': {'run_doc': run_doc['_id']},
                           'pax': pax_config,
                           'MongoDB': {'runs_database': args.mongo,
                                       'window': args.window * units.us,
                                       'left': args.left * units.us,
                                       'right': args.right * units.us,
                                       'multiplicity': args.multiplicity,
                                       'mega_event': args.mega_event
                                       }
                           }
            try:
                p = core.Processor(config_names=config_names,
                                   config_dict=config_dict)

                p.run()

            except pymongo.errors.ServerSelectionTimeoutError as e:
                log.exception(e)
                collection.update(query,
                                  {'$set': {status_name: 'error'}})
                raise

        # If we're trying to build a single run, don't try again to find it
        if args.name:
            break


def authenticate(client, database_name=None):
    try:
        mongo_user = os.environ['MONGO_USER']
    except KeyError:
        raise RuntimeError("You need to set the variable MONGO_USER."
                           "\texport MONGO_USER=eb")
    try:
        mongo_password = os.environ['MONGO_PASSWORD']
    except KeyError:
        raise RuntimeError("You need to set the variable MONGO_PASSWORD."
                           "\texport MONGO_PASSWORD=XXXXXX")

    if database_name is None:
        database = client.get_default_database()
    else:
        database = client[database_name]
    database.authenticate(mongo_user, mongo_password)


def handle_args():
    """Command line argument processing

    This routine is also responsible for setting up logging.
    """
    parser = argparse.ArgumentParser(description="Build XENON1T events from the"
                                                 " data aquisiton. This tools "
                                                 "starts the distributed "
                                                 "processing of events.")
    parser.add_argument('--name',
                        type=str,
                        help="Instead of building all waiting_to_be_processed runs,"
                             "look for this specific run and build it.")

    parser.add_argument('--impatient',
                        action='store_true',
                        help="Event builder will not wait for new data")
    parser.add_argument('--mega_event',
                        action='store_true',
                        help="used for trigger efficiency")
    trigger_group = parser.add_argument_group(title='Event builder settings',
                                              description='Configure trigger')
    trigger_group.add_argument('--multiplicity',
                               type=int,
                               help='Number pulses required for coincidence '
                                    'trigger',
                               default=10)
    trigger_group.add_argument('--window',
                               type=int,
                               help='Size of sliding window (us)',
                               default=1)
    trigger_group.add_argument('--left',
                               type=int, default=200,
                               help='Left extension to save (us)')
    trigger_group.add_argument('--right',
                               type=int, default=200,
                               help='Right extension to save (us)')

    parser.add_argument('--wait',
                        default=1,
                        type=int,
                        help="Wait time between searching if no data")

    parser.add_argument('--mongo',
                        default='mongodb://master:27017,master:27018/run',
                        type=str,
                        help='Run database MongoDB URI')

    # Log level control
    parser.add_argument('--log', default=None,
                        help="Set log level, e.g. 'debug'")
    # set up logging to file - see previous section for more details
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)s %(levelname)-8s %('
                               'message)s',
                        datefmt='%m-%d %H:%M',
                        filename='myapp.log',
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(asctime)s %(name)-12s: '
                                  '%(levelname)-8s %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)
    log = logging.getLogger('eb')
    args = parser.parse_args()
    return args, log

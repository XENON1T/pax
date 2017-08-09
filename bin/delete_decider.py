"""Decides which runs can/cannot be deleted

Currently this just deletes any run 2 hours after acquisition.
In the future this should also watch for warnings from other experiments (e.g. those watching for galactic supernovae).

For safety, the watching for external triggers and delete permission setting should happen in the same script.
It is not safe enough, for example, to just set a 'preserve' tag here, and let the event builder itself delete
runs without a preserve tag after X hours. Then, if this script was not running, we could accidentally delete the data
coincident with an interesting external trigger.

A run that is marked for preservation by this script should be mongodumped to disk, then the preserve tag should be
manually removed. This is not done in this script.
"""
import os
import time
import argparse
import datetime
from prettytable import PrettyTable

from pax.MongoDB_ClientMaker import PersistentRunsDBConnection

##
# Command line parsing
##

parser = argparse.ArgumentParser(description="Set delete permissions for XENON1T untriggered data")

parser.add_argument('--wait', default=10, type=int,
                    help="Wait time (sec) before repeating decision process. If 0, instead shuts down after one round.")
parser.add_argument('--check_only', action='store_true',
                    help="Just print which runs still have data present, then exit. Don't modify anything.")

marking_group = parser.add_argument_group(title='Marking current run for preservation')
marking_group.add_argument('--preserve_now', action='store_true',
                           help='Mark the run that is currently being acquired for preservation, then exit.')
marking_group.add_argument('--desc', default='Manually marked this run for preservation, no description given.',
                           help='Describe reason for preserving current run, will be included in runs db.')

connection_group = parser.add_argument_group(title='MongoDB connection settings')
# TODO: Unfortunate defaults are hardcoded here, again in configuration, and again in event-builder.py...
connection_group.add_argument('--host', type=str, default='gw',
                              help='MongoDB hostname')
connection_group.add_argument('--port', type=int, default=27017,
                              help='Port on host where MongoDB runs')
connection_group.add_argument('--user', type=str, default='eb',
                              help='User to connect to MongoDB')
connection_group.add_argument('--password', type=str, default=os.environ.get('MONGO_PASSWORD'),
                              help='Password to connect to MongoDB. '
                                   'If not provided, will try to use MONGO_PASSWORD from env')

args = parser.parse_args()


# Connect to runs db
rundb = PersistentRunsDBConnection({k: getattr(args, k) for k in ('user', 'password', 'host', 'port')})


while True:
    # Check if the runs db is still live, don't proceed otherwise.
    rundb.check()

    if args.preserve_now:
        new_external_trigger = True
        external_trigger_time = datetime.datetime.utcnow()
        external_trigger_description = args.desc

    elif args.check_only:
        new_external_trigger = False

    else:
        # Check for any NEW external triggers here.
        #
        # Add code here.
        #
        new_external_trigger = False

    if new_external_trigger:
        # Find the run that includes the trigger (if one exists), and mark it for preservation
        print("External trigger at UTC timestamp %0.2f: %s." % (external_trigger_time,
                                                                external_trigger_description))

        runs = list(rundb.collection.find({'detector': 'tpc',
                                           'start': {'$gt', external_trigger_time},
                                           'end':   {'$lt', external_trigger_time}},
                                          projection=['_id', 'number']))
        if len(runs) == 0:
            raise ValueError("No run found that matches trigger time. :-( "
                             "Perhaps the DAQ was off, or taking calibration data? Please verify manually.")

        if len(runs) > 1:
            raise ValueError("More than one run matches the trigger time?! Numbers: %s")

        run_doc = runs[0]
        print("Run %d contains the external trigger, marking it for preservation." % run_doc['number'])

        rundb.collection.find_one_and_update({'_id': run_doc['_id']},
                                             {'$set': {'preserve': True,
                                                       'preserve_trigger_timestamp': external_trigger_time,
                                                       'preserve_decision_timestamp': datetime.datetime.utcnow(),
                                                       'preserve_reason': external_trigger_description}})

    # Find and print which runs (should) still have untriggered data stored
    table = PrettyTable(['Run number', 'name', 'start', 'end', 'preserve', 'reason'])
    run_fields = 'number name start end preserve preserve_reason'.split()

    for run_doc in rundb.collection.find({'trigger.status': 'trigger_done'},
                                         projection=run_fields + ['_id']):
        table.add_row([run_doc.get(x, False) for x in run_fields])

        if (
                    not args.check_only and
                    not run_doc.get('preserve', False) and
                    run_doc['end'] > datetime.datetime.utcnow() + datetime.timedelta(hours=2)
        ):
            print("Marking run %d for deletion." % run_doc['number'])
            rundb.collection.find_one_and_update({'_id': run_doc['_id']},
                                                 {'$set': {'trigger.status': 'deleteme'}})

    print(str(table))

    if args.preserve_now or args.check_only:
        break

    print("Sleeping for %d seconds..." % args.wait)
    time.sleep(args.wait)

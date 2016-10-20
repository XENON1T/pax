#!/usr/env/python
import time
import argparse
import multiprocessing
try:
    from queue import Empty
except ImportError:
    from Queue import Empty

from pax import core, parallel


parser = argparse.ArgumentParser(description="Listen to a RabbitMQ server for instructions to create pax instances",)

parser.add_argument('--url', default=None,
                    help="URL to connect to the RabbitMQ server. "
                         "If not provided, tries to connect to a local server configured with default credentials.")

parser.add_argument('--cpus', default=2,
                    help='Maximum number of CPUS to dedicate to pax = maximum number of pax instances to start.')

parser.add_argument('--startup_queue', default='pax_startup',
                    help='Name for the pax startup request queue, usually just leave as pax_startup (default).')

parser.add_argument('--startup_queue', default='pax_crashes',
                    help='Name for the pax crash reporting queue, usually just leave as pax_crashes (default).')


args = parser.parse_args()

# Setup the startup queue. Here we will listen for messages for starting pax, which should look like
#  (pax_id, {config_names=..., config_paths=..., etc})
startup_queue = parallel.RabbitQueue(args.url, args.startup_queue)

# Setup the crash-watching fanout. Here we can send/receive messages denoting a pax crash
#  (pax_id, extra_info)
# where extra info should (at least when I get around to it ;-) be a nice message about the cause of the crash.
# When a crash message is enountered all paxes with pax_id will be terminated (SIGTERM).
# paxmaker can also send such a message if it encounters a crash in pax,
# which will fan out to every other connected paxmaker.
crash_fanout = parallel.RabbitFanOut(args.url, args.crash_watch_queue)

running_paxes = []
max_paxes = args.max_cpu

while True:
    time.sleep(1)

    p_by_status = parallel.group_by_status(running_paxes)
    running_paxes = p_by_status['running']

    # If any of our own paxes crashed, send a message to the crash fanout
    # This will inform everyone connected to the server (including ourselves)
    for pax_id in set([p.pax_id for p in p_by_status['crashed']]):
        crash_fanout.put((pax_id, 'exception tracking not yet implemented'))

    # Check and handle messages in the crash fanout
    try:
        pax_id, exception = crash_fanout.get()
        print("Terminating all paxes with id %s due to remote crash: %s" % (pax_id, exception))
        # Terminate all running paxes processes with a matching pax_id
        for p in running_paxes:
            if p.pax_id == pax_id:
                p.terminate()
        continue
    except Empty:
        pass

    # TODO: better status line
    print("%d pax slots available." % (len(running_paxes) - max_paxes))

    # Check and handle pax startup requests
    try:
        msg = startup_queue.get()
    except Empty:
        continue

    if len(running_paxes) >= max_paxes:
        # We're already full; can't start another pax. Let someone else do it.
        startup_queue.put(msg)
        continue

    # Start a new pax and append it to the list of running paxes
    pax_id, kwargs = msg
    newpax = multiprocessing.Process(target=core.Processor, kwargs=kwargs)
    newpax.pax_id = pax_id
    newpax.start()
    running_paxes.append(newpax)

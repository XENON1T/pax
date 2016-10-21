from copy import deepcopy
from collections import defaultdict
import multiprocessing
import time
import traceback
import pickle

from . import utils
from .core import Processor
from .configuration import combine_configs

import rabbitpy

try:
    import queue
except ImportError:
    import Queue as queue

Empty = queue.Empty

# Pax data queue message codes
REGISTER_PUSHER = -11
PUSHER_DONE = -12
NO_MORE_EVENTS = -42

##
# RabbitMQ interface
##

DEFAULT_RABBIT_URI = 'amqp://guest:guest@localhost:5672/%2f'


class RabbitQueue:
    """A wrapper for "ordinary" RabbitMQ queues to make them behave just like python standard library queues...
    -- at least the put, get and qsize methods...
    """

    def __init__(self, queue_name, uri=DEFAULT_RABBIT_URI):
        self.queue_name = queue_name
        self.conn = rabbitpy.Connection(uri)
        self.channel = self.conn.channel()
        self.queue = rabbitpy.Queue(self.channel, self.queue_name)
        self.queue.declare()

    def put(self, message):
        message = pickle.dumps(message)
        rabbitpy.Message(self.channel, message).publish('', self.queue_name)

    def get(self, **kwargs):
        """Get an item from the queue. Kwargs are ignored (often used in standard library queue.get calls)"""
        msg = self.queue.get(acknowledge=False)
        if msg is None:
            raise Empty
        return pickle.loads(msg.body)

    def qsize(self):
        return len(self.queue)

    def close(self):
        self.channel.close()
        self.conn.close()


class RabbitFanOut(RabbitQueue):
    """A wrapper similar to RabbitQueue for RabbitMQ FanOut exchanges
    """

    def __init__(self, exchange_name, uri=DEFAULT_RABBIT_URI):
        self.conn = rabbitpy.Connection(uri)
        self.channel = self.conn.channel()
        self.exchange = rabbitpy.Exchange(self.channel, exchange_name, exchange_type='fanout')
        self.exchange.declare()
        self.queue = rabbitpy.Queue(self.channel, exclusive=True)
        self.queue.declare()
        self.queue.bind(self.exchange)

    def put(self, message):
        message = pickle.dumps(message)
        rabbitpy.Message(self.channel, message).publish(self.exchange)


def add_rabbit_command_line_args(parser):
    """Add connection arguments for RabbitMQ parser"""
    rabbit_args = [
        ('username', 'guest', 'Username for'),
        ('password', 'guest', 'Password for'),
        ('host', 'localhost', 'Hostname of'),
        ('port', 5672, 'Port to connect to'),
    ]
    for setting, default, helpprefix in rabbit_args:
        parser.add_argument('--rabbit_%s' % setting,
                            default=default,
                            type=int if setting == 'port' else str,
                            help='%s the RabbitMQ message broker (default %s)' % (helpprefix, default))


def url_from_parsed_args(parsed_args):
    """Return RabbitMQ connection URL from argparser args"""
    return 'amqp://{username}:{password}@{host}:{port}/%2f'.format(**{x: getattr(parsed_args, 'rabbit_' + x)
                                                                      for x in 'username password host port'.split()})

##
# Pax multiprocessing code
##


def multiprocess_configuration(n_cpus, base_config_kwargs, processing_queue_kwargs, output_queue_kwargs):
    """Yields configuration override dicts for multiprocessing"""
    # Config overrides for child processes
    common_override = dict(pax=dict(autorun=True, show_progress_bar=False))

    input_override = dict(pax=dict(plugin_group_names=['input', 'output'],
                                   encoder_plugin=None,
                                   decoder_plugin=None,
                                   output='Queues.PushToQueue'),
                          Queues=processing_queue_kwargs)

    worker_override = {'pax': dict(input='Queues.PullFromQueue',
                                   output='Queues.PushToQueue',
                                   event_numbers_file=None,
                                   events_to_process=None),
                       'Queues.PullFromQueue': processing_queue_kwargs,
                       'Queues.PushToQueue': dict(preserve_ids=True,
                                                  many_to_one=True,
                                                  **output_queue_kwargs)}

    output_override = dict(pax=dict(plugin_group_names=['input', 'output'],
                                    encoder_plugin=None,
                                    decoder_plugin=None,
                                    event_numbers_file=None,
                                    events_to_process=None,
                                    input='Queues.PullFromQueue'),
                           Queues=dict(ordered_pull=True,
                                       **output_queue_kwargs))

    overrides = [('input', input_override)] + [('worker', worker_override)] * n_cpus + [('output', output_override)]

    for worker_type, worker_overide in overrides:
        new_conf = deepcopy(base_config_kwargs)
        new_conf['config_dict'] = combine_configs(new_conf.get('config_dict'),
                                                  common_override,
                                                  worker_overide)
        yield worker_type, new_conf


def maybe_multiprocess(args, **config_kwargs):
    """Start a pax with config_kwargs (config_names, config_dict, etc), and let argparser parser args args control
    whether or not we should multiprocess
    """
    if args.cpus > 1:
        if args.remote:
            url = url_from_parsed_args(args)
            multiprocess_remotely(n_cpus=args.cpus, url=url, **config_kwargs)
        else:
            multiprocess_locally(n_cpus=args.cpus, **config_kwargs)
    else:
        pax_instance = Processor(**config_kwargs)

        try:
            pax_instance.run()
        except (KeyboardInterrupt, SystemExit):
            print("\nShutting down all plugins...")
            pax_instance.shutdown()
            print("Exiting")


def multiprocess_locally(n_cpus, **kwargs):
    # Setup an output and worker queue
    manager = multiprocessing.Manager()
    processing_queue = manager.Queue()
    output_queue = manager.Queue()

    # Initialize the various worker processes
    running_workers = []

    configs = multiprocess_configuration(n_cpus,
                                         base_config_kwargs=kwargs,
                                         processing_queue_kwargs=dict(queue=processing_queue),
                                         output_queue_kwargs=dict(queue=output_queue))

    for _, config_kwargs in configs:
        running_workers.append(start_safe_processor(manager, **config_kwargs))

    # Check the health / status of the workers every second.
    while len(running_workers):
        time.sleep(1)

        # Filter out only the running workers
        p_by_status = group_by_status(running_workers)
        running_workers = p_by_status['running']

        if len(p_by_status['crashed']):
            for p in running_workers:
                p.terminate()
                raise RuntimeError("Pax multiprocessing crashed due to exception in one of the workers")

        status_line(running_workers, processing_queue, output_queue)


def multiprocess_remotely(n_cpus=2, pax_id=None, url=DEFAULT_RABBIT_URI,
                          startup_queue_name='pax_startup', crash_watch_fanout_name='pax_crashes',
                          **kwargs):
    manager = multiprocessing.Manager()
    if pax_id is None:
        pax_id = 'pax_%s' % utils.randomstring(6)

    # Setup an output and worker queue. We connect to the processing queue here just to display the queue size.
    pq_name = 'pax_%s_processing' % pax_id
    oq_name = 'pax_%s_output' % pax_id
    processing_queue = RabbitQueue(pq_name, url)
    output_queue = RabbitQueue(oq_name, url)

    startup_queue = RabbitQueue(startup_queue_name, url)
    crash_fanout = RabbitFanOut(crash_watch_fanout_name, url)

    # Initialize worker processes
    configs = multiprocess_configuration(n_cpus,
                                         base_config_kwargs=kwargs,
                                         processing_queue_kwargs=dict(queue_name=pq_name, queue_url=url),
                                         output_queue_kwargs=dict(queue_name=oq_name, queue_url=url))

    local_paxes = []
    for process_type, config_kwargs in configs:
        if process_type in ['input', 'output']:
            # Endpoints start as local processes
            w = start_safe_processor(manager, **config_kwargs)
            w.pax_id = pax_id
            local_paxes.append(w)
        else:
            # Workers start as remote processes
            startup_queue.put((pax_id, config_kwargs))

    # Check the health / status of the workers every second.
    while len(local_paxes):
        time.sleep(1)

        local_paxes = check_local_processes_while_remote_processing(local_paxes, crash_fanout)

        status_line(local_paxes, processing_queue, output_queue)


def status_line(local_processes, processing_queue, output_queue):
    utils.refresh_status_line("[Pax multiprocessing]: %d local processes, "
                              "%d messages in processing queue, %d in output queue" %
                              (len(local_processes), processing_queue.qsize(), output_queue.qsize()))


def check_local_processes_while_remote_processing(running_paxes, crash_fanout):
    """Check on locally running paxes in running_paxes, returns list of remaining running pax processes.
     - Remove any paxes that have exited normally
     - If a pax has crashed, push a message to the crash fanout to terminate all paxes with the same id
     - Look for crash fanout messages from other processes, and terminate local paxes with the same id
    """
    p_by_status = group_by_status(running_paxes)
    running_paxes = p_by_status['running']

    # If any of our own paxes crashed, send a message to the crash fanout
    # This will inform everyone connected to the server (including ourselves, on the next iteration)
    for crashed_w in p_by_status['crashed']:
        pax_id = crashed_w.pax_id
        traceb = crashed_w.shared_dict.get('traceback', "No traceback reported.")
        print("Pax %s crashed!\nDumping exception traceback:\n\n%s\n\nNotifying crash fanout." % (
            pax_id, format_exception_dump(traceb)
        ))
        crash_fanout.put((pax_id, traceb))

    # If any of the remote paxes crashed, we will learn about it from the crash fanout.
    try:
        pax_id, traceb = crash_fanout.get()
        print("Remote crash notification for pax %s.\n"
              "Remote exception traceback dump:\n\n%s\n.Terminating paxes with id %s." % (
                pax_id, format_exception_dump(traceb), pax_id))
        # Terminate all running paxes processes with a matching pax_id
        # Must build a new list here, if we call group_by_status again it would report every terminated process
        # as a crash without a traceback.
        new_running_paxes = []
        for p in running_paxes:
            if p.pax_id == pax_id:
                p.terminate()
            else:
                new_running_paxes.append(p)
        running_paxes = new_running_paxes
    except Empty:
        pass

    return running_paxes


def group_by_status(plist):
    """Given a list of multiprocess.Process processes, return dict of lists of processes by status.
    status keys: running, completed (for exit code 0) and crashed (exit code other than 0)
    """
    result = defaultdict(list)
    for p in plist:
        if p.exitcode is None:
            result['running'].append(p)
        elif p.exitcode == 0:
            result['completed'].append(p)
        else:
            result['crashed'].append(p)
    return result


def start_safe_processor(manager, **kwargs):
    """Start a processor with kwargs in a new process. Add an exception container in the exception attribute."""
    shared_dict = manager.dict()
    w = multiprocessing.Process(target=safe_processor, args=[shared_dict], kwargs=kwargs)
    w.start()
    w.shared_dict = shared_dict
    return w


def safe_processor(shared_dict, **kwargs):
    """Starts a pax processor with kwargs. If it dies with an exception, update the value in exception container."""
    try:
        Processor(**kwargs)
    except Exception:
        shared_dict['traceback'] = traceback.format_exc()
        raise


def format_exception_dump(traceb):
    return '\t\t'.join(traceb.splitlines(True))

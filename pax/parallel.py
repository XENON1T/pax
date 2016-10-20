from copy import deepcopy
from collections import defaultdict
import multiprocessing
import time
import pickle
try:
    from queue import Empty
except ImportError:
    from Queue import Empty

from .core import Processor
from .configuration import combine_configs

import rabbitpy

# Pax data queue message codes
REGISTER_PUSHER = -11
PUSHER_DONE = -12
NO_MORE_EVENTS = -42


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
        msg = self.queue.get()
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


def multiprocess_configuration(n_cpus, base_config_kwargs, processing_queue_kwargs, output_queue_kwargs):
    """Yields configuration override dicts for multiprocessing"""
    # Config overrides for child processes
    common_override = dict(pax=dict(autorun=True))

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

    for worker_overide in [input_override] + [worker_override] * n_cpus + [output_override]:
        new_conf = deepcopy(base_config_kwargs)
        new_conf['config_dict'] = combine_configs(new_conf.get('config_dict'),
                                                  common_override,
                                                  worker_overide)
        yield new_conf


def multiprocess_locally(n_cpus, **kwargs):
    # Setup an output and worker queue
    manager = multiprocessing.Manager()
    processing_queue = manager.Queue()
    output_queue = manager.Queue()

    # Initialize the various worker processes
    running_workers = []

    for config_kwargs in multiprocess_configuration(n_cpus,
                                                    base_config_kwargs=kwargs,
                                                    processing_queue_kwargs=dict(queue=processing_queue),
                                                    output_queue_kwargs=dict(queue=output_queue)):
        w = multiprocessing.Process(target=Processor, kwargs=config_kwargs)
        w.start()
        running_workers.append(w)

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

        # TODO: better status line
        print(running_workers, processing_queue.qsize(), output_queue.qsize())



# def multiprocess_remotely(n_cpus, **kwargs):



def group_by_status(plist):
    result = defaultdict(list)
    for p in plist:
        if p.exitcode is None:
            result['running'].append(p)
        elif p.exitcode is 0:
            result['completed'].append(p)
        else:
            result['crashed'].append(p)
    return result

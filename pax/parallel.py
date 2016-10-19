from copy import deepcopy
import multiprocessing
import time

from .core import Processor
from .configuration import combine_configs

REGISTER_PUSHER = -11
PUSHER_DONE = -12
NO_MORE_EVENTS = -42


def multiprocess_locally(n_cpus, **kwargs):
    # Setup an output and worker queue
    manager = multiprocessing.Manager()
    processing_queue = manager.Queue()
    output_queue = manager.Queue()

    # Config overrides for child processes
    common_override = dict(pax=dict(autorun=True))

    input_override = dict(pax=dict(plugin_group_names=['input', 'output'],
                                   encoder_plugin=None,
                                   decoder_plugin=None,
                                   output='Queues.PushToQueue'),
                          Queues=dict(queue=processing_queue))

    worker_override = {'pax': dict(input='Queues.PullFromQueue',
                                   output='Queues.PushToQueue',
                                   event_numbers_file=None,
                                   events_to_process=None),
                       'Queues.PullFromQueue': dict(queue=processing_queue),
                       'Queues.PushToQueue': dict(queue=output_queue,
                                                  preserve_ids=True,
                                                  many_to_one=True)}

    output_override = dict(pax=dict(plugin_group_names=['input', 'output'],
                                    encoder_plugin=None,
                                    decoder_plugin=None,
                                    event_numbers_file=None,
                                    events_to_process=None,
                                    input='Queues.PullFromQueue'),
                           Queues=dict(queue=output_queue,
                                       ordered_pull=True))

    # Initialize the various worker processes
    living_workers = []
    for worker_overide in [input_override] + [worker_override] * n_cpus + [output_override]:
        config_kwargs = deepcopy(kwargs)
        config_kwargs['config_dict'] = combine_configs(config_kwargs.get('config_dict'),
                                                       common_override,
                                                       worker_overide)
        living_workers.append(multiprocessing.Process(target=Processor, kwargs=config_kwargs))

    for w in living_workers:
        w.start()

    # Check the health / status of the workers every second.
    crashing_down = False
    while len(living_workers):
        time.sleep(1)

        for i, w in enumerate(living_workers):
            if w.exitcode == 0:
                # w is done
                living_workers[i] = None

            elif w.exitcode is None:
                # w is still running
                if crashing_down:
                    w.terminate()
                    living_workers[i] = None

            else:
                # w crashed or was terminated by a signal.
                crashing_down = True
                living_workers[i] = None

        living_workers = [w for w in living_workers if w is not None]

        # TODO: better status line
        print(living_workers, processing_queue.qsize(), output_queue.qsize())

    if crashing_down:
        raise RuntimeError("Pax multiprocessing crashed due to exception in one of the workers")

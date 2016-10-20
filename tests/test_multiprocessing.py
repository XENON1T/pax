import unittest
import shutil
import os
from mock import MagicMock

try:
    import queue
except ImportError:
    import Queue as queue   # flake8: noqa

from pax.parallel import multiprocess_locally
from pax.plugins.io.Queues import PullFromQueue, PushToQueue, NO_MORE_EVENTS, REGISTER_PUSHER, PUSHER_DONE
from pax.datastructure import Event


def fake_events(n):
    result = []
    for i in range(n):
        e = Event(n_channels=1, start_time=0, length=100, sample_duration=10)
        e.event_number = i
        result.append(e)
    return result

class TestMultiprocessing(unittest.TestCase):

    def test_ordered_pull(self):
        # Test pulling from a queue in order. Queue is here just a local (non-multiprocessing) queue
        q = queue.Queue()
        p = PullFromQueue(dict(queue=q, ordered_pull=True), processor=MagicMock())
        events = fake_events(20)
        q.put((2, events[8:]))
        q.put((0, events[:5]))
        q.put((1, events[5:8]))
        q.put((NO_MORE_EVENTS, None))
        for i, e in enumerate(p.get_events()):
            self.assertEqual(e.event_number, i)

    def test_pull_multiple(self):
        q = queue.Queue()
        p = PullFromQueue(dict(queue=q, ordered_pull=True), processor=MagicMock())
        events = fake_events(20)
        q.put((REGISTER_PUSHER, 'gast'))
        q.put((REGISTER_PUSHER, 'gozer'))
        q.put((2, events[8:]))
        q.put((0, events[:5]))
        q.put((PUSHER_DONE, 'gozer'))
        q.put((1, events[5:8]))
        q.put((PUSHER_DONE, 'gast'))
        for i, e in enumerate(p.get_events()):
            self.assertEqual(e.event_number, i)

    def test_push(self):
        # Test pushing to a local (non-multiprocessing) queue
        q = queue.Queue()
        p = PushToQueue(dict(queue=q), processor=MagicMock())

        # Submit a series of fake events, then shut down
        events = fake_events(22)
        for e in events:
            p.write_event(e)
        p.shutdown()

        blocks_out = []
        try:
            while True:
                blocks_out.append(q.get(timeout=1))
        except queue.Empty:
            pass

        # No more events message pushed to queue
        self.assertEqual(blocks_out[-1], (NO_MORE_EVENTS, None))
        blocks_out = blocks_out[:-1]

        # Block ids must be correct
        self.assertEqual([x[0] for x in blocks_out], [0, 1, 2])

        # Block sizes are correct
        self.assertEqual([len(x[1]) for x in blocks_out], [10, 10, 2])

    def test_push_preserveid(self):
        q = queue.Queue()
        p = PushToQueue(dict(queue=q, preserve_ids=True), processor=MagicMock())

        events = fake_events(22)
        for e in events[:5]:
            e.block_id = 0
        for e in events[5:9]:
            e.block_id = 1
        for e in events[9:]:
            e.block_id = 2

        for e in events:
            p.write_event(e)
        p.shutdown()

        blocks_out = []
        try:
            while True:
                blocks_out.append(q.get(timeout=1))
        except queue.Empty:
            pass

        # No more events message pushed to queue
        self.assertEqual(blocks_out[-1], (NO_MORE_EVENTS, None))
        blocks_out = blocks_out[:-1]

        # Block ids must be correct
        self.assertEqual([x[0] for x in blocks_out], [0, 1, 2])

        # Block sizes are correct
        self.assertEqual([len(x[1]) for x in blocks_out], [5, 9-5, 22-9])

    def test_multiprocessing(self):
        multiprocess_locally(n_cpus=2,
                             config_names='XENON100',
                             config_dict=dict(pax=dict(stop_after=10)))
    #
    # def test_process_event_list_multiprocessing(self):
    #     """Take a list of event numbers from a file, and process them on two cores
    #     """
    #     with open('temp_eventlist.txt', mode='w') as outfile:
    #         outfile.write("0\n7\n")
    #     config = {'pax': {'event_numbers_file': 'temp_eventlist.txt',
    #                       'plugin_group_names': ['input', 'output'],
    #                       'output_name': 'test_output',
    #                       'encoder_plugin': None,
    #                       'output': 'Table.TableWriter'},
    #               'Table.TableWriter': {'output_format': 'csv'}}
    #
    #     multiprocess_locally(n_cpus=2, config_names='XENON100', config_dict=config)
    #
    #     # Check we actually wrote two events (and a header row)
    #     self.assertTrue(os.path.exists('test_output'))
    #     self.assertTrue(os.path.exists('test_output/Event.csv'))
    #     with open('test_output/Event.csv') as infile:
    #         self.assertEqual(len(infile.readlines()), 3)
    #
    #     # Cleanup
    #     shutil.rmtree('test_output')
    #     os.remove('temp_eventlist.txt')


if __name__ == '__main__':
    # import logging
    # import sys
    # logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
    unittest.main()

import queue
import time
import heapq

from pax import plugin
from pax.core import MP_STATUS



class QueuePlugin:

    def check_crash(self):
        if self.get_status() == MP_STATUS['crashing']:
            exit('')

    def get_status(self):
        raise NotImplementedError

    def set_status(self):
        raise NotImplementedError

    def push_event_block(self, block_id, event_block):
        raise NotImplementedError

    def pull_event_block(self):
        """Pull event block from queue, or raise queue.Empty if no events queued"""
        raise NotImplementedError


class SharedMemoryQueuePlugin(QueuePlugin):

    def startup(self):
        self.queue = self.config['queue']
        self.status_magic_variable = self.config['status_magic_variable']
        # input_done for regular workers processing_done for output worker
        self.block_heap = []

    def get_status(self):
        return self.status_magic_variable.value

    def set_status(self, status_key):
        self.status_magic_variable.value = MP_STATUS[status_key]

    def push_event_block(self, block_id, events):
        self.queue.put((block_id, events))

    def pull_event_block(self):
        self.queue.get(block=True, timeout=1)



class QueuePullPlugin(plugin.InputPlugin, QueuePlugin):

    def startup(self):
        self.status_if_can_end = MP_STATUS[self.config.get('status_if_can_end', 'input_done')]

    def get_events(self):
        block_heap = self.block_heap
        block_id = -1  # Current block id. Needed for ordered pulling.

        while True:
            current_status = self.get_status()
            can_end = current_status == self.status_if_can_end
            try:
                if self.config['ordered']:
                    # We have to ensure the event blocks are pulled out in order.
                    # If we don't have the block we want yet, keep fetching event blocks from the queue
                    # and push them onto a heap.
                    while not (len(block_heap) and block_heap[0][0] == block_id + 1):
                        self.check_crash()
                        heapq.heappush(block_heap, self.pull_event_block())
                        self.log.debug("Just got a block, heap is now %d blocks long" % len(block_heap))
                        self.log.debug("Earliest block: %d, looking for block %s" % (block_heap[0][0],
                                                                                     block_id + 1))

                    # If we get here, we have the event block we need sitting at the top of the heap
                    block_id, event_block = heapq.heappop(block_heap)

                else:
                    block_id, event_block = self.pull_event_block()

            except queue.Empty:
                if can_end and not len(block_heap):
                    # We're done, no more events!
                    break
                # Queue is empty: wait for more data, then check can_end again
                self.log.debug("Found empty queue, can_end is %s, status is %s" % (can_end, current_status))
                time.sleep(1)
                continue

            try:
                self.log.debug("Now processing block %d" % block_id)
                for i, event in enumerate(event_block):
                    self.check_crash()
                    event.block_id = block_id
                    yield event

            except Exception:
                # Crash occurred during processing: notify everyone else, then die
                self.set_status('crashing')
                raise


class QueuePushPlugin(plugin.OutputPlugin, QueuePlugin):

    def startup(self):
        self.current_block = []
        self.block_size = self.config['pax'].get('event_block_size', 10)

    def write_event(self, event):
        self.current_block.append(event)
        if len(self.current_block) >= self.block_size:
            self.queue.put((event.block_id, event))
import queue
import time
import heapq

from pax import plugin
from pax.core import MP_STATUS


class SharedMemoryQueuePlugin:

    def startup(self):
        self.queue = self.config['queue']
        self.status = self.config['status']
        # input_done for regular workers processing_done for output worker
        self.status_if_can_end = MP_STATUS[self.config.get('status_if_can_end', 'input_done')]
        if self.config['ordered']:
            self.block_heap = []

    def check_crash(self):
        if self.status.value == MP_STATUS['crashing']:
            exit('')

    def push_to_queue(self, event_block, block_id):



class PullSharedMemoryQueue(plugin.InputPlugin, SharedMemoryQueuePlugin):





    def get_events(self):
        block_heap = self.block_heap
        block_id = -1  # Current block id. Needed for ordered pulling.

        while True:
            can_end = self.status_if_can_end

            if self.config['ordered']:
                # We have to ensure the event blocks are pulled out in order.
                try:
                    # If we don't have the block we want yet, keep fetching event blocks from the queue
                    # and push them onto the heap.
                    # Note: if one block takes much longer than the others, this would slurp all blocks
                    # while waiting for the difficult one to come through, potentially exhausting your RAM...
                    while not (len(block_heap) and block_heap[0][0] == block_id + 1):
                        self.check_crash()
                        heapq.heappush(block_heap, self.queue.get(block=True, timeout=1))
                        self.log.debug("Just got a block, heap is now %d blocks long" % len(block_heap))
                        self.log.debug("Earliest block: %d, looking for block %s" % (block_heap[0][0],
                                                                                     block_id + 1))

                except queue.Empty:
                    if can_end and not len(block_heap):
                        # We're done!
                        break
                    # Queue is empty, but either we're not in the processing_done status, or we're waiting for a
                    # block which hasn't arrived on the queue yet. Wait, then try the queue again.
                    time.sleep(1)
                    continue

                # If we get here, we have the event block we need sitting at the top of the heap
                block_id, event_block = heapq.heappop(block_heap)

            else:
                try:
                    block_id, event_block = self.queue.get(block=True, timeout=1)
                except queue.Empty:
                    if can_end:
                        # We're done, no more events!
                        break
                    # Queue is empty: wait for more data, then check can_end again
                    self.log.debug("Found empty queue, can_end is %s, status is %s" % (can_end, self.status.value))
                    time.sleep(1)
                    continue

            try:
                self.log.debug("%s now processing block %d" % (self.worker_id, block_id))
                for i, event in enumerate(event_block):
                    self.check_crash()
                    event.block_id = block_id
                    yield event

            except Exception:
                # Crash occurred during processing: notify everyone else, then die
                self.status.value = MP_STATUS['crashing']
                raise


class PushSharedMemoryQueue(plugin.OutputPlugin, SharedMemoryQueuePlugin):

    def write(self, event):
        # TODO: gather in blocks

        self.queque.put((event.block_id, event))
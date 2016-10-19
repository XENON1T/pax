import queue
import time
import heapq

from pax import plugin


##
# Generic queue handling
##
NO_MORE_EVENTS = "we're out of events! get more funding!"

class QueuePlugin:
    no_more_events = False

    def startup(self):
        self.init_queue()

    def init_queue(self):
        raise NotImplementedError

    def push_queue(self, block_id, event_block):
        raise NotImplementedError

    def get_queue_size(self):
        raise NotImplementedError

    def pull_queue(self):
        """Pull event block from queue, or raise queue.Empty if no events queued"""
        raise NotImplementedError


class QueuePullPlugin(QueuePlugin, plugin.InputPlugin):
    # We may get eventproxies rather than real events
    do_output_check = False

    def startup(self):
        QueuePlugin.startup(self)
        # If we need to order events received from the queue before releasing them, we need a heap
        # NB! If you enable this, you must GUARANTEE no other process will be consuming from this queue
        # (otherwise there will be holes in the event block ids, triggering an infinite wait)
        self.ordered_pull = self.config.get('ordered_pull', True)
        self.block_heap = []

    def get_block(self):
        """Get a block of events from the queue, or raise queue.Empty if no events are available
        """
        if self.no_more_events:
            # There are no more events.
            # There could be stuff left on the queue, but then it's a None = NoMoreEvents message for other consumers.
            raise queue.Empty

        block_id, event_block = self.pull_queue()

        if event_block == NO_MORE_EVENTS:
            # The last event has been popped from the queue. Push None back on the queue for
            # the benefit of other consumers.
            self.no_more_events = True
            self.push_queue(block_id, NO_MORE_EVENTS)
            raise queue.Empty

        return block_id, event_block

    def get_events(self):
        block_heap = self.block_heap
        block_id = -1

        while True:
            try:
                if self.ordered_pull:
                    # We have to ensure the event blocks are pulled out in order.
                    # If we don't have the block we want yet, keep fetching event blocks from the queue
                    # and push them onto a heap.

                    # While the next event we wan't isn't on the block heap, pull blocks from queue into the heap
                    while not (len(block_heap) and block_heap[0][0] == block_id + 1):
                        new_block = self.get_block()
                        heapq.heappush(block_heap, new_block)
                        self.log.debug("Just got block %d, heap is now %d blocks long" % (new_block[0],
                                                                                          len(block_heap)))
                        self.log.debug("Earliest block: %d, looking for block %s" % (block_heap[0][0],
                                                                                     block_id + 1))

                    # If we get here, we have the event block we need sitting at the top of the heap
                    block_id, event_block = heapq.heappop(block_heap)

                else:
                    block_id, event_block = self.get_block()

            except queue.Empty:
                if self.no_more_events and not len(block_heap):
                    self.log.debug("All done!")
                    # We're done, no more events!
                    break
                # The queue is empty so we must wait for the next event / The event we wan't hasn't arrived on the heap.
                self.log.debug("Found empty queue, no more events is %s, len block heap is %s, sleeping for 1 sec" % (
                    self.no_more_events, len(block_heap)))
                time.sleep(1)
                continue

            self.log.debug("Now processing block %d" % block_id)
            for i, event in enumerate(event_block):
                yield event

        self.log.debug("Exited get_events loop")


class QueuePushPlugin(QueuePlugin, plugin.OutputPlugin):
    # We must be allowed to route eventproxies as well as actual events
    do_input_check = False
    do_output_check = False

    def startup(self):
        QueuePlugin.startup(self)

        self.max_queue_blocks = self.config.get('max_queue_blocks', 100)
        self.current_block = []
        self.current_block_id = 0
        self.max_block_size = self.config.get('event_block_size', 10)

    def write_event(self, event):
        self.current_block.append(event)
        if len(self.current_block) == self.max_block_size:
            self.send_block()

    def push_block(self, block_id, event_block):
        while self.get_queue_size() >= self.max_queue_blocks:
            self.log.debug("Max queue size %d reached, waiting to push block")
            time.sleep(1)
        self.push_queue(block_id, event_block)

    def send_block(self):
        self.push_queue(self.current_block_id, self.current_block)
        self.current_block_id += 1
        self.current_block = []

    def shutdown(self):
        self.send_block()
        self.push_queue(self.current_block_id, NO_MORE_EVENTS)


##
# Queue backends
##

class SharedMemoryQueuePlugin():
    """Shared memory queues receive their queue objects (stdlib queue.Queue instances) via the config dict
    """

    def init_queue(self):
        self.queue = self.config['queue']

    def pull_queue(self):
        return self.queue.get(block=True, timeout=1)

    def push_queue(self, block_id, event_block):
        self.queue.put((block_id, event_block))

    def get_queue_size(self):
        return self.queue.qsize()


class PullFromSharedMemoryQueue(SharedMemoryQueuePlugin, QueuePullPlugin):
    pass


class PushToSharedMemoryQueue(SharedMemoryQueuePlugin, QueuePushPlugin):
    pass

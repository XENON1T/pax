try:
    import queue
except ImportError:
    import Queue as queue   # flake8: noqa
import time
import heapq

from pax import plugin
from pax.parallel import NO_MORE_EVENTS


##
# Generic queue handling
##

class QueuePlugin:
    def startup(self):
        self.init_queue()

    def init_queue(self):
        raise NotImplementedError

    def push_queue(self, stuff):
        raise NotImplementedError

    def get_queue_size(self):
        raise NotImplementedError

    def pull_queue(self):
        """Pull event block from queue, or raise queue.Empty if no events queued"""
        raise NotImplementedError


class QueuePullPlugin(QueuePlugin, plugin.InputPlugin):
    # We may get eventproxies rather than real events
    do_output_check = False
    no_more_events = False

    def startup(self):
        QueuePlugin.startup(self)
        # If we need to order events received from the queue before releasing them, we need a heap
        # NB! If you enable this, you must GUARANTEE no other process will be consuming from this queue
        # (otherwise there will be holes in the event block ids, triggering an infinite wait)
        self.ordered_pull = self.config.get('ordered_pull', False)
        self.block_heap = []

    def get_block(self):
        """Get a block of events from the queue, or raise queue.Empty if no events are available
        """
        if self.no_more_events:
            # There are no more events.
            # There could be stuff left on the queue, but then it's a None = NoMoreEvents message for other consumers.
            raise queue.Empty

        result = self.pull_queue()

        if result == NO_MORE_EVENTS:
            # The last event has been popped from the queue. Push None back on the queue for
            # the benefit of other consumers.
            self.no_more_events = True
            self.push_queue(NO_MORE_EVENTS)
            raise queue.Empty

        else:
            block_id, event_block = result
            # Annotate each event with the block id.
            # If the output is a queue push plugin, we need it to use the same block id
            # (else block_id's may get duplicated)
            for e in event_block:
                e.block_id = block_id

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
                self.log.debug("Here's what's on the heap: %s" % block_heap)
                time.sleep(1)
                continue

            self.log.debug("Now processing block %d, %d events" % (block_id, len(event_block)))
            for i, event in enumerate(event_block):
                self.log.debug("Yielding event number %d" % event.event_number)
                yield event

        self.log.debug("Exited get_events loop")


class QueuePushPlugin(QueuePlugin, plugin.OutputPlugin):
    # We must be allowed to route eventproxies as well as actual events
    do_input_check = False
    do_output_check = False

    def startup(self):
        QueuePlugin.startup(self)

        self.max_queue_blocks = self.config.get('max_queue_blocks', 100)
        self.max_block_size = self.config.get('event_block_size', 10)
        self.preserve_ids = self.config.get('preserve_ids', False)
        self.send_no_more_events_signal = self.config.get('send_no_more_events_signal', True)
        self.current_block = []
        if self.preserve_ids:
            # We get events with block_id's already set, and must return them in groups with the same id
            # We can assume the id's come in order, however
            self.current_block_id = None
        else:
            # We're getting events that have never been on a queue before, and can freely assign block id's
            self.current_block_id = 0

    def write_event(self, event):
        if self.preserve_ids:
            if event.block_id != self.current_block_id:
                self.send_block()
                self.current_block_id = event.block_id
            self.current_block.append(event)

        else:
            self.current_block.append(event)
            if len(self.current_block) == self.max_block_size:
                self.send_block()
                self.current_block_id += 1

    def send_block(self):
        """Sends the current block if it has any events in it, then resets the current block to []
        Does NOT change self.current_block_id!
        """
        if len(self.current_block):
            while self.get_queue_size() >= self.max_queue_blocks:
                self.log.debug("Max queue size %d reached, waiting to push block")
                time.sleep(1)
            self.push_queue((self.current_block_id, self.current_block))
        self.current_block = []

    def shutdown(self):
        self.send_block()
        if self.send_no_more_events_signal:
            self.push_queue(NO_MORE_EVENTS)


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

    def push_queue(self, stuff):
        self.queue.put(stuff)

    def get_queue_size(self):
        return self.queue.qsize()


class PullFromSharedMemoryQueue(SharedMemoryQueuePlugin, QueuePullPlugin):
    pass


class PushToSharedMemoryQueue(SharedMemoryQueuePlugin, QueuePushPlugin):
    pass

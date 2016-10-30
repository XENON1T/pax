import time
import heapq

from pax import plugin, utils, exceptions, datastructure
from pax.parallel import queue, RabbitQueue, NO_MORE_EVENTS, REGISTER_PUSHER, PUSHER_DONE, DEFAULT_RABBIT_URI


def get_queue_from_config(config):
    """Given a queueplugin config, get the queue from it
    Yeah, should have maybe made base class with this as only method...
    """
    if 'queue' in config:
        return config['queue']
    elif 'queue_name' in config:
        return RabbitQueue(config['queue_name'],
                           config.get('queue_url', DEFAULT_RABBIT_URI))


class PullFromQueue(plugin.InputPlugin):
    # We may get eventproxies rather than real events
    do_output_check = False
    no_more_events = False

    def startup(self):
        self.queue = get_queue_from_config(self.config)
        # If we need to order events received from the queue before releasing them, we need a heap
        # NB! If you enable this, you must GUARANTEE no other process will be consuming from this queue
        # (otherwise there will be holes in the event block ids, triggering an infinite wait)
        self.ordered_pull = self.config.get('ordered_pull', False)
        self.time_slept_since_last_response = 0
        self.block_heap = []
        self.pushers = []

        # If no message has been received for this amount of seconds, crash.
        self.timeout_after_sec = self.config.get('timeout_after_sec', float('inf'))
        self.max_blocks_on_heap = self.config.get('max_blocks_on_heap', 250)

    def get_block(self):
        """Get a block of events from the queue, or raise queue.Empty if no events are available
        """
        if self.no_more_events:
            # There are no more events.
            # There could be stuff left on the queue, but then it's a None = NoMoreEvents message for other consumers.
            raise queue.Empty

        head, body = self.queue.get(block=True, timeout=1)

        if head == NO_MORE_EVENTS:
            # The last event has been popped from the queue. Push None back on the queue for
            # the benefit of other consumers.
            self.no_more_events = True
            self.log.info("Received no more events message, putting it back on queue for others")
            self.queue.put((NO_MORE_EVENTS, None))
            raise queue.Empty

        elif head == REGISTER_PUSHER:
            # We're in a many-push to one-pull situation.
            # One of the pushers has just announced itself.
            self.pushers.append(body)
            self.log.debug("Registered new pusher: %s" % body)
            return self.get_block()

        elif head == PUSHER_DONE:
            # A pusher just proclaimed it will no longer push events
            self.pushers.remove(body)
            self.log.debug("Removed pusher: %s. %d remaining pushers" % (body, len(self.pushers)))
            if not len(self.pushers):
                # No pushers left, stop processing once there are no more events.
                # This assumes all pushers will register before the first one is done!
                self.queue.put((NO_MORE_EVENTS, None))
            return self.get_block()

        else:
            block_id, event_block = head, body
            # Annotate each event with the block id. This information must be preserved inside the events, because
            # if the output is a queue push plugin, we need it to use the same block id.
            # (else block_id's may get duplicated, causing halts/crashes/mayhem).
            new_block = []
            for e in event_block:
                if isinstance(e, datastructure.EventProxy):
                    # Namedtuples are immutable, so we need to create a new event proxy with the same raw data
                    e2 = datastructure.make_event_proxy(e, data=e.data, block_id=block_id)
                else:
                    e.block_id = block_id
                    e2 = e
                new_block.append(e2)
            event_block = new_block

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
                        if len(block_heap) > self.max_blocks_on_heap:
                            raise exceptions.EventBlockHeapSizeExceededException(
                                "We have received over %d blocks without receiving the next block id (%d) in order. "
                                "Likely one of the block producers has died without telling anyone." % (
                                    self.max_blocks_on_heap, block_id + 1))
                        self.log.debug("Just got block %d, heap is now %d blocks long" % (
                            new_block[0], len(block_heap)))
                        self.log.debug("Earliest block: %d, looking for block %s" % (block_heap[0][0], block_id + 1))

                    # If we get here, we have the event block we need sitting at the top of the heap
                    block_id, event_block = heapq.heappop(block_heap)

                    assert block_id >= 0

                else:
                    block_id, event_block = self.get_block()

                self.time_slept_since_last_response = 0

            except queue.Empty:
                if self.no_more_events and not len(block_heap):
                    self.log.debug("All done!")
                    # We're done, no more events!
                    break

                # The queue is empty so we must wait for the next event / The event we wan't hasn't arrived on the heap.
                self.log.debug("Found empty queue, no more events is %s, len block heap is %s, sleeping for 1 sec" % (
                    self.no_more_events, len(block_heap)))
                if len(block_heap) > 0.3 * self.max_blocks_on_heap:
                    self.log.warning("%d blocks on heap, will crash if more than %d" % (len(block_heap),
                                                                                        self.max_blocks_on_heap))

                time.sleep(1)
                self.processor.timer.last_t = time.time()    # Time spent idling shouldn't count for the timing report
                self.time_slept_since_last_response += 1
                if self.time_slept_since_last_response > self.timeout_after_sec:
                    raise exceptions.QueueTimeoutException(
                        "Waited for more than %s seconds to receive events; "
                        "lost confidence they will ever come." % self.timeout_after_sec)

                continue

            self.log.debug("Now processing block %d, %d events" % (block_id, len(event_block)))
            for i, event in enumerate(event_block):
                self.log.debug("Yielding event number %d" % event.event_number)
                yield event

        self.log.debug("Exited get_events loop")

    def shutdown(self):
        if hasattr(self.queue, 'close'):
            self.queue.close()


class PushToQueue(plugin.OutputPlugin):
    # We must be allowed to route eventproxies as well as actual events
    do_input_check = False
    do_output_check = False

    def startup(self):
        self.queue = get_queue_from_config(self.config)
        self.max_queue_blocks = self.config.get('max_queue_blocks', 100)
        self.max_block_size = self.config.get('event_block_size', 10)
        self.preserve_ids = self.config.get('preserve_ids', False)
        self.many_to_one = self.config.get('many_to_one', False)

        # If we can't push a message due to a full queue for more than this number of seconds, crash
        # since probably the process responsible for pulling from the queue has died.
        self.timeout_after_sec = self.config.get('timeout_after_sec', float('inf'))

        if self.many_to_one:
            # Generate random name and tell the puller we're in town
            self.pusher_name = utils.randomstring(20)
            self.queue.put((REGISTER_PUSHER, self.pusher_name))

        self.current_block = []
        self.current_block_id = 0

    def write_event(self, event):
        if self.preserve_ids:
            assert event.block_id >= 0    # Datastructure default is -1, if we see that here we are in big doodoo
            if event.block_id != self.current_block_id:
                # A change in the block id must always be just after sending a block
                # otherwise the pax chain is using inconsistent event block sizes
                assert len(self.current_block) == 0
                self.current_block_id = event.block_id

        else:
            # We have to set the block id's.
            event.block_id = self.current_block_id

        self.current_block.append(event)
        # Send events once the max block size is reached. Do not wait until event with next id arrives:
        # that can take forever if we're doing low-rate processing with way to much cores.
        if len(self.current_block) == self.max_block_size:
            self.send_block()

            # If we're setting the id's, from now on, we have to set with the next number
            if not self.preserve_ids:
                self.current_block_id += 1

    def send_block(self):
        """Sends the current block if it has any events in it, then resets the current block to []
        Does NOT change self.current_block_id!
        """
        seconds_slept_with_queue_full = 0
        if len(self.current_block):
            while self.queue.qsize() >= self.max_queue_blocks:
                self.log.info("Max queue size %d reached, waiting to push block")
                seconds_slept_with_queue_full += 1
                time.sleep(1)
                if seconds_slept_with_queue_full >= self.timeout_after_sec:
                    raise exceptions.QueueTimeoutException(
                        "Blocked from pushing to the queue for more than %s seconds; "
                        "lost confidence we will ever be able to." % self.timeout_after_sec)
            self.queue.put((self.current_block_id, self.current_block))
        self.current_block = []

    def shutdown(self):
        self.send_block()
        if self.many_to_one:
            self.queue.put((PUSHER_DONE, self.pusher_name))
        else:
            self.queue.put((NO_MORE_EVENTS, None))
        if hasattr(self.queue, 'close'):
            self.queue.close()

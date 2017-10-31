import numpy as np
from pax.trigger import TriggerPlugin


class InvertTrigger(TriggerPlugin):
    """Inverts the trigger, building events exactly where they weren't triggered.
    """
    def startup(self):
        self.last_batch_end_time = 0

    def process(self, data):
        data.event_ranges = np.concatenate(([self.last_batch_end_time],
                                            data.event_ranges.ravel(),
                                            [data.last_time_searched])).reshape(data.event_ranges.shape)
        self.last_batch_end_time = data.last_time_searched

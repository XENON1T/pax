import numpy as np
from pax.trigger import TriggerPlugin


class SaveEverythingTrigger(TriggerPlugin):
    """Segment the data in consecutive ranges of event_separation duration.
    Optionally only make such events between start_at and stop_at.
    NB: This will save a lot of data!!
    """

    def startup(self):
        self.event_size = self.config['event_separation']
        self.last_event_time = self.config.get('start_at', 0)   # Last event stop time emitted
        self.stop_at = self.config.get('stop_at', float('inf'))

    def process(self, data):
        event_ranges = []
        while self.last_event_time < min(self.stop_at,
                                         data.last_time_searched - self.event_size):
            event_ranges.append([self.last_event_time,
                                 self.last_event_time + self.event_size])
            self.last_event_time += self.event_size
        data.event_ranges = np.array(event_ranges, dtype=np.int64)

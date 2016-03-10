"""A very basic trigger: trigger at a fixed interval, regardless of the data"""
import numpy as np

from pax.trigger_modules.base import TriggerModule


class TimedTrigger(TriggerModule):
    numeric_id = 1

    next_time = 0

    def get_event_ranges(self, signals):
        events = []
        while self.trigger.last_time_searched > self.next_time + self.config['event_length']:
            events.append([self.next_time, self.next_time + self.config['event_length']])
            self.next_time += self.config['trigger_interval']

        yield from self.make_events(events, signals)

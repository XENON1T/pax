import itertools
import numpy as np

from pax import plugin, datastructure


class DummyInput(plugin.InputPlugin):
    """A dummy input plugin that yields a single empty event, then exists"""

    def get_events(self):
        yield datastructure.Event.empty_event()


class GarbageInput(plugin.InputPlugin):
    """Produce infinitely many events with nonsense data. Useful to diagnose memory leaks"""

    def get_events(self):
        for event_number in itertools.count():
            event = datastructure.Event.empty_event()
            event.event_number = event_number
            event.pulses = [datastructure.Pulse(left=0,
                                                right=100,
                                                channel=0,
                                                raw_data=np.random.randint(0, 16000, size=100))
                            for _ in range(self.config.get('garbage_pulses_per_event', 10000))]
            yield event


class DummyOutput(plugin.OutputPlugin):
    "Stores last event in self.last_event, useful for testing"

    def write_event(self, event):
        self.last_event = event


class DummyTransform(plugin.TransformPlugin):
    pass


class DummyTransform2(plugin.TransformPlugin):
    pass


class DummyTransform3(plugin.TransformPlugin):
    pass

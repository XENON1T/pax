from pax import plugin
from pax.tasks import process


class SubmitEvent(plugin.TransformPlugin):

    def startup(self):
        self.cache = []

    def process_event(self, event):
        self.log.info("Submitting event")

        primer = {'n_channels': self.config['n_channels'],
                  'start_time': event.start_time,
                  'sample_duration': event.sample_duration,
                  'stop_time': event.stop_time
                  }
        self.cache.append(primer)
        if len(self.cache) > 10:
            process.delay(self.cache)
            self.cache = []
        return event

from pax import plugin


class BlankHEEvents(plugin.TransformPlugin):
    """Plugin that destroys all pulses in high-energy events. Useful to apply a software HE-prescale after the fact,
    e.g. to create "minimum bias" raw datasets.
    """

    def transform_event(self, event):
        if event.n_pulses > self.config['max_n_pulses']:
            event.pulses = []
        return event

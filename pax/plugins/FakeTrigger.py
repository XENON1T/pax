import logging
import numpy as np

from pax import plugin, trigger


class FakeTrigger(plugin.TransformPlugin):
    """Runs the XENON1T trigger over events (which were already built earlier)
    This does not change how many events are seen, but just adds trigger signal annotations to the events.
    Useful for diagnostic purposes -- DO NOT USE in actual data processing
    """
    debug = False

    def startup(self):
        if self.debug:
            # Make sure the trigger logger is set to debug loglevel
            logging.getLogger('Trigger').setLevel(logging.DEBUG)
            self.log.setLevel(logging.DEBUG)

        trig_conf = self.processor.config['Trigger']

        # To ensure we get all signals from the event, trigger on all signals
        # TODO: once signals outside events can be retrieved, drop this hack
        trig_conf['every_signal_triggers'] = True

        self.trigger = trigger.Trigger(trig_conf)

        # The trigger must always think no more data is coming,
        # so every time we get trigger ranges it will process its complete buffer
        self.trigger.more_data_is_coming = False

    def transform_event(self, event):
        # The pulses are kept in order of (channel, left) in pax, but the trigger expects time-sorted pulses
        times = np.array([p.left for p in event.pulses], dtype=np.int)
        times.sort()
        times *= self.config['sample_duration']

        self.trigger.add_new_data(times=times,
                                  last_time_searched=event.stop_time)

        # Accumulate all signals from the trigger (don't care about actual event ranges)
        sigs = [s for _, s in self.trigger.get_trigger_ranges()]
        sigs.append(self.trigger.signals_beyond_events)

        if not len(sigs):
            self.log.info("Trigger returned no signals!")
        else:
            event.trigger_signals = np.concatenate(sigs)
            self.log.info("Trigger returned %d signals" % len(event.trigger_signals))

        return event

    def shutdown(self):
        self.log.info(self.trigger.get_end_of_run_info())

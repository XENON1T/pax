import logging
import numpy as np

from pax import plugin, trigger, units


class FakeTrigger(plugin.TransformPlugin):
    """Runs the XENON1T trigger over events (which were already built earlier)
    This does not change how many events are seen, but just adds trigger signal annotations to the events.
    Useful for diagnostic purposes -- DO NOT USE in actual data processing
    """
    debug = True

    def startup(self):
        if self.debug:
            self.log.setLevel(logging.DEBUG)
            # Make sure the trigger logger is set to debug loglevel
            logging.getLogger('Trigger').setLevel(logging.DEBUG)
            # Make sure the trigger plugins' loggers are set to debug loglevel
            for pname in self.processor.config['Trigger']['trigger_plugins']:
                logging.getLogger(pname).setLevel(logging.DEBUG)

        # Add "pmts" to config since trigger requires it. See issue #300.
        self.processor.config['DEFAULT']['pmts'] = [
            dict(pmt_position=0,
                 digitizer=dict(module=0, channel=0))
        ]

        self.trigger = trigger.Trigger(self.processor.config)

    def transform_event(self, event):
        # The pulses are kept in order of (channel, left) in pax, but the trigger expects time-sorted pulses
        times = np.array([p.left for p in event.pulses], dtype=np.int)
        times.sort()
        times *= self.config['sample_duration']

        # Accumulate signals from the main trigger (don't care about actual event range for now).
        # Always let trigger think this is the last data: ensures all data gets used
        # TODO: Get signals outside event out directly somehow. Or not. This is a test plugin anyway.
        sigs = [s for _, s in self.trigger.run(start_times=times,
                                               last_time_searched=1 * units.s,
                                               last_data=True)]

        if not len(sigs):
            self.log.info("Trigger returned no signals / events!")
        else:
            event.trigger_signals = np.concatenate(sigs)
            self.log.info("Trigger returned %d signals" % len(event.trigger_signals))

        return event

    def shutdown(self):
        self.log.info(self.trigger.shutdown())

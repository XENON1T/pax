import numpy as np

from pax import plugin,  datastructure, exceptions


class SortPulses(plugin.TransformPlugin):
    """
    Sorts pulses by channel, then left
    """

    def transform_event(self, event):
        event.pulses = sorted(event.pulses, key=lambda p: (p.channel, p.left))
        return event


class ConcatenateAdjacentPulses(plugin.TransformPlugin):
    """Concatenates directly adjacent pulses in the same channel
    This ensures baselines / noise levels won't get recalculated in the middle
    of some data region (possibly containing hits).
    Assumes pulses are already sorted by channel, then left. Use SortPulses if this is not automatic.
    """

    def transform_event(self, event):
        if len(event.pulses) < 2:
            return event

        good_pulses = []
        last_pulse = 0  # Keep track of last pulse not yet merged into previous one

        for pulse_i in range(1, len(event.pulses)):
            prev_pulse = event.pulses[last_pulse]
            pulse = event.pulses[pulse_i]

            if pulse.channel == prev_pulse.channel and pulse.left == prev_pulse.right + 1:
                # Pulse is directly adjacent to previous one in same channel: merge it
                self.log.debug("Concatenating adjacent DAQ pulses %d-%d and %d-%d in channel %s" % (
                    prev_pulse.left, prev_pulse.right, pulse.left, pulse.right, pulse.channel))
                prev_pulse.right = pulse.right
                prev_pulse.raw_data = np.concatenate((prev_pulse.raw_data, pulse.raw_data))
                # If there are no pulses after this, add the merged pulse to the good pulses list
                if pulse_i == len(event.pulses) - 1:
                    good_pulses.append(prev_pulse)
                # Else we need to keep checking: maybe the next pulse is directly adjacent to the new merged pulse
                # hence we don't advance the last_pulse counter

            else:
                # Don't concatenate this pulse to previous one
                # We can add the previous pulse to the list of good pulses
                good_pulses.append(prev_pulse)
                # ... and advance the last_pulse counter
                last_pulse = pulse_i
                # If there are no pulses after this, add the current pulse to the good pulses list
                if pulse_i == len(event.pulses) - 1:
                    good_pulses.append(pulse)

        event.pulses = good_pulses
        return event


class CheckBoundsAndCount(plugin.TransformPlugin):
    """Check if pulses extend beyond event bounds
    If so, truncate them or raise exception, depending on config
    """

    def startup(self):
        self.truncate_pulses_partially_outside = self.config.get('truncate_pulses_partially_outside', False)
        self.allow_pulse_completely_outside = self.config.get('allow_pulse_completely_outside', False)

    def transform_event(self, event):

        # Sanity check for sample_duration
        if not self.config['sample_duration'] == event.sample_duration:
            raise ValueError('Event %s quotes sample duration = %s ns, but sample_duration is set to %s!' % (
                event.event_number, event.sample_duration, self.config['sample_duration']))

        event_length = event.length()
        pulses_to_ignore = []

        for pulse_i, pulse in enumerate(event.pulses):

            start_index = pulse.left
            length = pulse.length
            end_index = pulse.right
            channel = pulse.channel

            ##
            #  Pulse bounds checking / truncation (see issue 43)
            ##
            overhang = end_index - (event_length - 1)

            if start_index < 0 or end_index < 0 or overhang > 0:

                # If completely outside, mark as to-be-ignored with warning, or give error, according to config
                # to-be-ignored pulses are removed later (can't do here as we're iterating over pulses)
                if overhang >= length or start_index <= -length or end_index < 0:
                    text = ('Pulse %s in channel %s (%s-%s) is entirely outside '
                            'event bounds (%s-%s)! See issue #43.' % (pulse_i, channel, start_index, end_index,
                                                                      0, event_length - 1))
                    if self.allow_pulse_completely_outside:
                        self.log.warning(text)
                        pulses_to_ignore.append(pulse_i)
                        continue
                    else:
                        raise exceptions.PulseBeyondEventError(text)

                pulse_wave = pulse.raw_data

                # If partially outside, truncate with debug message, or give error, according to config
                message = 'Pulse %s in channel %s (%s-%s) is partially outside ' \
                          'event bounds (%s-%s). See issue #43' % (
                              pulse_i, channel, start_index, end_index, 0, event_length - 1)

                if not self.truncate_pulses_partially_outside:
                    raise exceptions.PulseBeyondEventError(message)
                self.log.debug(message)

                # Truncate the pulse. Remember start_index < 0!
                if start_index < 0:
                    pulse_wave = pulse_wave[-start_index:]
                    start_index = 0
                if overhang > 0:
                    pulse_wave = pulse_wave[:-overhang]
                    end_index = event_length - 1

                # Update the pulse data, so hit finder won't look at old un-truncated pulse
                event.pulses[pulse_i] = datastructure.Pulse(left=start_index,
                                                            right=end_index,
                                                            channel=channel,
                                                            raw_data=pulse_wave)

        # Remove the to-be-ignored-pulses
        event.pulses = [p for p_i, p in enumerate(event.pulses) if p_i not in pulses_to_ignore]

        # Count the number of pulses per channel. Must be done at the end, since pulses can be ignored (see above)
        for p in event.pulses:
            event.n_pulses_per_channel[p.channel] += 1
        event.n_pulses = event.n_pulses_per_channel.sum()

        return event


# Alias for old configs
CheckBounds = CheckBoundsAndCount

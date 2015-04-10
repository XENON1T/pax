from pax import plugin,  datastructure, exceptions


class CheckBounds(plugin.TransformPlugin):
    """Check if pulses extend beyond event bounds
    If so, truncate them or raise exception, depending on config
    """

    def startup(self):
        self.truncate_pulses_partially_outside = self.config.get('truncate_pulses_partially_outside', False)

    def transform_event(self, event):

        # Sanity check for sample_duration
        if not self.config['sample_duration'] == event.sample_duration:
            raise ValueError('Event %s quotes sample duration = %s ns, but sample_duration is set to %s!' % (
                event.event_number, event.sample_duration, self.config['sample_duration']))

        event_length = event.length()

        for occ_i, occ in enumerate(event.pulses):

            start_index = occ.left
            length = occ.length
            end_index = occ.right
            channel = occ.channel

            ##
            #  Pulse bounds checking / truncation (see issue 43)
            ##

            overhang = end_index - (event_length - 1)

            if start_index < 0 or end_index < 0 or overhang > 0:

                # Always throw error if pulse is completely outside event
                if overhang >= length or start_index <= -length or end_index < 0:
                    raise exceptions.PulseBeyondEventError(
                        'Pulse %s in channel %s (%s-%s) is entirely outside '
                        'event bounds (%s-%s)! See issue #43.' % (
                            occ_i, channel, start_index, end_index, 0, event_length - 1))

                pulse_wave = occ.raw_data

                # If partially outside, truncate with warning, or give error, according to config
                message = 'Pulse %s in channel %s (%s-%s) is partially outside ' \
                          'event bounds (%s-%s). See issue #43' % (
                              occ_i, channel, start_index, end_index, 0, event_length - 1)

                if not self.truncate_pulses_partially_outside:
                    raise exceptions.PulseBeyondEventError(message)
                self.log.warning(message)

                # Truncate the pulse. Remember start_index < 0!
                if start_index < 0:
                    pulse_wave = pulse_wave[-start_index:]
                    start_index = 0
                if overhang > 0:
                    pulse_wave = pulse_wave[:-overhang]
                    end_index = event_length - 1

                # Update the pulse data, so hit finder won't look at old un-truncated occ
                event.pulses[occ_i] = occ = datastructure.Pulse(
                    left=start_index,
                    right=end_index,
                    channel=channel,
                    raw_data=pulse_wave
                )

        return event

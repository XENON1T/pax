from pax import plugin,  datastructure, exceptions


class CheckBounds(plugin.TransformPlugin):
    """Check if pulses extend beyond event bounds
    If so, truncate them or raise exception, depending on config
    """

    def startup(self):
        self.truncate_occurrences_partially_outside = self.config.get('truncate_occurrences_partially_outside', False)

    def transform_event(self, event):

        # Sanity check for sample_duration
        if not self.config['sample_duration'] == event.sample_duration:
            raise ValueError('Event %s quotes sample duration = %s ns, but sample_duration is set to %s!' % (
                event.event_number, event.sample_duration, self.config['sample_duration']))

        event_length = event.length()

        for occ_i, occ in enumerate(event.occurrences):

            start_index = occ.left
            length = occ.length
            end_index = occ.right
            channel = occ.channel

            ##
            #  Pulse bounds checking / truncation (see issue 43)
            ##

            overhang = end_index - (event_length - 1)

            if start_index < 0 or end_index < 0 or overhang > 0:

                # Always throw error if occurrence is completely outside event
                if overhang >= length or start_index <= -length or end_index < 0:
                    raise exceptions.OccurrenceBeyondEventError(
                        'Occurrence %s in channel %s (%s-%s) is entirely outside '
                        'event bounds (%s-%s)! See issue #43.' % (
                            occ_i, channel, start_index, end_index, 0, event_length - 1))

                occurrence_wave = occ.raw_data

                # If partially outside, truncate with warning, or give error, according to config
                message = 'Occurrence %s in channel %s (%s-%s) is partially outside ' \
                          'event bounds (%s-%s). See issue #43' % (
                              occ_i, channel, start_index, end_index, 0, event_length - 1)

                if not self.truncate_occurrences_partially_outside:
                    raise exceptions.OccurrenceBeyondEventError(message)
                self.log.warning(message)

                # Truncate the occurrence. Remember start_index < 0!
                if start_index < 0:
                    occurrence_wave = occurrence_wave[-start_index:]
                    start_index = 0
                if overhang > 0:
                    occurrence_wave = occurrence_wave[:-overhang]
                    end_index = event_length - 1

                # Update the occurrence data, so hit finder won't look at old un-truncated occ
                event.occurrences[occ_i] = occ = datastructure.Occurrence(
                    left=start_index,
                    right=end_index,
                    channel=channel,
                    raw_data=occurrence_wave
                )

        return event

from pax import plugin


class SkipEventsWithRecord(plugin.SelectionPlugin):
    """Stop processing for events that meet a condition, but write their numbers to a text file."""

    def startup(self):
        self.skipped_event_numbers_file = open(self.config['skipped_event_numbers_file'], 'w')

    def test_event(self, event):
        """Test event against blinding conditions.
        """

        # Should we skip this event?
        should_we_skip = self.should_we_skip_event(event)

        # Write this event number to the skipped events file, then stop processing this event
        if should_we_skip:
            self.skipped_event_numbers_file.write("%d\n" % event.event_number)
            return False

        # Not skipped: continue processing
        return True

    def shutdown(self):
        self.skipped_event_numbers_file.close()

    def should_we_skip_event(self, event):
        raise NotImplementedError


class SkipEvenEvents(SkipEventsWithRecord):
    """Skips and records numbers for all even events"""

    def should_we_skip_event(self, event):
        return event.event_number % 2 == 0


class SkipEventNumbersFromFile(plugin.SelectionPlugin):
    """Skips event numbers specified in a newline-delimited text file"""

    def startup(self):
        with open(self.config['events_to_skip_file'], mode='r') as f:
            self.config['events_to_skip'] = list(map(int, f.readlines()))

    def test_event(self, event):
        # Continue only if event number not in events_to_skip
        return event.event_number not in self.config['events_to_skip']

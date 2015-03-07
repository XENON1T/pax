import glob
import os

from pax import core, plugin


class InputFromFolder(plugin.InputPlugin):

    ##
    # Base class methods you don't need to override
    ##
    def startup(self):
        self.raw_data_files = []
        self.current_file_number = None
        self.current_filename = None
        self.current_first_event = None
        self.current_last_event = None

        input_name = core.data_file_name(self.config['input_name'])
        if not os.path.exists(input_name):
            raise ValueError("Can't read from %s: it does not exist!" % input_name)

        if input_name.endswith('.' + self.file_extension):
            self.log.debug("InputFromFolder: Single file mode")
            self.init_file(input_name)
        else:
            self.log.debug("InputFromFolder: Directory mode")

            file_names = glob.glob(os.path.join(input_name, "*." + self.file_extension))
            file_names.sort()

            self.log.debug("InputFromFolder: Found these files: %s", str(file_names))

            if len(file_names) == 0:
                raise ValueError("InputFromFolder: No %s files found in input directory %s!" % (self.file_extension,
                                                                                                input_name))

            for fn in file_names:
                self.init_file(fn)

        # Select the first file
        self.select_file(0)

        # Set the number of total events
        self.number_of_events = sum([fr['last_event'] - fr['first_event'] + 1
                                     for fr in self.raw_data_files])

    def init_file(self, filename):
        """Find out the first and last event contained in filename
        Appends {'filename': ..., 'first_event': ..., 'last_event':...} to self.raw_data_files
        """
        first_event, last_event = self.get_first_and_last_event_number(filename)
        self.log.debug("InputFromFolder: Initializing %s", filename)
        self.raw_data_files.append({'filename': filename,
                                    'first_event': first_event,
                                    'last_event': last_event})

    def select_file(self, i):
        """Selects the ith file from self.raw_data_files for reading
        Will be called by get_single_event (and once by startup)
        """
        if self.current_file_number is not None:
            self.close_current_file()
        if i < 0 or i >= len(self.raw_data_files):
            raise RuntimeError("Invalid file index %s: %s files loaded" % (i,
                                                                           len(self.raw_data_files)))

        self.log.info("InputFromFolder: Selecting file %s "
                      "(number %d/%d in folder) for reading" % (self.current_filename,
                                                                i + 1,
                                                                len(self.raw_data_files)))

        self.current_file_number = i
        f_info = self.raw_data_files[i]
        self.current_filename = f_info['filename']
        self.current_first_event = f_info['first_event']
        self.current_last_event = f_info['last_event']

        self.start_to_read_file(self.current_filename)

    def shutdown(self):
        self.close_current_file()

    def get_events(self):
        """Iterate through all events in the file / folder"""
        for file_i, file_info in enumerate(self.raw_data_files):
            self.select_file(file_i)
            for event in self.get_all_events_in_current_file():
                yield event

    @plugin.BasePlugin._timeit
    def get_single_event(self, event_number):
        """Get a single event, automatically selecting the right file"""
        if not self.current_first_event <= event_number <= self.current_last_event:
            # Time to open a new file!
            self.log.debug("InputFromFolder: Event %d is not in the current file (%d-%d), "
                           "so opening a new file..." % (event_number,
                                                         self.current_first_event,
                                                         self.current_last_event))
            for i, file_info in enumerate(self.raw_data_files):
                if file_info['first_event'] <= event_number <= file_info['last_event']:
                    self.select_file(i)
                    break
            else:
                raise ValueError("None of the loaded files contains event %d!" % event_number)

        return self.get_single_event_in_current_file(event_number - self.current_first_event)

    ##
    # Child class should override these
    ##

    file_extension = '.xed'

    def get_first_and_last_event_number(self, filename):
        """Return the first and last event number in file specified by filename"""
        raise NotImplementedError()

    def start_to_read_file(self, filename):
        """Opens the file specified by filename for reading"""
        raise NotImplementedError()

    def close_current_file(self):
        """Close the currently open file"""
        pass

    ##
    # Override this if you support random access
    ##
    def get_single_event_in_current_file(self, event_position):
        """Uses iteration to emulate random access to events
        Note -- this takes the event POSITION in the file, not the absolute event number!
        """
        for event_i, event in enumerate(self.get_all_events_in_current_file()):
            if event_i == event_position:
                return event
        raise RuntimeError("Current file has no %d th event, and some check didn't pick this up.\n"
                           "Either the file is very nasty, or the reader is bugged!" % event_position)

    ##
    # Override this if you DO NOT support random access, or if random access is slower
    ##
    def get_all_events_in_current_file(self):
        """Uses random access to iterate over all events"""
        for event_i in range(self.current_first_event, self.current_last_event + 1):
            yield self.get_single_event_in_current_file(event_i - self.current_first_event)

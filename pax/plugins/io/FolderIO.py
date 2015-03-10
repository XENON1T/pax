import glob
import os
import time

from pax import core, plugin


class InputFromFolder(plugin.InputPlugin):
    """Read from a folder containing several small files, each containing with events"""

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

        self.current_file_number = i
        f_info = self.raw_data_files[i]
        self.current_filename = f_info['filename']
        self.current_first_event = f_info['first_event']
        self.current_last_event = f_info['last_event']

        self.log.info("InputFromFolder: Selecting file %s "
                      "(number %d/%d in folder) for reading" % (self.current_filename,
                                                                i + 1,
                                                                len(self.raw_data_files)))

        self.start_to_read_file(self.current_filename)

    def shutdown(self):
        self.close_current_file()

    def get_events(self):
        """Iterate through all events in the file / folder"""
        self.ts = time.time()   # We must keep track of time ourselves, BasePlugin.timeit is a function decorator
        for file_i, file_info in enumerate(self.raw_data_files):
            self.select_file(file_i)
            for event in self.get_all_events_in_current_file():
                self.total_time_taken += (time.time() - self.ts) * 1000
                yield event
                self.ts = time.time()       # Restart clock

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

    # If reaading in from a folder-of-files format not written by FolderIO,
    # you'll probably have to overwrite this. (e.g. XED does)
    def get_first_and_last_event_number(self, filename):
        """Return the first and last event number in file specified by filename"""
        _, _, first_event, last_event = os.path.splitext(os.path.basename(filename))[0].split('-')
        return int(first_event), int(last_event)

    ##
    # Child class should override these
    ##

    file_extension = None

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


class WriteToFolder(plugin.OutputPlugin):
    """Write to a folder containing several small files, each containing a <= a fixed number of events"""

    def startup(self):
        self.events_per_file = self.config['events_per_file']
        self.first_event_in_current_file = None
        self.last_event_written = None

        self.output_dir = self.config['output_name']
        if os.path.exists(self.output_dir):
            raise ValueError("Output directory %s already exists, can't write your %ss there!" % (
                self.output_dir, self.file_extension))
        else:
            os.mkdir(self.output_dir)

        self.tempfile = os.path.join(self.output_dir, 'temp.' + self.file_extension)

    def open_new_file(self, first_event_number):
        """Opens a new file, closing any old open ones"""
        if self.last_event_written is not None:
            self.close_current_file()
        self.first_event_in_current_file = first_event_number
        self.events_written_to_current_file = 0
        self.start_writing_file(filename=self.tempfile)

    def write_event(self, event):
        """Write one more event to the folder, opening/closing files as needed"""
        if self.last_event_written is None \
                or self.events_written_to_current_file >= self.events_per_file:
            self.open_new_file(first_event_number=event.event_number)

        self.write_event_to_current_file(event)

        self.events_written_to_current_file += 1
        self.last_event_written = event.event_number

    def close_current_file(self):
        """Closes the currently open file, if there is one"""
        if self.last_event_written is None:
            self.log.info("You didn't write any events... Did you crash pax?")
            return

        self.stop_writing_current_file()

        # Rename the temporary file to reflect the events we've written to it
        os.rename(self.tempfile,
                  os.path.join(self.output_dir,
                               '%s-%d-%06d-%06d.%s' % (self.config['tpc_name'],
                                                       self.config['run_number'],
                                                       self.first_event_in_current_file,
                                                       self.last_event_written,
                                                       self.file_extension)))

    def shutdown(self):
        self.close_current_file()

    ##
    # Child class should override these
    ##

    file_extension = None

    def start_writing_file(self, filename):
        raise NotImplementedError

    def write_event_to_current_file(self, event):
        raise NotImplementedError

    def stop_writing_current_file(self):
        raise NotImplementedError

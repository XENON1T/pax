"""I/O plugin base classes for input to/from folders or zipfiles
"""
import glob
import zlib
import os
import shutil

from bson import json_util

from six.moves import input
from pax import utils, plugin
from pax.datastructure import EventProxy


class InputFromFolder(plugin.InputPlugin):
    """Read from a folder containing several small files, each containing events"""

    ##
    # Base class methods you don't need to override
    ##
    def startup(self):
        self.raw_data_files = []
        self.current_file_number = None
        self.current_filename = None
        self.current_first_event = None
        self.current_last_event = None

        input_name = utils.data_file_name(self.config['input_name'])
        if not os.path.exists(input_name):
            raise ValueError("Can't read from %s: it does not exist!" % input_name)

        if not os.path.isdir(input_name):
            if not input_name.endswith('.' + self.file_extension):
                self.log.error("input_name %s does not end "
                               "with the expected file extension %s" % (input_name,
                                                                        self.file_extension))
                return
            self.log.debug("InputFromFolder: Single file mode")
            self.init_file(input_name)

        else:
            self.log.debug("InputFromFolder: Directory mode")
            file_names = glob.glob(os.path.join(input_name, "*." + self.file_extension))
            # Remove the pax_info.json file from the file list-- the JSON I/O will thank us
            file_names = [fn for fn in file_names if not fn.endswith('pax_info.json')]
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
        self.number_of_events = sum([fr['n_events'] for fr in self.raw_data_files])

    def init_file(self, filename):
        """Find out the first and last event contained in filename
        Appends {'filename': ..., 'first_event': ..., 'last_event':..., 'n_events':...} to self.raw_data_files
        """
        first_event, last_event, n_events = self.get_event_number_info(filename)
        self.log.debug("InputFromFolder: Initializing %s", filename)
        self.raw_data_files.append({'filename': filename,
                                    'first_event': first_event,
                                    'last_event': last_event,
                                    'n_events': n_events})

    def select_file(self, i):
        """Selects the ith file from self.raw_data_files for reading
        Will be called by get_single_event (and once by startup)
        """
        if self.current_file_number is not None:
            self.close()
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

        self.open(self.current_filename)
        self.event_numbers_in_current_file = self.get_event_numbers_in_current_file()

    def shutdown(self):
        # hasattr check is needed to prevent extra error if pax crashes before the plugin runs
        if hasattr(self, 'current_file'):
            self.close()

    def get_events(self):
        """Iterate through all events in the file / folder"""
        # We must keep track of time ourselves, BasePlugin.timeit is a function decorator,
        # so it won't work well with generators like get_events for an input plugin
        for file_i, file_info in enumerate(self.raw_data_files):
            if self.current_file_number != file_i:
                self.select_file(file_i)
            for event in self.get_all_events_in_current_file():
                yield event

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
                raise ValueError("None of the loaded files contains event %d! "
                                 "Available event ranges: %s" % (event_number, [(q['first_event'], q['last_event'])
                                                                                for q in self.raw_data_files]))

        if event_number not in self.event_numbers_in_current_file:
            raise ValueError("Event %d does not exist in the file containing events %d - %d!\n"
                             "Event numbers which do exist in file: %s" % (event_number,
                                                                           self.current_first_event,
                                                                           self.current_last_event,
                                                                           self.event_numbers_in_current_file))

        return self.get_single_event_in_current_file(event_number)

    # If reading in from a folder-of-files format not written by FolderIO,
    # you'll probably have to overwrite this. (e.g. ReadXED does)
    def get_event_number_info(self, filename):
        """Return the first, last and total event numbers in file specified by filename"""
        stuff = os.path.splitext(os.path.basename(filename))[0].split('-')
        if len(stuff) == 4:
            # Old format, which didn't have an event numbers field... progress bar will be off...
            _, _, first_event, last_event = stuff
            return int(first_event), int(last_event), int(last_event) - int(first_event) + 1
        elif len(stuff) == 5:
            _, _, first_event, last_event, n_events = stuff
            return int(first_event), int(last_event), int(n_events)
        else:
            raise ValueError("Invalid file name: %s. Should be tpcname-something-firstevent-lastevent-nevents.%s" % (
                filename, self.file_extension))

    ##
    # Child class should override these
    ##

    file_extension = ''

    def open(self, filename):
        """Opens the file specified by filename for reading"""
        raise NotImplementedError()

    def close(self):
        """Close the currently open file"""
        pass

    ##
    # Override this if you support non-continuous event numbers
    ##
    def get_event_numbers_in_current_file(self):
        return list(range(self.current_first_event, self.current_last_event + 1))

    ##
    # Override this if you support random access
    ##
    def get_single_event_in_current_file(self, event_number):
        """Uses iteration to emulate random access to events
        This does not check if the event actually exist: get_events is supposed to do that.
        """
        for event_i, event in enumerate(self.get_all_events_in_current_file()):
            if event.event_number == event_number:
                return event
        raise RuntimeError("Current file has no event %d, and some check didn't pick this up.\n"
                           "Either the file is very nasty, or the reader is bugged!" % event_number)

    ##
    # Override this if you DO NOT support random access, or if random access is slower than iteration
    ##
    def get_all_events_in_current_file(self):
        """Uses random access to iterate over all events"""
        for event_number in self.event_numbers_in_current_file:
            yield self.get_single_event_in_current_file(event_number)


class WriteToFolder(plugin.OutputPlugin):
    """Write to a folder containing several small files, each containing <= a fixed number of events"""

    def startup(self):
        self.events_per_file = self.config.get('events_per_file', 50)
        self.first_event_in_current_file = None
        self.last_event_written = None

        self.output_dir = self.config['output_name']
        if os.path.exists(self.output_dir):
            if self.config.get('overwrite_output', False):
                if self.config['overwrite_output'] == 'confirm':
                    print("\n\nOutput dir %s already exists. Overwrite? [y/n]:" % self.output_dir)
                    if input().lower() not in ('y', 'yes'):
                        print("\nFine, Exiting pax...\n")
                        exit()
                self.log.info("Overwriting output directory %s" % self.output_dir)
                shutil.rmtree(self.output_dir)
                os.mkdir(self.output_dir)
            else:
                raise ValueError("Output directory %s already exists, can't write your %ss there!" % (
                    self.output_dir, self.file_extension))
        else:
            self.log.info("Creating output directory %s" % self.output_dir)
            os.mkdir(self.output_dir)

        # Write the metadata to JSON
        with open(os.path.join(self.output_dir, 'pax_info.json'), 'w') as outfile:
            outfile.write(json_util.dumps(self.processor.get_metadata(),
                          sort_keys=True))

        # Start the temporary file. Events will first be written here, until events_per_file is reached
        self.tempfile = os.path.join(self.output_dir, 'temp.' + self.file_extension)

    def open_new_file(self, first_event_number):
        """Opens a new file, closing any old open ones"""
        if self.last_event_written is not None:
            self.close_current_file()
        self.first_event_in_current_file = first_event_number
        self.events_written_to_current_file = 0
        self.open(filename=self.tempfile)

    def write_event(self, event):
        """Write one more event to the folder, opening/closing files as needed"""
        if self.last_event_written is None \
                or self.events_written_to_current_file >= self.events_per_file:
            self.open_new_file(first_event_number=event.event_number)

        self.write_event_to_current_file(event)

        self.events_written_to_current_file += 1
        self.last_event_written = event.event_number

    def close_current_file(self):
        """Closes the currently open file, if there is one. Also handles temporary file renaming. """
        if self.last_event_written is None:
            self.log.info("You didn't write any events... Did you crash pax?")
            return

        self.close()

        # Rename the temporary file to reflect the events we've written to it
        os.rename(self.tempfile,
                  os.path.join(self.output_dir,
                               '%s-%d-%06d-%06d-%06d.%s' % (self.config['tpc_name'],
                                                            self.config['run_number'],
                                                            self.first_event_in_current_file,
                                                            self.last_event_written,
                                                            self.events_written_to_current_file,
                                                            self.file_extension)))

    def shutdown(self):
        if self.has_shut_down:
            self.log.error("Attempt to shutdown %s twice!" % self.__class__.__name__)
        else:
            self.close_current_file()

    ##
    # Child class should override these
    ##

    file_extension = None

    def open(self, filename):
        raise NotImplementedError

    def write_event_to_current_file(self, event):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


##
# Encoders for zipfiles of events
# Zipfile readers themselves are in plugins/io/Zip.py
# (they have to be in /pax/plugins/... to be found)
##
class ReadZippedDecoder(plugin.TransformPlugin):
    do_input_check = False

    def transform_event(self, event_proxy):
        data = zlib.decompress(event_proxy.data)
        return self.decode_event(data)

    def decode_event(self, event):
        raise NotImplementedError


class WriteZippedEncoder(plugin.TransformPlugin):
    """Encode and compress an event for entry into a zipfile.
    Note we use zlib, not zip's deflate, for compression.
    """
    do_output_check = False

    def startup(self):
        self.compresslevel = self.config.get('compresslevel', 4)

    def transform_event(self, event):
        event_number = event.event_number
        data = self.encode_event(event)
        data = zlib.compress(data, self.compresslevel)
        return EventProxy(data=data, event_number=event_number)

    def encode_event(self, event):
        raise NotImplementedError

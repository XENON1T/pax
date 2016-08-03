import numpy as np
from collections import OrderedDict
from pax.trigger import TriggerPlugin
from pax import exceptions


class DeadTimeTally(TriggerPlugin):
    """Counts the total dead time due to each veto system (HEV, busy) and stores it in the trigger data
    TODO: once we start operating the high-energy veto, we need to monitor the exact time either one of them is on.

    I have not tested the speed at all: if it is slow, we need to move the loop over times to numba.
    """

    def startup(self):
        # Read the on/off status channels from the configuration
        self.systems = OrderedDict()
        self.special_channels = {}
        for detector, channels in self.trigger.pax_config['DEFAULT']['channels_in_detector'].items():
            if detector.endswith('_on') or detector.endswith('_off'):
                assert len(channels) == 1
                channel = channels[0]
                system, status = detector.split('_')
                if system not in self.systems:
                    self.systems[system] = {'active': False, 'start_of_current_dead_time': 0, 'dead_time_tally': 0}
                if status == 'on':
                    self.systems[system]['on_channel'] = channel
                elif status == 'off':
                    self.systems[system]['off_channel'] = channel
                else:
                    raise exceptions.InvalidConfigurationError("Veto system status must be "
                                                               "'on' or 'off', not %s" % status)
                self.special_channels[channel] = {'system': system, 'means_on': status == 'on'}

        self.log.info("Dead time tally info: %s" % str(self.systems))
        self.log.info("Special channels: %s" % str(self.special_channels))

    def save_monitor_data(self):
        # Save the dead time in this interval
        dead_times = {'time': self.next_save_time}
        for system_name, system in self.systems.items():

            if system['active']:
                # The system is currently active!
                # Register dead time up to the save boundary, then change the start time to the boundary
                system['dead_time_tally'] += self.next_save_time - system['start_of_current_dead_time']
                system['start_of_current_dead_time'] = self.next_save_time

            dead_times[system_name] = system['dead_time_tally']
            system['dead_time_tally'] = 0

        self.trigger.save_monitor_data('dead_time_info', dead_times)

    def process(self, data):
        self.next_save_time = self.config['dark_rate_save_interval']
        self.save_interval = self.config['dark_rate_save_interval']
        special_times = data.times[np.in1d(data.times['pmt'], list(self.special_channels.keys()))]
        self.log.info("Found %d signal in on/off acquisition monitor channels" % len(special_times))

        for t in special_times:

            while t['time'] > self.next_save_time:
                self.save_monitor_data()
                self.next_save_time += self.save_interval

            ch_info = self.special_channels[t['pmt']]
            system_name = ch_info['system']
            system = self.systems[system_name]
            if system['active']:
                if ch_info['means_on']:
                    self.log.warning("%s-on signal received while system was already active!\n"
                                     "The signal has been ignored." % system_name)
                else:
                    # System has turned off
                    system['active'] = False
                    system['dead_time_tally'] += t['time'] - system['start_of_current_dead_time']
                    pass
            else:
                if ch_info['means_on']:
                    # System has turned on
                    system['active'] = True
                    system['start_of_current_dead_time'] = t['time']
                    pass
                else:
                    self.log.warning("%s-off signal received while system was not yet active!\n"
                                     "The signal has been ignored." % system_name)
                    pass

        if data.last_data:
            # Save the dead time info for the final part of the run
            # If there is no dead time anywhere in the run, this is actually the only time
            # we store information!
            while data.last_time_searched > self.next_save_time:
                self.save_monitor_data()
                self.next_save_time += self.save_interval

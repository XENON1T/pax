========
Usage
========

To use processor for Analyzing XENON1T, see /bin/paxer --help for available options:


        usage: paxer [-h] [--input [INPUT]] [--output [OUTPUT]] [--log LOG]
                     [--config {3d_plotter,bern,daq_injector,Mongo,newDSP,XAMS,XED,XENON100,XENON1T} [{3d_pl
        otter,bern,daq_injector,Mongo,newDSP,XAMS,XED,XENON100,XENON1T} ...]]
                     [--config_path CONFIG_PATH [CONFIG_PATH ...]]
                     [--plot | --plot_to_dir PLOT_TO_DIR] [--event EVENT [EVENT ...] |
                     --stop_after STOP_AFTER]

        Process XENON1 data

        optional arguments:
          -h, --help            show this help message and exit
          --input [INPUT]       File, database or directory to read events from
          
          --output [OUTPUT]     File, database or directory to write events to
          
          --log LOG             Set log level, e.g. 'debug'
          
          --config {3d_plotter,bern,daq_injector,Mongo,newDSP,XAMS,XED,XENON100,XENON1T} [{3d_plotter,bern,daq_injector,Mongo,newDSP,XAMS,XED,XENON100,XENON1T} ...]
                                Name(s) of the pax configuration(s) to use.
          --config_path CONFIG_PATH [CONFIG_PATH ...]
                                Path(s) of the configuration file(s) to use.
          --plot                Plot summed waveforms on screen
          --plot_to_dir PLOT_TO_DIR
                                Save summed waveform plots in directory
          --event EVENT [EVENT ...]
                                Process particular event(s).
          --stop_after STOP_AFTER
                                Stop after STOP_AFTER events have been processed.

For more advanced usage you would want to change the configuration: you can make your own configuration file.
For a description of the settings you can use, see the comments in _base.ini and XENON100.ini.

from pax import core, units

print("Processing the real data...")
core.Processor(config_names=['XENON100', 'newDSP'], config_dict={
    'pax' : {
        'input_name'  :     'xe100_120402_2000_000000.xed',
        'output_name' :     'xe100_120402_2000_000000',
        'input' :           'XED.XedInput',
        'output':           'Pandas.WritePandas',
    },
    'Pandas.WritePandas' : {
        'output_format': 'hdf',
    }
}).run()

print("Simulating some fake data...")
for simulation_name in ('fake_s1s', 'fake_s2s'):
    core.Processor(config_names=['XENON100', 'newDSP'], config_dict={
        'pax' : {
            'input_name'  :     simulation_name + '.csv',
            'output_name' :     simulation_name,
            'input' :           'WaveformSimulator.WaveformSimulatorFromCSV',
            'output':           'Pandas.WritePandas',
        },
        'WaveformSimulator' : {
            'event_repetitions' : 100,
            'electron_lifetime_liquid' : 1000 * units.s,
        },
        'Pandas.WritePandas' : {
            'output_format': 'hdf',
        }
    }).run()
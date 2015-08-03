"""
Generates some processor output for real data and simulated data.

The electron lifetime is set to 'infinity', so the area of S2s
is more under control.

1/100 waveforms are plot so you can verify they are sensible.
 -- this is always a good idea for simulated data!
"""

from pax import core, units

# print("Processing the real data...")
# core.Processor(config_names=['XENON100'], config_dict={
    # 'pax' : {
        # 'input_name'  :     'xe100_120402_2000_000000.xed',
        # 'output_name' :     'xe100_120402_2000_000000',
        # 'input' :           'XED.ReadXED',
        # 'output':           ['BulkOutput.BulkOutput','Plotting.PlotEventSummary']
    # },
	# 'Plotting.PlotEventSummary' : {
		# 'plot_every':	100,
		# 'output_dir':	'real_data_waveforms'
	# }
# }).run()

print("Simulating some fake data...")
for simulation_name in ('fake_s1s', 'fake_s2s'):
    core.Processor(config_names=['XENON100'], config_dict={
        'pax' : {
            'input_name'  :     simulation_name + '.csv',
            'input' :           'WaveformSimulator.WaveformSimulatorFromCSV',
            'output':           ['BulkOutput.BulkOutput','Plotting.PlotEventSummary'],
        },
        'WaveformSimulator' : {
            'event_repetitions' : 100,
            'electron_lifetime_liquid' : 1000 * units.s,
			'wire_field_parameter': 0,
			'drift_velocity_gas': 3*units.mm/units.us,
        },
        'BulkOutput.BulkOutput' : {
			'output_name' :     simulation_name,
        },
		'Plotting.PlotEventSummary' : {
			'plot_every':	100,
			'output_dir':	simulation_name + '_waveforms',
		}
    }).run()
from pax import plugin, units


class DumpSumWaveformToBinary(plugin.OutputPlugin):
    def startup(self):
        c = self.config
        # Factor from assemble_signals, without the ADC -> voltage stuff, with gain=2e6
        self.conversion_factor = c['digitizer_t_resolution'] / (
            c['pmt_circuit_load_resistor'] * c['external_amplification'] * units.electron_charge * 2e6
        )
        self.log.debug("Conversion factor %s" % self.conversion_factor)

    def write_event(self, event):
        filename = self.config['output_dir'] + '/' + str(event.event_number) + '.' +  self.config['extension']
        waveform_to_dump = event.get_waveform(self.config['waveform_to_dump']).samples
        if self.config['dump_in_units'] == 'voltage':
            waveform_to_dump /= self.conversion_factor
        with open(filename, 'wb') as output:
            waveform_to_dump.tofile(output)

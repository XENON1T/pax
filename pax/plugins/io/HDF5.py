__author__ = 'tunnell'

import pax.datastructure as ds

from pax import plugin

import tables


class HDF5Output(plugin.OutputPlugin):

    def startup(self):
        self.h5_file = tables.open_file(self.config['hdf5file'], 'w')

        self.hdf5_fields = {}

        internal_classes = [ds.Event(),
                            ds.Peak(),
                            ds.ReconstructedPosition()]

        for ic in internal_classes:
            fields = {}
            for key, value in ic._fields.items():
                if value.hdf5_type() is not None:
                    fields[key] = value.hdf5_type()

            name = ic.__class__.__name__

            self.hdf5_fields[name] = fields

        self.hdf5_fields['Peak']['event_number'] = self.hdf5_fields[
            'Event']['event_number']

        # Filters are used for compression.  We use the blosc algorithm.
        compression_filter = tables.Filters(complevel=5,
                                complib='blosc')

        self.tables = {}
        self.rows = {}

        for key, value in self.hdf5_fields.items():
            table = type('%s_table' % key.lower(),
                         (tables.IsDescription,),
                         value)

            self.tables[key] = self.h5_file.createTable('/',
                                                        '%s_table' % key.lower(),
                                                        table,
                                                        filters=compression_filter)
            self.rows[key] = self.tables[key].row

    def write_event(self, event):
        self.log.debug('HDF5ing event')

        event_dict = event.to_dict()

        for key, val in self.hdf5_fields['Event'].items():
            if isinstance(val, tables.Col):
                self.rows['Event'][key] = event_dict[key]

        self.rows['Event'].append()

        for peak in event_dict['peaks']:
            peak = peak.to_dict()
            for key, val in self.hdf5_fields['Peak'].items():
                if isinstance(val, tables.Col):
                    if key == 'event_number':
                        self.rows['Peak'][key] = event_dict[key]
                    else:
                        self.rows['Peak'][key] = peak[key]
            self.rows['Peak'].append()

    def shutdown(self):
        for table in self.tables.values():
            table.flush()

        self.h5_file.close()

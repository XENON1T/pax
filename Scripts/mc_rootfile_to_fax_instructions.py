import ROOT
import csv

f = ROOT.TFile('Neutron-4FaX-10k.root')
t = f.Get("t1") # For Xerawdp use T1, for MC t1

# From https://xecluster.lngs.infn.it/dokuwiki/doku.php?id=xenon:xenon100:analysis:roottree
variables = (
    ('x',             'Nest_x',     0.1),
    ('y',             'Nest_y',     0.1),
    ('depth',         'Nest_z',     -0.1),
    ('s1_photons',    'Nest_nph',   1),
    ('s2_electrons',  'Nest_nel',   1),
    ('t',             'Nest_t',     10**(9)),
)

for event_i in range(t.GetEntries()):
    t.GetEntry(event_i)
    print(event_i)

    # Get stuff from root files
    values = {}
    for (variable_name, root_thing_name, _) in variables:
        values[variable_name] = getattr(t, root_thing_name)

    # Convert to peaks dictionary
    npeaks = len(values[variables[0][0]])
    peaks = []
    for i in range(npeaks):
        peaks.append({'event' : event_i})
        for (variable_name, _, conversion_factor) in variables:
            peaks[-1][variable_name] = values[variable_name][i] * conversion_factor

    # Subtract depth of gate mesh, see xenon:xenon100:mc:roottree, bottom of page
    for p in peaks:
        p['depth'] -= 2.15+0.25

    # Sort by time
    peaks.sort(key = lambda p:p['t'])

    # Write to csv
    for peak_i, peak_data in enumerate(peaks):
        if event_i == 0 and peak_i == 0:
            headers = ['event'] + [q[0] for q in variables]
            output = open('fax_instructions.csv', 'wb')
            csvwriter = csv.DictWriter(output, headers)
            print(headers)
            #print(','.join(headers))
            #output.write(','.join(headers))
            csvwriter.writer.writerow(csvwriter.fieldnames)
        #output.write(','.join([peak_data[h] for h in headers]))
        csvwriter.writerow(peak_data)

        

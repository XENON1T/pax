"""Perform a comparison between pax and xerawdp

Please generate a ROOT file using the XeAnalaysiScripts.  Be sure to also have
the file Loader.C whereever you run this file.  Also, you run this file in the
directory containing all the h5 files.
"""
import os
import pickle

from tables import openFile
import matplotlib.pyplot as plt
import numpy as np

import ROOT


def run_comparison():
    ROOT.gROOT.ProcessLine('.L Loader.C')

    root_filename = '../trim_xe100_run10_AmBe_cuts_run_10.root'
    dataset = 'xe100'#_110210_1926_0002'

    root_file = ROOT.TFile(root_filename)
    t1 = root_file.Get('T1')
    t2 = root_file.Get('T2')
    # t3 = root_file.Get('T3')

    pax_file = None

    variables_to_compare = {'S2[0]': [], 'x': [], 'y': []}

    number_of_not_found_S2s = 0

    for i in range(t1.GetEntries()):
        t1.GetEntry(i)
        t2.GetEntry(i)

        # Remove stupid null characters using translate
        this_dataset = str(t1.Filename)
        this_dataset = this_dataset.translate(dict.fromkeys(range(32)))

        if dataset not in this_dataset:
            continue

        if pax_file is None or this_dataset not in pax_file.filename:
            if pax_file:
                pax_file.close()


            filename = '%s.h5' % this_dataset
            #filename = 'data/hdf5_data_2/%s.h5' % this_dataset
            if not os.path.exists(filename):
                continue
            try:
                pax_file = openFile(filename, mode='r')
            except:
                print('#Failed to open', filename)
                print('rm', filename)
                continue
            #print(pax_file)

        time = np.uint64(t1.TimeSec) * np.uint64(1e8)
        time += np.uint64(t1.TimeMicroSec) * np.uint64(100)
        time += np.uint64(t2.S2sPeak[0])
        time *= np.uint64(10)

        # Find event number
        this_event = None
        for event in pax_file.root.event_table:
            if event['start_time'] < time < event['stop_time']:
                this_event = event
                break

        have_I_found_an_S2 = False

        for peak in pax_file.root.peak_table.where("(event_number == %d) & (type == b's2')" % this_event['event_number']):
            assert this_event['start_time'] < time < this_event['stop_time']

            start = peak['left'] * this_event['sample_duration'] + this_event[
                'start_time']
            stop = peak['right'] * this_event['sample_duration'] + this_event[
                'start_time']

            if start > time or time > stop:
                continue

            if t2.S2sTot[0] > 1000:
                continue

            variables_to_compare['S2[0]'].append((t2.S2sTot[0],
                                                  peak['area']))

            found = False
            for track in pax_file.root.reconstructedposition_table.where(
                            "(event_number == %d) & (index_of_maximum == %d)" % (
                    this_event['event_number'],
                    peak['index_of_maximum'])):
                if not found:
                    found = True
                else:
                    raise RuntimeError("found twice")

                for i, variable in enumerate(['x', 'y']):
                    #print(variable, (t2.S2sPosNn[0][i],
                    #                 track[variable]))
                    variables_to_compare[variable].append((t2.S2sPosNn[0][i],
                                                           track[variable]))


    pickle.dump(variables_to_compare, open( "save.p", "wb" ))

if not os.path.exists("save.p"):
    run_comparison()

variables_to_compare = pickle.load( open( "save.p", "rb" ) )

for k, v in variables_to_compare.items():
    print(k, v)
    x, y = zip(*v)
    x, y = np.array(x), np.array(y)

    plt.figure()
    plt.title(k)
    plt.scatter(x, y)
    plt.xlabel('root')
    plt.ylabel('pax')
    plt.savefig('%s_scatter.eps' % k)

    plt.figure()
    plt.title(k)
    plt.hist((x - y))
    plt.xlabel('root - pax')
    plt.savefig('%s_hist.eps' % k)

    plt.figure()
    plt.title(k)
    plt.hexbin(x, y, bins=40)
    plt.xlabel('root')
    plt.ylabel('pax')
    plt.savefig('%s_2dhist.eps' % k)

x0, x = zip(*variables_to_compare['x'])
x0, x = np.array(x0), np.array(x)
dx = x - x0

y0, y = zip(*variables_to_compare['y'])
y0, y =np.array(y0), np.array(y)
dy = y - y0

plt.figure()
plt.title(k)
plt.plot(dx, dy, '.')
plt.xlabel('dx [mm]')
plt.ylabel('dy [mm]')
plt.savefig('dx_dy.eps')


#plt.show()

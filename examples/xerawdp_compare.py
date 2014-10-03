from tables import openFile
import matplotlib.pyplot as plt
import numpy as np

import ROOT

root_filename = 'data/trim_xe100_run10_AmBe_cuts_run_10.root'
dataset = 'xe100_110210_1926_00000'

root_file = ROOT.TFile(root_filename)
root_trees = {}
for i in [1, 2, 3]:
    key = 'T%d' % i
    root_trees[key] = root_file.Get(key)

pax_file = None

assert root_trees['T1'].GetEntries() == root_trees['T2'].GetEntries()

variables_to_compare = {'S2[0]' : []}

for i in range(root_trees['T1'].GetEntries()):
    for val in root_trees.values():
        val.GetEntry(i)

    # Remove stupid null characters using translate
    this_dataset = str(root_trees['T1'].Filename)
    this_dataset = this_dataset.translate(dict.fromkeys(range(32)))

    if dataset not in this_dataset:
        continue

    if pax_file is None or this_dataset not in pax_file.filename:
        if pax_file:
            pax_file.close()

        filename = 'data/%s.h5' % this_dataset
        pax_file = openFile(filename, mode='r')

    time = np.uint64(root_trees['T1'].TimeSec) * np.uint64(1e8)
    time += np.uint64(root_trees['T1'].TimeMicroSec) * np.uint64(100)
    time += np.uint64(root_trees['T2'].S2sPeak[0])
    time *= np.uint64(10)

    # Find event number
    this_event = None
    for event in pax_file.root.event_table:
        if event['start_time'] < time < event['stop_time']:
            this_event = event
            break

    for peak in pax_file.root.peak_table.where("(event_number == %d) & (type == b's2')" % this_event['event_number']):
        assert this_event['start_time'] < time < this_event['stop_time']

        start = peak['left'] * this_event['sample_duration'] + this_event['start_time']
        stop = peak['right'] * this_event['sample_duration'] + this_event['start_time']

        if start < time < stop:
            print(root_trees['T2'].S2sTot[0],
                  peak['area'])
            variables_to_compare['S2[0]'].append((root_trees['T2'].S2sTot[0],
                                                  peak['area']))


for k, v in variables_to_compare.items():
    print(k)

    x, y = zip(*v)
    x, y = np.array(x), np.array(y)
    print(x,y)

    plt.figure()
    plt.title(k)
    plt.scatter(x, y)
    plt.xlabel('root')
    plt.ylabel('pax')

    plt.figure()
    plt.title(k)
    plt.hist((x-y)/x)
    plt.xlabel('root - pax')

plt.show()





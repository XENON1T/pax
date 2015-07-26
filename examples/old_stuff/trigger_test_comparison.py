from tables import openFile
import matplotlib.pyplot as plt
import numpy as np

# Open the file for reading
file_mongo = openFile("output_mongo.h5", mode = "r")
file_xed = openFile("output_xed.h5", mode = "r")

# Get the root group

results = []

skipped_none_in_xed = []
good = []
all = []

for event_xed in file_xed.root.event_table:
    peak_xed = None
    for peak_xed in file_xed.root.peak_table.where("(event_number == %d) & (type == b's2')" % event_xed['event_number']):
        break

    if peak_xed == None:
        skipped_none_in_xed.append(event_xed['event_number'])
        continue
    start_xed = peak_xed['left'] * event_xed['sample_duration'] + event_xed['start_time']
    maximum_xed = peak_xed['index_of_maximum'] * event_xed['sample_duration'] + event_xed['start_time']
    stop_xed = peak_xed['right'] * event_xed['sample_duration'] + event_xed['start_time']

    print('XED says:', peak_xed['type'], '%d pe' % peak_xed['area'], start_xed, maximum_xed, stop_xed)

    found_peak = False
    for event_mongo in file_mongo.root.event_table.where('(start_time < %d) & (%d < stop_time)' % (start_xed, stop_xed)):
            
        for peak_mongo in file_mongo.root.peak_table.where('(event_number == %d)' % event_mongo['event_number']):
            start_mongo = peak_mongo['left'] * event_mongo['sample_duration'] + event_mongo['start_time']
            maximum_mongo = peak_mongo['index_of_maximum'] * event_mongo['sample_duration'] + event_mongo['start_time']
            stop_mongo = peak_mongo['right'] * event_mongo['sample_duration'] + event_mongo['start_time']
            
            if start_xed < maximum_mongo < stop_xed:
                if found_peak:
                    print('wtf, already found?', event_mongo['event_number'], event_xed['event_number'])
                found_peak = True

                print('MONGO says:', peak_mongo['type'], '%d pe' % peak_mongo['area'], start_mongo, maximum_mongo, stop_mongo)
                print('diff',  start_xed - start_mongo, stop_xed - stop_mongo)

    if found_peak:
        good.append(peak_xed['area'])
    else:
        print('missed', event_xed['event_number'])
    all.append(peak_xed['area'])

print(len(all))
# Close the file
file_mongo.close()
file_xed.close()

bins = np.linspace(0, 200, 10)
print(bins)
a, _, _ = plt.hist(all, bins=bins, label='all')
b, _, _ = plt.hist(good, bins=bins, label='good')

a = np.array(a)
b = np.array(b)

plt.legend()
plt.show()


plt.figure()
plt.plot(0.5 * (bins[:-1] + bins[1:]),
         b/a, label='trigger eff')
print(b/a)

print(a,b)
plt.legend()
plt.show()


print('skipped', skipped_none_in_xed)

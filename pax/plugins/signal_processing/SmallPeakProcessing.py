import numpy as np
from pax import plugin, datastructure, dsputils, units


##
# Temporary stuff for diagnostic plotting
##
import matplotlib.pyplot as plt
import pandas

def make_plot(func, xmin, xmax, dx, *args, **kwargs):
    x = np.linspace(xmin, xmax, (xmax-xmin)/dx)
    y = list(map(func, x))
    plt.plot(x, y, *args, **kwargs)

def bin_centers(bin_edges):
    return np.array( (np.roll(bin_edges, 1) + bin_edges)/2 )[1:]

def chist(data, bins):
    hist, bins = np.histogram(data,bins)
    hist = np.cumsum(hist)/len(data)
    return bin_centers(bins), hist

def chist_plot(data, bins, *args, **kwargs):
    bcs, hist = chist(data, bins)
    plt.plot(bcs, hist, *args, **kwargs)

def make_plot(func, xmin, xmax, dx, *args, **kwargs):
    x = np.linspace(xmin, xmax, (xmax-xmin)/dx)
    y = list(map(func, x))
    plt.plot(x, y, *args, **kwargs)


class ClusterAndClassifySmallPeaks(plugin.TransformPlugin):

    def startup(self):
        self.dt = dt = self.config['digitizer_t_resolution']
        self.cluster_separation_length = self.config['cluster_separation_time']
        self.classification_mode = self.config['classification_mode']
        self.channels_in_detector = {
            'tpc':  self.config['pmts_top'] | self.config['pmts_bottom'],
        }
        for det, chs in self.config['external_detectors'].items():
            self.channels_in_detector[det] = chs

        if self.classification_mode == 'waveform_models':
            # Build the waveform models
            # Be careful with median as zeroing function: leads to spike at 0! (if odd-numbered sampling points)
            self.log.info("Building waveform models...")
            self.simulator = self.processor.simulator
            self.wvmodels = {}
            for rt in ('ER', 'NR'):
                # Infinite S1s
                self.wvmodels['s1_%s_inf'%rt] = WaveformModel(
                    training_set=self.processor.simulator.s1_photons(100000, rt),
                    zero_function=np.mean,
                    base_type='s1'
                )

                # Small S1s: 3 pe for demonstration
                # mean normalization introduces a sample-size dependent effect
                self.wvmodels['s1_%s_3'%rt] = WaveformModel(
                    training_set=self.simulator.s1_photons(3*10000, rt),
                    zero_function=np.mean,
                    sample_length=3,
                    base_type='s1'
                )

            # Deeper S2s
            # TODO: depth=10 problematic. Can't just mix in a few depth=10, there's drift time...
            self.wvmodels['s2_10cm'] = WaveformModel(
                training_set=self.simulator.s2_scintillation(self.simulator.s2_electrons(10000, z=10*units.cm)),
                zero_function=np.mean,
                base_type='s2'
            )

            # Surface S2s (i.e. including single-electron S2s)
            self.wvmodels['s2_surface'] = WaveformModel(
                training_set=self.simulator.s2_scintillation(electron_arrival_times=np.zeros(1000)),
                zero_function=np.mean,
                base_type='s2'
            )
            self.log.info("...done")

            # for wv_name, wv_model in self.wvmodels.items():
            #     make_plot(wv_model.stat.pdf, -units.us, units.us, 10*units.ns, label=wv_name)
            # plt.yscale('log')
            # plt.legend()
            # plt.xlabel('time (ns) from zero')
            # plt.ylabel('pdf')
            # plt.show()


    def transform_event(self, event):

        # Handle each detector separately
        for detector in self.channels_in_detector.keys():

            # Hmzz, python has no do_while, so..
            redo_classification = True
            while redo_classification:
                redo_classification = False

                # Get all single-pe data in a list of dicts, sorted by index_of_maximum
                spes = sorted([
                    p.to_dict() for p in event.channel_peaks
                                if p.channel in self.channels_in_detector[detector]
                                and (not self.config['exclude_bad_channels'] or not event.is_channel_bad[p.channel])
                ], key=lambda x: x['index_of_maximum'])

                if not len(spes):
                    clusters = []
                    continue

                times = [s['index_of_maximum'] * self.dt for s in spes]
                assert(times == sorted(times))
                time_clusters = dsputils.cluster_by_diff(times, self.cluster_separation_length, return_indices=True)

                # Make a list of dicts of spe clusters (essentially a dataframe, but I want to do a for loop...)
                clusters = [{
                        'spes':         cluster_spes,
                        'n_spes':       len(cluster_spes),
                        # TODO: add extents of min/max peaks
                        'left':         spes[cluster_spes[0]]['left'],
                        'right':        spes[cluster_spes[-1]]['right'],
                        'type':         'unknown',
                    } for cluster_spes in time_clusters]

                dark_count = {}

                for c in clusters:
                    # Find how many channels show something (noise, bad)
                    # and how many are good & show photons

                    coincident_occurrences = event.occurrences_interval_tree.search(c['left'], c['right'], strict=False)
                    c['channels_with_something'] = set(self.channels_in_detector[detector]) & \
                                                   set([oc[2]['channel'] for oc in coincident_occurrences])
                    c['channels_with_photons'] = set([spes[x]['channel'] for x in c['spes']])
                    c_times = [times[i] for i in c['spes']]
                    c['mad'] = dsputils.mad(c_times)

                    if len(c['channels_with_something']) > 2 * len(c['channels_with_photons']):
                        c['type'] = 'noise'
                        continue

                    if len(c['channels_with_photons']) == 1:
                        c['type'] = 'lone_pulse'
                        channel = spes[c['spes'][0]]['channel']
                        dark_count[channel] = dark_count.get(channel, 0) + 1
                        continue

                    if self.classification_mode == 'waveform_models':

                        # Use the waveform models for classification
                        # TODO handle pile-up in WaveformModel: pass all info, not just times

                        # Desparate measure to make c_times continuous
                        # TODO: implement better arrival time estimate in peakfinder based on second bin
                        c_times += np.random.normal(0, self.dt, len(c_times))

                        classification_results = []
                        for wv_name, wv_model in self.wvmodels.items():
                            ks_stat, ks_p = wv_model.timing_kstest(c_times)
                            classification_results.append({
                                'name':              wv_name,
                                'base_type':         wv_model.base_type,
                                'ks_p':              ks_p,
                                'ks_stat':           ks_stat,
                                'total_likelihood':  wv_model.total_likelihood(c_times)
                            })

                        # TODO: store in event class
                        classification_results.sort(key=lambda x: x['total_likelihood'], reverse=True)

                        # print(pandas.DataFrame(classification_results))
                        # chist_plot(c_times - np.mean(c_times), 10, label='Empiral CDF')
                        # make_plot(self.wvmodels['s1_ER_3'].stat.cdf, -100, 100, 1, label='3pe model CDF')
                        # make_plot(self.wvmodels['s1_ER_inf'].stat.cdf, -100, 100, 1, label='inf_pe model CDF')
                        # plt.legend(loc='lower right')
                        # plt.show()

                        # If all likelihoods < 10**-6 we choose 'unknown'
                        # Else we take the model with the highest likelihood,
                        # TODO: classify as unknown if likelihood of other base type is similar
                        # TODO: take KS test into account as well
                        classification_options = [c for c in classification_results
                                                  if c['total_likelihood'] > -6] #and c['ks_stat'] < 0.5]
                        if len(classification_options) == 0:
                            c['type'] = 'unknown'
                        else:
                            c['type'] = max(classification_options, key=lambda x : x['total_likelihood'])['base_type']


                    else:
                        # Simple ad-hoc classification

                        if c['mad'] < 10:
                            c['type'] = 's1'
                        else:
                            if c['n_spes'] < 5:
                                c['type'] = 'unknown'
                            else:
                                c['type'] = 's2'

                # Look for channels with abnormal dark rate
                for ch, dc in dark_count.items():
                    if dc > self.config['maximum_lone_pulses_per_channel']:
                        self.log.debug(
                            "Channel %s shows an abnormally high lone pulse rate (%s): its spe pulses will be excluded" % (
                                ch, dc))
                        event.is_channel_bad[ch] = True
                        if self.config['exclude_bad_channels']:
                            redo_classification = True

            # Classification is now done, so add the peaks to the datastructure
            # clusters was last set in the while redo_classification loop...
            for c in clusters:
                # We need an index_of_maximum and height, these we can only get from the sum waveform]
                # TODO: do we really want these in datastructure even for peaks reconstructed from spes?
                sum_wave = event.get_waveform(detector).samples[c['left'] : c['right'] + 1]
                max_idx = np.argmax(sum_wave)
                height = sum_wave[max_idx]
                event.peaks.append(datastructure.Peak({
                    'index_of_maximum':     max_idx + c['left'],
                    'height':               height,
                    'left':                 c['left'],
                    'right':                c['right'],
                    'area':                 sum([spes[x]['area'] for x in c['spes']]),
                    'does_pmt_contribute':  np.array(
                        [ch in c['channels_with_photons'] for ch in range(self.config['n_pmts'])],
                        dtype=np.bool),
                    'area_per_pmt':         np.array([
                                                sum([spes[x]['area'] for x in c['spes'] if spes[x]['channel'] == ch])
                                                for ch in range(len(event.pmt_waveforms))]),
                    'type':                 c['type'],
                    'detector':             detector,
                    'mean_absolute_deviation': c['mad'],
                }))

        return event


##
# Stuff for waveform modelling
##
import scipy
from scipy import stats, interpolate

class WaveformModel:

    def __init__(self, training_set, zero_function, sample_length=None, fudge=5*units.ns, base_type='weird'):

        self.base_type = base_type
        self.zero_function = zero_function

        # Add some timing fudge
        training_set += np.random.normal(0, fudge, len(training_set))

        # Apply mean normalization
        if sample_length is not None:
            # Chop into parts of length sample_length, apply zero_function to each
            training_set = training_set.reshape((-1, sample_length))
            training_set -= zero_function(training_set, axis=1)[:,np.newaxis]
            training_set = training_set.reshape((-1))
        else:
            training_set -= zero_function(training_set)

        # Poor man's binning - is this good enough or bad?
        # TODO: don't hardcode 10ns
        # training_set = np.around(training_set/10)*10

        self.stat = MCStatistic.from_mc(training_set) #, n_points_for_cdf=len(np.unique(training_set)))

    def timing_kstest(self, sample):
        sample -= self.zero_function(sample)
        return stats.kstest(sample, self.stat.cdf)

    def total_likelihood(self, sample):
        # Child classes can override this to include additional penalties for e.g. n_photons, hitpattern, ...
        # TODO: not quite though, sample is still a plain list...
        return self.timing_likelihood(sample)

    def timing_likelihood(self, sample):

        sample -= self.zero_function(sample)
        log_ls = np.log10(self.stat.pdf(sample))

        #A single outlier should not push everything to -inf
        log_ls = np.clip(log_ls, -50, 0)
        log_ls[np.isnan(log_ls)] = -50

        return  1/len(sample) * np.sum(log_ls)

#TODO: maybe keep binning, subclass rv_discrete instead?


#TODO: belongs in separate package
#TODO: doesn't handle spikes in CDF well (e.g. half the values at 0, rest smeared -> pdf nan at 0)
class MCStatistic(scipy.stats.rv_continuous):

    #init defines cdf by monte carlo, lets scipy take care of everything else
    @classmethod
    def from_mc(cls, mc_training_sample=None, n_points_for_cdf=None, **kwargs):
        if n_points_for_cdf is None:
            n_points_for_cdf = int(2*len(mc_training_sample)**(0.6))
        s = mc_training_sample
        s.sort()
        # TODO: Maybe first interpolate all points, then use http://scipy-central.org/item/53/1/adaptive-sampling-of-1d-functions
        # Poor man's adaptive sampling...
        # First divide into chunks to determine how much data varies in this range
        n_chunks = n_points_for_cdf**0.5
        presample_points = np.linspace(0, len(s)-1, int(n_chunks)).astype('int')
        points_to_sample = np.diff(s[presample_points])**0.8 # Sample density ~ (data variation)
        points_to_sample *= n_points_for_cdf/np.sum(points_to_sample)
        points_to_sample += 1
        points_to_sample = points_to_sample.astype('int')
        # print(n_points_for_cdf, presample_points, points_to_sample)
        # Now sample all the ranges
        sampling_points = []
        for i in range(len(points_to_sample)):
            sampling_points.append(np.linspace(
                presample_points[i],
                presample_points[i+1],
                points_to_sample[i],
                endpoint=False
            ).astype('int'))
        sampling_points = np.concatenate(sampling_points)
        cdf_training_points = np.array((s[sampling_points],sampling_points/len(s)))
        return cls.from_cdf_training_points(cdf_training_points, **kwargs)

    @classmethod
    def from_cdf_training_points(cls, cdf_training_points, **kwargs):
        self = cls(**kwargs)
        self.cdf_training_points = cdf_training_points
        self.training_limits = (cdf_training_points[0][0], cdf_training_points[0][-1])
        #Constuct empirical cdf
        self.cdf_itp = interpolate.interp1d(*self.cdf_training_points)
        return self

    @classmethod
    def from_file(cls, filename):
        cls.from_cdf_training_points(np.load(filename))

    def to_file(self, filename):
        np.save(filename, self.cdf_training_points)

    def _cdf(self, x):
        # Support for single values
        try:
            x[0]
        except TypeError:
            x = np.array([x])
        # Clipping = Extrapolate cdf constant outside training limits
        # too scared to clip in-place..
        x_clipped = np.zeros(len(x))
        np.clip(x, self.training_limits[0], self.training_limits[1], x_clipped)
        return self.cdf_itp(x_clipped)

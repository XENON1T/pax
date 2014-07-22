from pax import plugin, units
import numpy as np

# TODO: this part is not obvious, you need a paragraph on what pruning is
# TODO: does order of prunning matter?


# decision: none: accept, string: reject, string specifies reason


def is_s2(peak):  # put in plugin base class?
    return peak['peak_type'] in ('large_s2', 'small_s2')


class PeakPruner(plugin.TransformPlugin):

    def transform_event(self, event):
        for peak_index, p in enumerate(event['peaks']):
            # If this is the first peak pruner, we have to set up some values
            if not 'rejected' in p:
                p['rejected'] = False
                p['rejection_reason'] = None
                p['rejected_by'] = None

            # If peak has been rejected earlier, we don't have to test it
            # In the future we may want to disable this to test how the
            # prunings depend on each other
            if p['rejected']:
                continue
            # Child class has to define decide_peak
            decision = self.decide_peak(p, event, peak_index)
            # None means accept the peak. Anything else is a rejection reason.
            if decision != None:
                p['rejected'] = True
                p['rejection_reason'] = decision
                p['rejected_by'] = str(self.__class__.__name__)
        return event

    def decide_peak(self, peak, event, peak_index):
        raise NotImplementedError("This peak pruner forgot to implement decide_peak...")


class PruneNonIsolatedPeaks(PeakPruner):
    # mean of test_before samples before interval must be less than before_to_height_ratio_max times the maximum value in the interval
    # Same for test_after
    # NB: tests the PREPEAK, not the actual peak!!! (XeRawDP behaviour)

    def startup(self):
        # TODO: These should be in configuration...
        self.settings = {
            'test_before': {'large_s2': 21, 'small_s2': 10},
            'test_after': {'large_s2': 21, 'small_s2': 10},
            'before_to_height_ratio_max': {'large_s2': 0.05, 'small_s2': 0.05},
            'after_to_height_ratio_max': {'large_s2': 0.05, 'small_s2': 0.05}
        }

    def decide_peak(self, peak, event, peak_index):
        if peak['peak_type'] == 's1': return None

        # Find which settings to use for this type of peak
        settings = {}

        for settingname, settingvalue in self.settings.items():
            settings[settingname] = self.settings[settingname][peak['peak_type']]


        signal = event['processed_waveforms'][peak['input']]

        #Calculate before_mean and after_mean
        assert not 'before_mean' in peak    #Fails if you run the plugin twice!
        # This seems more Xerawdp-like, but then we get messy overlapping peaks... have to test for those first...
        left_to_use  = min(peak['left'],  peak['prepeak_left'])  if peak['peak_type'] == 'large_s2' else peak['left']
        right_to_use = max(peak['right'], peak['prepeak_right']) if peak['peak_type'] == 'large_s2' else peak['right']
        #left_to_use  = peak['prepeak_left']  if peak['peak_type'] == 'large_s2' else peak['left']
        #right_to_use =  peak['prepeak_right'] if peak['peak_type'] == 'large_s2' else peak['right']

        peak['before_mean'] = np.mean(
            signal[max(0, left_to_use - settings['test_before']): left_to_use])
        peak['after_mean'] = np.mean(

            signal[right_to_use: min(len(signal), right_to_use + settings['test_after'])])

        #Do the testing
        if peak['before_mean'] > settings['before_to_height_ratio_max'] * peak['height'] \
            or peak['after_mean'] > settings['after_to_height_ratio_max'] * peak['height']:
            return 'peak is not isolated enough' #Todo: add stuff
            #return '%s samples before peak contain stuff (mean %s, which is more than %s (%s x peak height))' % (settings['test_before'], peak['before_mean'], settings['before_to_height_ratio_max'] * peak['height'], settings['before_to_height_ratio_max'])
            #return '%s samples after peak contain stuff (mean %s, which is more than %s (%s x peak height))' % (settings['test_after'], peak['after_mean'], settings['after_to_height_ratio_max'] * peak['height'], settings['after_to_height_ratio_max'])
        return None


# class PruneWideS1s(PeakPruner):
#
#     def decide_peak(self, peak, event, peak_index):
#         if peak['peak_type'] != 's1':
#             return None
#         filtered_wave = event['processed_waveforms']['filtered_for_large_s2']
#         max_in_filtered = peak['filtered_for_large_s2']['position_of_max_in_waveform']
#         filtered_width = extent_until_threshold(filtered_wave,
#                                                 start=max_in_filtered,
#                                                 threshold=0.25*filtered_wave[max_in_filtered])
#         if filtered_width > 50:
#             return 'S1 FWQM in filtered_wv is %s samples, higher than 50.' % filtered_width
#         return None


class PruneWideShallowS2s(PeakPruner):

    def decide_peak(self, peak, event, peak_index):
        if str(peak['peak_type']) != 'small_s2':
            return None
        threshold = 0.062451  # 1 mV/bin = 0.1 mV/ns
        peakwidth = (peak['right'] - peak['left']) / units.ns
        ratio = peak['top_and_bottom']['height'] / peakwidth
        if ratio > threshold:
            return 'Max/width ratio %s is higher than %s' % (ratio, threshold)
        return None


# class PruneS1sWithNearbyNegativeExcursions(PeakPruner):
#
#     def decide_peak(self, p, event, peak_index):
#         if p['peak_type'] != 's1':
#             return None
#         data = event['processed_waveforms']['top_and_bottom']
#         negex = p['lowest_nearby_value'] = min(data[
#             max(0,p['index_of_max_in_waveform']-500) :
#             min(len(data)-1,p['index_of_max_in_waveform'] + 101)
#         ])  #Window used by s1 filter: todo: don't hardcode
#         maxval =  p['uncorrected_sum_waveform_for_xerawdp_matching']['height']
#         factor = 3
#         if not maxval > factor * abs(negex):
#             return 'Nearby negative excursion of %s, height (%s) not at least %s x as large.' % (negex, maxval, factor)
#         return None


class PruneS1sInS2Tails(PeakPruner):

    def decide_peak(self, peak, event, peak_index):
        if peak['peak_type'] != 's1':
            return None
        if not 'stop_looking_for_s1s_after' in event:
            # Determine where to stop looking for S1s
            # Certainly stop looking after the largest S2 (XerawDP behaviour)
            s2areas = [p['top_and_bottom']['area'] for p in event['peaks'] if is_s2(p)]
            if s2areas == []:
                # No S2s in this waveform - S1 always ok
                event['stop_looking_for_s1s_after'] = float('inf')
                return None
            # DANGER ugly code ahead...
            s2maxarea = max(s2areas)
            for i, p in enumerate(event['peaks']):
                if p['top_and_bottom']['area'] == s2maxarea:
                    event['stop_looking_for_s1s_after'] = p['left']
            # Stop earlier if there is an earlier S2 whose amplitude exceeds a treshold
            treshold = 3.12255  # S2 amplitude after which no more s1s are looked for
            larges2boundaries = [p['left'] for p in event['peaks'] if is_s2(p) and p['top_and_bottom']['height'] > treshold]
            if larges2boundaries != []:
                event['stop_looking_for_s1s_after'] = min(event['stop_looking_for_s1s_after'], min(larges2boundaries))
        if peak['left'] > event['stop_looking_for_s1s_after']:
            return 'S1 starts at %s, which is beyond %s, the starting position of a "large" S2.' % (peak['left'], event['stop_looking_for_s1s_after'])
        return None


class PruneS2sInS2Tails(PeakPruner):

    def decide_peak(self, peak, event, peak_index):
        if peak['peak_type'] != 'small_s2':
            return None
        if not 'stop_looking_for_s2s_after' in event:
            # Determine where to stop looking for S2s
            # Stop if there is an earlier S2 whose amplitude exceeds a treshold
            treshold = 624.151  # S2 amplitude after which no more s2s are looked for
            larges2boundaries = [p['left'] for p in event['peaks'] if is_s2(p) and p['top_and_bottom']['height'] > treshold]
            if larges2boundaries == []:
                # No large S2s in this waveform - S2 always ok
                event['stop_looking_for_s2s_after'] = float('inf')
                return None
            event['stop_looking_for_s2s_after'] = min(larges2boundaries)
        if peak['left'] > event['stop_looking_for_s2s_after']:
            return 'S2 starts at %s, which is beyond %s, the starting position of a "large" S2.' % (peak['left'], event['stop_looking_for_s2s_after'])
        return None


class PruneS2sInS2Tails(PeakPruner):

    def decide_peak(self, peak, event, peak_index):
        if peak['peak_type'] != 'small_s2':
            return None
        treshold = 624.151  # S2 amplitude after which no more s2s are looked for
        if not hasattr(self, 'earliestboundary'):
            s2boundaries = [p['left'] for p in event['peaks'] if is_s2(p) and p['top_and_bottom']['height'] > treshold]
            if s2boundaries == []:
                self.earliestboundary = float('inf')
            else:
                self.earliestboundary = min(s2boundaries)
        if peak['left'] > self.earliestboundary:
            return 'Small S2 starts at %s, which is beyond %s, the starting position of a "large" S2.' % (peak['left'], self.earliestboundary)
        return None

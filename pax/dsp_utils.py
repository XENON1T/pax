"""
Helper functions for peakfinders and other plugins that look at waveforms
Don't put in hacks for Xerawdp matching!
"""


def main():
    #Todo: move this to tests!
    w = list(range(10))+list(reversed(range(10)))
    print(w)
    # Left and right walking
    assert walk_waveform(w, 3, IndexReached(7), direction='right')[1] == 7
    assert walk_waveform(w, 7, IndexReached(3), direction='left')[1] == 3
    # Threshold crossing
    assert walk_waveform(w, 7, IsBelowThreshold(5, min_crossing_length=3), direction='left')[1] == 4
    assert walk_waveform(w, 7, IsBelowThreshold(5, min_crossing_length=3), direction='right')[1] == 15
    #Todo: add more here
    # All conditions failing
    assert isinstance(walk_waveform(w, 7, IsBelowThreshold(-5))[0], WaveformEndReached)
    assert walk_waveform(w, 7, IsBelowThreshold(-5))[1] == len(w)-1
    assert isinstance(walk_waveform(w, 7, IsBelowThreshold(-5), direction='left')[0], WaveformEndReached)
    assert walk_waveform(w, 7, IsBelowThreshold(-5), direction='left')[1] == 0
    # Several conditions
    assert walk_waveform(w, 3, [IsAboveThreshold(8), IndexReached(7)])[1] == 7




##
## Waveform walking
##
def walk_waveform(signal, start, stop_conditions, direction="right"):
    d = {
        'last_possible_index': len(signal)-1,
        'previous_index': float('nan'),
        'previous_value': float('nan'),    #Nnumerical comparisons with float('nan') always give False
    }
    # Argument checking
    if 0 <= start <= d['last_possible_index']:
        d['current_value'] = signal[start]
        d['current_index'] = start
    else:
        raise ValueError("Start index %s not between %s and %s!" % (start, 0, d['last_possible_index']))
    if direction in ('left','right'):
        d['direction'] = direction
    else:
        raise ValueError("Invalid direction: %s" % direction)
    if isinstance(stop_conditions, StopCondition):
        # Only once stop condition specified, that's ok
        stop_conditions = (stop_conditions,)
    while 1:
        if not 0 <= d['current_index'] <= d['last_possible_index']:
            # We've gone too far!
            # Todo: Complain
            return WaveformEndReached(), d['previous_index']    #()'s are so isinstance tests work
        d['current_value'] = signal[ d['current_index'] ]
        # Check all the stop_conditions in turn
        for stop_condition in stop_conditions:
            result = stop_condition.evaluate(d)
            if result is True:
                # The stop condition evaluates to True, i.e. we can stop here!
                return stop_condition, d['current_index']
            if result is not False:
                 # The condition has specified a special return result
                return stop_condition, result
        d['previous_index'] = d['current_index']
        d['previous_value'] = d['current_value']
        d['current_index'] += -1 if direction == 'left' else 1

class StopCondition(object):

    def startup(self, d):
        pass

    def evaluate(self, d):
        raise NotImplementedError

# #Setup syntax sugar for & and | or on stop conditions...
# Overkill, you only need OR anyway, will slow things down if in long chain
#
#
# class CompositeCondition(StopCondition):
#     """Condition composed of several conditions, use only for syntax sugar"""
#     def __init__(self, conditions, join_operator):
#         self.conditions = conditions
#         self.join_operator = join_operator
#
#     def startup(self, d):
#         [c.startup(d) for c in self.conditions]
#
#     def evaluate(self, d):
#         for c in conditions:
#             c.evaluate(d)
#             if self.join_operator == 'or':
#             any([c.evaluate(d) for c in self.conditions])
#         elif self.join_operator == 'and':
#             all([c.evaluate(d) for c in self.conditions])
#
# def join_conditions_by_and(condition1, condition2):  CompositeCondition([condition1,condition2],'and')
# def join_conditions_by_or(condition1, condition2):   CompositeCondition([condition1,condition2],'or')
# StopCondition.__and__ = join_conditions_by_and
# StopCondition.__or__  = join_conditions_by_or

class WaveformEndReached(StopCondition):
    # Used by waveform walker to prevent crashes
    def evaluate(self, d):
        raise ValueError("You shouldn't include this condition in the stop_conditions list!")


class StartValueExceeded(StopCondition):
    def evaluate(self, d):
        return d['current_value'] > d['start_value']


class IndexReached(StopCondition):
    def __init__(self, limit_index):
        self.limit_index = limit_index

    def startup(self, d):
        if (
            d['direction'] == 'left' and d['start_index'] < self.limit_index or
            d['direction'] == 'right'and d['start_index'] > self.limit_index
        ):
            pass
            # Already beyond intended endpoint!
        if self.limit_index < 0:
            pass
            # Crash occurs before condition is satisfied!
        if self.limit_index > d['last_possible_index']:
            pass
            # Crash occurs before condition is satisfied!

    def evaluate(self, d):
        if d['direction'] == 'left' and d['current_index'] < self.limit_index:
            raise RuntimeError("index overshoot! elaborate")
        if d['direction'] == 'right' and d['current_index'] > self.limit_index:
            raise RuntimeError("index overshoot! elaborate")
        return d['current_index'] == self.limit_index


class ThresholdCrossed(StopCondition):
    def __init__(self, threshold, min_crossing_length=1, cross_to='either', real_crossing=True, ):
        self.threshold = threshold
        if min_crossing_length < 1 or not isinstance(min_crossing_length, int):
            raise ValueError("Crossing length must be an integer > 1")
            #Well, could allow 0 and tell walker condition is already satisfied
        self.min_crossing_length = min_crossing_length
        self.real_crossing = real_crossing
        self.cross_to = cross_to
        if self.cross_to not in ('either', 'above', 'below'):
            raise ValueError("Invalid cross to argument: %s" % self.cross_to)
        self.crossing_length = 0

    def evaluate(self, d):
        if self.real_crossing:
            crossed_above = d['previous_value'] < self.threshold < d['current_value']
            crossed_below = d['current_value']  < self.threshold < d['previous_value']
        else:
            crossed_above = self.threshold < d['current_value']
            crossed_below = d['current_value'] < self.threshold
        if self.cross_to == 'above':
            crossed = crossed_above
        elif self.cross_to == 'below':
            crossed = crossed_below
        else: #self.cross_to is 'either':
            crossed = (crossed_above or crossed_below)
        if crossed:
            self.crossing_length += 1
            if self.crossing_length == self.min_crossing_length:
                # We're done! Return the index of the original crossing
                return d['current_index'] + (1-self.min_crossing_length)*(-1 if d['direction'] == 'left' else 1)
        else:
            self.crossing_length = 0
        return False

# Convenience forms
class CrossAboveThreshold(ThresholdCrossed):
    def __init__(self, threshold, min_crossing_length=1):
        ThresholdCrossed.__init__(self, threshold, min_crossing_length, cross_to='above', real_crossing=True)

class CrossBelowThreshold(ThresholdCrossed):
    def __init__(self, threshold, min_crossing_length=1):
        ThresholdCrossed.__init__(self, threshold, min_crossing_length, cross_to='below', real_crossing=True)

class IsAboveThreshold(ThresholdCrossed):
    def __init__(self, threshold, min_crossing_length=1):
        ThresholdCrossed.__init__(self, threshold, min_crossing_length, cross_to='above', real_crossing=False)

class IsBelowThreshold(ThresholdCrossed):
    def __init__(self, threshold, min_crossing_length=1):
        ThresholdCrossed.__init__(self, threshold, min_crossing_length, cross_to='below', real_crossing=False)


class XerawdpSlopeInversionCondition(StopCondition):
    pass


##
## Width computations
##


if __name__ == '__main__':
    main()
"""Strax conversion functions
"""

import numpy as np
import numba


def records_needed(pulse_length, samples_per_record):
    """Return records needed to store pulse_length samples"""
    return 1 + (pulse_length - 1) // samples_per_record


def sort_by_time(x):
    """Sort pulses by time, then channel.
    Assumes you have no more than 10k channels, and records don't span
    more than 100 days. TODO: FIX this
    """
    # I couldn't get fast argsort on multiple keys to work in numba
    # So, let's make a single key...
    if not len(x):
        return
    sort_key = (x['time'] - x['time'].min()) * 10000 + x['channel']
    sort_i = np.argsort(sort_key)
    return x[sort_i]


def record_dtype(samples_per_record):
    """Data type for a waveform record.
    Length can be shorter than the number of samples in data,
    this indicates a record with zero-padding at the end.
            """
    return [
        (('Channel/PMT number',
          'channel'), np.int16),
        (('Time resolution in ns',
          'dt'), np.int16),
        (('Start time of the interval (ns since unix epoch)',
          'time'), np.int64),
        # Don't try to make O(second) long intervals!
        (('Length of the interval in samples',
          'length'), np.int32),
        # Sub-dtypes MUST contain an area field
        # However, the type varies: float for sum waveforms (area in PE)
        # and int32 for per-channel waveforms (area in ADC x samples)
        (("Integral in ADC x samples",
          'area'), np.int32),
        # np.int16 is not enough for some PMT flashes...
        (('Length of pulse to which the record belongs (without zero-padding)',
          'pulse_length'), np.int32),
        (('Fragment number in the pulse',
          'record_i'), np.int16),
        (('Baseline in ADC counts. data = int(baseline) - data_orig',
          'baseline'), np.float32),
        (('Level of data reduction applied (strax.ReductionLevel enum)',
          'reduction_level'), np.uint8),
        # Note this is defined as a SIGNED integer, so we can
        # still represent negative values after subtracting baselines
        (('Waveform data in ADC counts above baseline',
          'data'), np.int16, 110)
    ]


@numba.jit(nopython=True, nogil=True, cache=True)
def baseline(records, baseline_samples=40):
    """Subtract pulses from int(baseline), store baseline in baseline field
    :param baseline_samples: number of samples at start of pulse to average
    Assumes records are sorted in time (or at least by channel, then time)

    Assumes record_i information is accurate (so don't cut pulses before
    baselining them!)
    """
    samples_per_record = len(records[0]['data'])

    # Array for looking up last baseline seen in channel
    # We only care about the channels in this set of records; a single .max()
    # is worth avoiding the hassle of passing n_channels around
    last_bl_in = np.zeros(records['channel'].max() + 1, dtype=np.int16)

    for d in enumerate(records):

        # Compute the baseline if we're the first record of the pulse,
        # otherwise take the last baseline we've seen in the channel
        if d.record_i == 0:
            bl = last_bl_in[d.channel] = d.data[:baseline_samples].mean()
        else:
            bl = last_bl_in[d.channel]

        # Subtract baseline from all data samples in the record
        # (any additional zeros should be kept at zero)
        last = min(samples_per_record,
                   d.pulse_length - d.record_i * samples_per_record)
        d.data[:last] = int(bl) - d.data[:last]
        d.baseline = bl
    return records


def integrate(records):
    for i, r in enumerate(records):
        records[i]['area'] = r['data'].sum()
    return records


def pax_to_records(self, samples_per_record=110, events_per_chunk=10):
    results = []
    samples_per_record = 110

    def finish_results():
        nonlocal results
        records = np.concatenate(results)
        if len(records) != 0:
            # In strax data, records are always stored
            # sorted, baselined and integrated
            records = sort_by_time(records)
            records = baseline(records)
            records = integrate(records)
            results = []
        return records

    pulse_lengths = np.array([p.length
                              for p in self.pulses])
    n_records_tot = records_needed(pulse_lengths, samples_per_record).sum()
    records = np.zeros(n_records_tot,
                       dtype=record_dtype(samples_per_record))
    output_record_index = 0  # Record offset in data

    for p in self.pulses:
        n_records = records_needed(p.length, samples_per_record)
        for rec_i in range(n_records):
            r = records[output_record_index]
            r['time'] = (self.start_time
                         + p.left * 10
                         + rec_i * samples_per_record * 10)
            r['channel'] = p.channel
            r['pulse_length'] = p.length
            r['record_i'] = rec_i
            r['dt'] = 10

            # How much are we storing in this record?
            if rec_i != n_records - 1:
                # There's more chunks coming, so we store a full chunk
                n_store = samples_per_record
                if not p.length > samples_per_record * (rec_i + 1):
                    raise AssertionError()
            else:
                    # Just enough to store the rest of the data
                    # Note it's not p.length % samples_per_record!!!
                    # (that would be zero if we have to store a full record)
                n_store = p.length - samples_per_record * rec_i

            if not 0 <= n_store <= samples_per_record:
                raise AssertionError()
            r['length'] = n_store

            offset = rec_i * samples_per_record
            r['data'][:n_store] = p.raw_data[offset:offset + n_store]
            output_record_index += 1

    results.append(records)

    if len(results):
        records = finish_results()
        return records

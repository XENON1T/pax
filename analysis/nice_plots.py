import numpy as np

import matplotlib  # Needed for font size spec, color map transformation function bla bla

matplotlib.rc('font', size=16)
import matplotlib.pyplot as plt

try:
    from pax.units import us
except ImportError:
    ns = 1
    us = 1000

# import seaborn as sns   # Sets different, maybe nicer style
# sns.set(style="ticks")

# Field range, log scale info for variables
fields_in_dataset = {
    'peaks': {
        'area': {
            'range': (1, 10 ** 6),
            'logscale': True,
            'label': 'Peak area (pe)',
        },
        'hit_time_std': {
            'range': (10, 4 * us),
            'logscale': True,
            'label': 'Hit time std (ns)',
        },
        'area_fraction_top': {
            'range': (-0.01, 1.01),
            'label': 'Fraction of area in top PMTs',
        },
        'median_absolute_deviation': {
            'range': (10, 10 ** 4),
            'logscale': True,
            'label': 'Hit time MAD (ns)',
        },
        'hit_time_mean': {
            'range': (0, 350 * us),
            'label': 'Peak time (hit time mean, ns)',
        },
        'full_range': {
            'range': (10, 20 * us),
            'logscale': True,
            'label': 'Peak width (full hit range, ns)',
        },
        'top_hitpattern_spread': {
            'range': (0.01, 6),
            'label': 'Spread of top hitpattern (cm)',
        },
    },
    'events': {
        'drift_time': {
            'range': (0 * us, 80 * us),
            'label': 'Drift time (ns)',
        },
        's2_hit_time_std': {
            'range': (0, 3 * us),
        },
        's2_area': {
            'range': (1000, 10 ** 6),
            'logscale': True,
            'label': 'Peak area (pe)',
        },

    }
}

# Copy field info from peaks to s1_... and s2_... in event
for pt in ('s1', 's2'):
    for fname, finfo in fields_in_dataset['peaks'].items():
        newlabel = "%s %s" % (pt.capitalize(), finfo['label'])
        event_fname = "%s_%s" % (pt, fname)
        # Some custom field info may already be present (e.g. custom range for s1 area),
        # if so, don't overwrite it
        event_finfo = fields_in_dataset['events'].get(event_fname, finfo.copy())
        # Update the label in any case
        event_finfo['label'] = newlabel
        fields_in_dataset['events'][event_fname] = event_finfo


class NicePlot(object):
    dimlabels = ['x', 'y', 'z']
    dimensions = 0
    clip_data_outside_range = False
    remove_data_outside_range = False
    mask = None

    def __init__(self, all_data, dataset_name=None, **kwargs):

        self.colormap = kwargs.get('colormap', plt.cm.jet)

        for dl in self.dimlabels[:self.dimensions]:

            # Get the field names for each dimension
            if not dl in kwargs:
                raise ValueError('No data for dimension %s specified!' % dl)
            fieldname = kwargs[dl]
            setattr(self, dl, fieldname)

            if dataset_name is not None:
                field_info = fields_in_dataset[dataset_name].get(fieldname, {})
            else:
                field_info = {}

            # Get data, label, logscale values
            the_data = all_data[fieldname]
            setattr(self, dl + '_data', the_data)
            setattr(self, dl + '_label', field_info.get('label', fieldname))
            if dl + '_logscale' in kwargs:
                setattr(self, dl + '_logscale', kwargs[dl + '_logscale'])
            else:
                setattr(self, dl + '_logscale',
                        field_info.get('logscale', False))

            # Load defaults for x, y, z limits, if needed
            d_range = kwargs.get(dl + '_range', (None, None))
            # If user did not specify a range,
            # and we have a default range for this field, use it
            if d_range == (None, None) and 'range' in field_info:
                d_range = field_info['range']
            # If the user specified automatic, the plot should take care of range determination
            if d_range == 'Automatic':
                d_range = (None, None)
            # If the range has not been fixed, but it must be fixed for the plot we're going to make
            # we'll now determine the range from the data
            if d_range == (None, None) and (self.remove_data_outside_range or self.clip_data_outside_range):
                d_range = (np.min(the_data), np.max(the_data))
            setattr(self, dl + '_range', d_range)

            if self.clip_data_outside_range:
                setattr(self, dl + '_data',
                        np.clip(getattr(self, dl + '_data'), d_range[0], d_range[1]))

            # If we must remove data, update the mask
            if self.remove_data_outside_range:
                d = getattr(self, dl + '_data')
                mask = (d >= d_range[0]) & (d <= d_range[1])
                if self.mask is None:
                    self.mask = mask
                else:
                    self.mask = mask & self.mask

        # Remove data outside range, if we were asked to do so
        if self.mask is not None:
            for dl in self.dimlabels[:self.dimensions]:
                setattr(self,
                        dl + '_data',
                        getattr(self,
                                dl + '_data')[self.mask])


class ColoredScatterPlot(NicePlot):
    dimensions = 3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grainsize = kwargs.get('grainsize', 5)
        self.grainalpha = kwargs.get('grainalpha', 1)

    def plot(self):

        # Draw figure
        plt.figure(figsize=(10, 7))
        plt.scatter(self.x_data, self.y_data, c=self.z_data,
                    marker='.', edgecolors='none',
                    s=self.grainsize, alpha=self.grainalpha,
                    vmin=self.z_range[0], vmax=self.z_range[1],
                    norm=(matplotlib.colors.LogNorm() if self.z_logscale else None),
                    cmap=self.colormap)
        plt.colorbar(label=self.z_label)

        # Set limits
        if self.x_logscale:
            plt.xscale('log')
        if self.y_logscale:
            plt.yscale('log')
        plt.xlim(*self.x_range)
        plt.ylim(*self.y_range)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)

        plt.show()


class DensityPlot(NicePlot):
    dimensions = 2
    # Extent options don't seem to like logscales, so:
    remove_data_outside_range = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count_logscale = kwargs.get('count_logscale', False)
        self.gridsize = kwargs.get('gridsize', 100)

    def plot(self):
        plt.figure(figsize=(10, 7))
        plt.hexbin(self.x_data, self.y_data,
                   gridsize=self.gridsize,
                   xscale='log' if self.x_logscale else 'linear',
                   yscale='log' if self.y_logscale else 'linear',
                   bins='log' if self.count_logscale else None,
                   cmap=self.colormap)

        # Set labels
        plt.colorbar(label=('Log10 ' if self.count_logscale else '') + 'Count')
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.show()


class BasicHist(NicePlot):
    dimensions = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bins = kwargs.get('n_bins', 10)
        self.count_logscale = kwargs.get('count_logscale', False)
        self.cumulative = kwargs.get('cumulative', False)
        self.normed = kwargs.get('normed', False)
        self.hist_type = kwargs.get('hist_type', 'step')

    def plot(self):
        if self.x_logscale:
            bins = 10 ** np.linspace(np.log10(self.x_range[0]), np.log10(self.x_range[1]))
        else:
            bins = np.linspace(self.x_range[0], self.x_range[1], self.n_bins)

        plt.hist(self.x_data,
                 range=self.x_range,
                 bins=bins,
                 cumulative=self.cumulative,
                 normed=self.normed,
                 histtype=self.hist_type)

        if self.x_logscale:
            plt.xscale('log')

        if self.count_logscale:
            plt.yscale('log')

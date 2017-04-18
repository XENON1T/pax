__author__ = 'axel'
import matplotlib.pyplot as plt
plt.rc('font', size=16)

def make_plot(hits_df, field, xaxislabel = "x-axis", yaxislabel = "y-axis", nof_bins=500, xlim=(-1,5000), ylim= None, legendlabel=None, axes = None, **kwargs):
    if axes is not None:
        ax = axes
    else:
        fig, ax = plt.subplots(nrows=7, ncols=2, sharex=True, sharey=True, squeeze=False, figsize=(20,14))
    plt.tight_layout()
    for i, axi in enumerate(ax.reshape(-1)):
        #ax[6][0].set_xlabel(xaxislabel)
        #ax[6][1].set_xlabel(xaxislabel)
            axi.set_xlim(xlim)
            if ylim:
                axi.set_ylim(ylim)
            axi.set_ylabel(yaxislabel)
            axi.set_yscale("log")
            try:
                axi.hist([xhits[field] for xhits in hits_df if (xhits["channel"] == i)],
                                #range=(0,10),
                                bins=nof_bins,
                                histtype="step",
                                label=legendlabel,
                                **kwargs)
            except ValueError:
                axi.annotate("no hits", xy=(0.5,0.5), xycoords="axes fraction", horizontalalignment="center")

            axi.annotate("Channel # %i"%i, xy=(0.8, 0.8), xycoords="axes fraction")

    #return fig


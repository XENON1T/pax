"""Form clutsers in (x,y,t)
"""
from pax import plugin
import numpy
from sklearn.cluster import DBSCAN
import numpy as np

from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


class DBSCANCluster(plugin.TransformPlugin):

    def __init__(self, config):
        plugin.TransformPlugin.__init__(self, config)

        self.topArrayMap = config['topArrayMap']

    def transform_event(self, event):
        self.log.fatal(event.keys())

        X = []

        # need function to get PMT location...
        for pmt, data in event['channel_waveforms'].items():
            if pmt not in self.topArrayMap.keys():
                continue
            pmt_location = self.topArrayMap[pmt]
            indices = np.flatnonzero(data)

            for index in indices:
                # for j in range(int(data[index])):
                X.append([pmt_location['x'],
                          pmt_location['y']])

            # print(X)

        ##############################################################################
        # Compute DBSCAN
        db = DBSCAN().fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        print('Estimated number of clusters: %d' % n_clusters_)

        ##############################################################################
        # Plot result
        import matplotlib.pyplot as plt

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = 'k'

            class_member_mask = (labels == k)

            xy = X[class_member_mask & core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=14)

            xy = X[class_member_mask & ~core_samples_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                     markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.show()

        return event

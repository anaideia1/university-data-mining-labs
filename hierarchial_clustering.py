import numpy as np


class HierarchialClustering:
    def __init__(self, n_clusters, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None

    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.arange(n_samples)

        while len(np.unique(self.labels_)) > self.n_clusters:
            print(f'Temporary number of clasters is {len(np.unique(self.labels_))} (With point labels {self.labels_})')
            min_distance = np.inf
            merge_index = None

            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if self.labels_[i] != self.labels_[j]:
                        distance = self._calculate_distance(X[i], X[j])

                        if distance < min_distance:
                            min_distance = distance
                            merge_index = (i, j)

            i, j = merge_index
            self._merge_clusters(i, j)
        print(f'Result number of clasters is {len(np.unique(self.labels_))} (With point labels {self.labels_})')

    def _calculate_distance(self, point1, point2):
        if self.linkage == 'single':
            # Single-linkage (minimum distance)
            return np.min(np.abs(point1 - point2))
        elif self.linkage == 'complete':
            # Complete-linkage (maximum distance)
            return np.max(np.abs(point1 - point2))
        elif self.linkage == 'average':
            # Average-linkage (average distance)
            return np.mean(np.abs(point1 - point2))
        elif self.linkage == 'centroid':
            # Centroid-linkage (distance between centroids)
            centroid1 = np.mean(point1)
            centroid2 = np.mean(point2)
            return np.abs(centroid1 - centroid2)

    def _merge_clusters(self, i, j):
        print(f'Merging clusters number {i} and {j}...')
        cluster_i = self.labels_[i]
        cluster_j = self.labels_[j]

        for k in range(len(self.labels_)):
            if self.labels_[k] == cluster_j:
                self.labels_[k] = cluster_i


# Example usage
# A 3 3
# B 3 2
# C 4 2
# D 1 5
# E 1 3
# F 5 1
# G 5 6
# H 4 6
# I 5 2
# J 2 1
X = np.array([[3, 3],
              [3, 2],
              [4, 2],
              [1, 5],
              [1, 3],
              [5, 1],
              [5, 6],
              [4, 6],
              [5, 2],
              [2, 1]])
n_clusters = 3

print('------------------------------------------------')
print('Single-linkage hierarchical clustering')
single_link_clustering = HierarchialClustering(n_clusters, linkage='single')
single_link_clustering.fit(X)
print("Single-linkage clusters:", single_link_clustering.labels_)

print('------------------------------------------------')
print('Complete-linkage hierarchical clustering')
complete_link_clustering = HierarchialClustering(n_clusters, linkage='complete')
complete_link_clustering.fit(X)
print("Complete-linkage clusters:", complete_link_clustering.labels_)

print('------------------------------------------------')
print('Average-linkage hierarchical clustering')
average_link_clustering = HierarchialClustering(n_clusters, linkage='average')
average_link_clustering.fit(X)
print("Average-linkage clusters:", average_link_clustering.labels_)

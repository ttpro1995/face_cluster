from face_pic import FacePic
from collections import *

class Clusterspace:
    def __init__(self, clusters):
        self.clusters_space = []
        self.distances = defaultdict()
        self.clusters_space += clusters
        self.calculate_distance()

    def add_clusters(self, clusters):
        """

        :param clusters: a list of clusters
        :return:
        """
        self.clusters_space +=clusters
        self.calculate_distance()

    def calculate_distance(self, force_recalculate = False):
        # idx_a < idx_b
        for idx_a, cluster_a in enumerate(self.clusters_space):
            for idx_b, cluster_b in enumerate(self.clusters_space):
                if (idx_a >= idx_b):
                    continue # only do for idx_a < idx_b
                if ((idx_a, idx_b) in self.distances.keys()) and (not force_recalculate):
                    continue # already exist, skip
                self.distances[idx_a, idx_b] = cluster_a.distance(cluster_b)

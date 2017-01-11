from face_pic import FacePic
from collections import *


def _find_key(my_dict, value):
    """
    :param my_dict: defaultdict
    :param value: a value you want to find
    :return:
    key of that value
    """
    my_dict_val = my_dict.values()
    my_dict_item = my_dict.items()

    idx = my_dict_val.index(value)
    key = my_dict_item[idx][0]

    return key

class Clusterspace:
    def __init__(self, clusters):
        self.clusters_space = []
        self.distances = defaultdict()
        self.clusters_space += clusters
        self.calculate_distance()

    def add_clusters(self, clusters):
        """
        will ignore empty cluster
        :param clusters: a list of clusters
        :return:
        """
        for cl in clusters:
            if len(cl) == 0:
                clusters.remove(cl) # do not add empty cluster

        self.clusters_space +=clusters
        self.calculate_distance()


    def calculate_distance(self, force_recalculate = False):
        """
        Calculate the distance matrix
        :param force_recalculate: recalculate all exist valid cluster
        :return:
        """

        # idx_a < idx_b
        for idx_a, cluster_a in enumerate(self.clusters_space):
            for idx_b, cluster_b in enumerate(self.clusters_space):
                if (idx_a >= idx_b): # not for this, prevent dublicate
                    continue # only do for idx_a < idx_b
                if ((idx_a, idx_b) in self.distances.keys()): # when distance is exist
                    if (not cluster_a.working) or (not cluster_b.working):
                        del self.distances[idx_a, idx_b] # delete distance of not working cluster
                    if (not force_recalculate):
                        continue # already exist, skip
                if (not cluster_a.working) or (not cluster_b.working):
                    continue # skip those not working
                if (not cluster_a.is_mergeable(cluster_b)):
                    continue # skip if not mergeable

                self.distances[idx_a, idx_b] = cluster_a.distance(cluster_b)


    def merge_closest(self, threshold):
        """
        Merge pair of closest cluster
        until closest vector distance > threshold
        :return:
        """
        self.calculate_distance()
        min_dist = min(self.distances.values())
        keys = _find_key(self.distances,min_dist)

        while(min_dist < threshold):
            idx_a = keys[0]
            idx_b = keys[1]
            print ('merge ',idx_a, idx_b)
            new_cluster, child_cluster = self.clusters_space[idx_a].merge(self.clusters_space[idx_b], True)
            self.add_clusters([new_cluster, child_cluster])

            ## redo the calculation
            self.calculate_distance()
            min_dist = min(self.distances.values())
            keys = _find_key(self.distances, min_dist)

    def getWorkingCluster(self):
        """

        :return: a list of working cluster
        """
        working_cluster = []
        for cluster in self.clusters_space:
            if cluster.working == True:
                working_cluster.append(cluster)

        return working_cluster
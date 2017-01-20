from face_pic import FacePic
from face_cluster import FaceCluster
from collections import *
import itertools
from os import listdir
import numpy as np
from os.path import isfile
import dill
import pickle
from Encoder import encoder
from munkres import Munkres

# E = encoder()
RECOGNISE_THRESHOLD = 0.9
MERGE_DISTANCE_THRESHOLD = 2.0
INCREMENTAL_TRAIN_THRESHOLD = 20
TRAIN_THRESHOLD = 40
SMALLEST_SIZE_NAME_THRESHOLD = 8


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
    """
    self.host: biggest cluster
    self.distances: distance matrix of cluster
    self.data_dir: string
    """
    def __init__(self, data_dir, clusters = None, mac = None):
        self.mac = mac
        self.data_dir = data_dir
        self.clusters_space = []
        self.distances = defaultdict()
        self.frams = set()
        self.host = None
        if clusters:
            self.clusters_space += clusters
            self.calculate_distance()

    def add_clusters(self, clusters):
        """
        will ignore empty cluster
        :param clusters: a list of clusters
        :return:
        """
        for cl in clusters:
            if len(cl) != 0:
                self.clusters_space.append(cl)

        self.calculate_distance()



    def update_fram(self):
        cur_frams = set()
        for frame in self.data_dir:
            cur_frams.add(frame)
        return cur_frams

    def renew(self):
        self.clusters_space = []
        self.distances = defaultdict()
        self.frams = set()

    def save_model(self):
        d = defaultdict()
        d["clusters_space"] = self.clusters_space
        d["data_dir"] = self.data_dir
        d["distances"] = self.distances
        d["frams"] = self.frams
        pickle.dump(d, open("trained_space.p", "wb"), protocol=2)

    def load_model(self):
        space = pickle.load(open("trained_space.p", "rb"))
        self.clusters_space = space["clusters_space"]
        self.data_dir = space["data_dir"]
        self.distances = space["distances"]
        self.frams = space["frams"]

    def incremental_train(self):
        print "-----------------------------INCREMENTAL TRAINING----------------------------------"
        new_clusters = []
        newframs = self.update_fram() - self.frams
        if len(newframs) == 0:
            return
        for fram in newframs:
            f_path = self.data_dir + "/" + fram
            clusters = self.read_fram(f_path)
            new_clusters.extend(clusters)
            self.frams.add(fram)
        self.add_clusters(new_clusters)
        self.merge_closest(MERGE_DISTANCE_THRESHOLD)
        self.name()
        self.save_model()

    def find_host_cluster(self):
        """
        Find biggest cluster
        update self.host
        :return:
        """
        biggest_cluster = max(self.clusters_space, key=len)
        self.host = biggest_cluster
        return self.host

    # def train(self):
    #     print "#fram = " + str(len(self.update_fram()))
    #     if isfile("trained_space.p") and len(self.clusters_space) == 0:
    #         self.load_model()
    #
    #     newframs = self.update_fram() - self.frams
    #     if len(newframs) > 0 and len(self.update_fram())%TRAIN_THRESHOLD == 0:
    #         print "-----------------------------TRAINING----------------------------------"
    #         self.renew()
    #
    #         data = defaultdict(lambda :defaultdict(list))
    #         path2rep = defaultdict(lambda :0)
    #         for frame in listdir(self.data_dir):
    #             for cluster in listdir(self.data_dir + "/" + frame):
    #                 for pic in listdir(self.data_dir + "/" + frame + "/" + cluster):
    #                     path = self.data_dir + "/" + frame + "/" + cluster + "/" + pic
    #                     data[frame][cluster].append(path)
    #                     path2rep[path] = E.get_rep_preprocessed(path)
    #
    #         clusters = []
    #         for frame in data.keys():
    #             for cluster in data[frame].keys():
    #                 FacePics = set()
    #                 for path in data[frame][cluster]:
    #                     FacePics.add(FacePic(path2rep[path], frame, path))
    #                 clusters.append(FaceCluster(FacePics))
    #             self.frams.add(frame)
    #         self.add_clusters(clusters)
    #         self.merge_closest(MERGE_DISTANCE_THRESHOLD)
    #         self.name()
    #         self.save_model()
    #
    #     elif len(self.update_fram()) > TRAIN_THRESHOLD \
    #             and len(newframs)%INCREMENTAL_TRAIN_THRESHOLD == 0  \
    #             and len(newframs)%TRAIN_THRESHOLD != 0:
    #             self.incremental_train()
    #     else:
    #         return


    def is_matchable(self):
        pass

    def show__working_cluster(self):
        _, idx = self.getWorkingCluster()
        for i in idx:
            print "Cluster_" + str(i)
            self.clusters_space[i].show_faces()

    def show_working_cluster_and_distance(self):
        _, idx = self.getWorkingCluster()
        for (a, b) in itertools.combinations(idx, 2):
            dist = self.clusters_space[a].distance(self.clusters_space[b])
            print "Dist(" + str(a) + " , " + str(b) + ") = " + str(dist)
            print "Faces in " + str(a)
            self.clusters_space[a].show_faces()
            print "Faces in " + str(b)
            self.clusters_space[b].show_faces()

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


    def merge_closest(self, threshold = MERGE_DISTANCE_THRESHOLD):
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
            new_cluster, child_cluster, a_child_cluster = self.clusters_space[idx_a].merge(self.clusters_space[idx_b], True)
            self.add_clusters([new_cluster, child_cluster, a_child_cluster])

            if len(self.distances.values()) == 0:
                break
            ## redo the calculation
            self.calculate_distance()
            min_dist = min(self.distances.values())
            keys = _find_key(self.distances, min_dist)

    def getWorkingCluster(self):
        """

        :return: a list of working cluster and its index
        """
        working_cluster = []
        idx = []
        count = 0
        for cluster in self.clusters_space:
            if cluster.working == True:
                working_cluster.append(cluster)
                idx.append(count)
            count += 1

        return working_cluster, idx

    def name(self):
        print "------------------------------------NAMING-------------------------------------"
        a, _ = self.getWorkingCluster()
        for cluster in a:
            if cluster.name == "no_name" and len(cluster.facepics)>SMALLEST_SIZE_NAME_THRESHOLD:
                cluster.show_faces()
                cluster.make_name()


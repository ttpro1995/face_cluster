import numpy as np
from math import sqrt
from collections import defaultdict
from numpy import linalg as LA
MERGE_THRESHOLD = 0.4

class FaceCluster:
    """
    A cluster of face pic

    facepics: a set of FacePic
    facepicsrep: a numpy matrix, each row is FacePic.rep
    frame_id: a set of frame id that facepics belong to
    n_frame_ids: a dict, the freq of each frame id
    working: boolean status of cluster
    """

    def __init__(self, facepics):
        """
        :param facepics: a set of facepic
        """
        self.facepics = facepics
        self.facepicsrep = None
        self.frame_ids = set()
        self.n_frame_ids = defaultdict()
        self.working = True
        for facepic in facepics:
            self.frame_ids.add(facepic.frame_id)
            if (facepic.frame_id not in self.n_frame_ids.keys()):
                self.n_frame_ids[facepic.frame_id] = 1
            else:
                self.n_frame_ids[facepic.frame_id] +=1

            if (type(self.facepicsrep).__module__ != np.__name__):
                self.facepicsrep = np.array([facepic.rep])
            else:
                self.facepicsrep = np.vstack([self.facepicsrep, facepic.rep])

    def merge(self, acluster):
        """
        merge 2 cluster, set 2 parent cluster working status to false
        :param acluster: another cluster
        :return:
        a cluster
        """
        facepics = self.facepics | acluster.facepics
        self.working = False
        acluster.working = False
        new_cluster = FaceCluster(facepics)
        return new_cluster

    def __len__(self):
        """
        :return: Number of vector in cluster
        """
        tmp = self.facepicsrep.shape[0]
        return int(tmp)

    def is_mergeable(self, acluster):
        """
        Check if this cluster is mergeable with acluster
        :param acluster: a cluster
        :return:
         True or False
        """
        collision = self.frame_ids & acluster.frame_ids
        ncollision = len(collision)
        a = float(ncollision) / len(self.facepics)
        b = float(ncollision) / len(acluster.facepics)
        c = max(a,b)
        if c < MERGE_THRESHOLD:
            return True
        else:
            return False

    def mean(self):
        """
        Calculate mean vector of cluster
        :return:
        mean vector of all facepics
        """
        return np.mean(self.facepicsrep, axis=0)

    def distance(self, acluster):
        """
        Calculate distance of 2 cluster
        :param acluster: target cluster
        :return:
            distance of 2 cluster (float)
        """
        coef = sqrt(float(2*len(self)*len(acluster))/(len(self)+len(acluster)))
        tmp = LA.norm(self.mean()-acluster.mean())
        d =  coef * tmp
        return d




































































































































































302
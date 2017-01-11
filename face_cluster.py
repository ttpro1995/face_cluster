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
        self.freq_frame_ids = defaultdict() # freq each frame id appear
        self.working = True
        for facepic in facepics:
            self.frame_ids.add(facepic.frame_id)
            if (facepic.frame_id not in self.freq_frame_ids.keys()):
                self.freq_frame_ids[facepic.frame_id] = 1
            else:
                self.freq_frame_ids[facepic.frame_id] +=1

            if (type(self.facepicsrep).__module__ != np.__name__):
                self.facepicsrep = np.array([facepic.rep])
            else:
                self.facepicsrep = np.vstack([self.facepicsrep, facepic.rep])

    def merge(self, acluster, split_child):
        """
        merge 2 cluster, set 2 parent cluster working status to false
        :param acluster: another cluster
        :param split_child: True or False (True will split 2 intersection cluster before merge
        :return:
        a cluster
        an intersection from self cluster
        an intersection from acluster
        """

        if (split_child):
            intersec_frame = self.frame_ids & acluster.frame_ids
            child_cluster = self.split(intersec_frame)
            a_child_cluster = acluster.split(intersec_frame)

        facepics = self.facepics | acluster.facepics
        self.working = False
        acluster.working = False
        new_cluster = FaceCluster(facepics)

        if split_child:
            return new_cluster, child_cluster, a_child_cluster
        else:
            return new_cluster

    def split(self, frame_ids):
        """
        split into child cluster contain all face belong to frame_id
        the face which is split into smaller cluster will be remove from self cluster
        :param frame_id: the list of frame_id
        :return: child cluster contain frame id in frame_ids
        """
        splitface = []

        for idx, face in enumerate(self.facepics):
            if (face.frame_id in frame_ids):
                splitface.append(face)

        for face in splitface:
            self.facepics.remove(face)

        child_cluster = FaceCluster(splitface)
        self.__init__(self.facepics) # reinit the facepics
        return child_cluster





    def __len__(self):
        """
        :return: Number of vector in cluster
        """

        return int(len(self.facepics))

    def is_mergeable(self, acluster):
        """
        Check if this cluster is mergeable with acluster
        :param acluster: a cluster
        :return:
         True or False
        """
        #collisions_frames = [k for k in self.freq_frame_ids.keys() if k in acluster.freq_frame_ids.keys()]
        collisions_frames = self.frame_ids & acluster.frame_ids
        if len(self.facepics)==0:
            self.working = False # empty cluster not working
        if len(acluster.facepics) == 0:
            acluster.working = False

        if (self.working == False) or (acluster.working == False):
            return False # not mergable

        self.ncollision = 0
        ancollision = 0
        if len(acluster.facepics) == 0:
            print 'break'
        for collision_id in collisions_frames:
            self.ncollision += self.freq_frame_ids[collision_id]
            ancollision +=acluster.freq_frame_ids[collision_id]
        a = float(self.ncollision) / len(self.facepics)
        b = float(ancollision) / len(acluster.facepics)
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
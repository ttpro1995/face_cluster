import pytest
import sys,os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from face_cluster import FaceCluster
from face_pic import FacePic
from cluster_space import Clusterspace
import numpy as np


class TestClusterSpace():

    def testFacePic(self):
        meow1 = FacePic(np.array([1, 1]), 'f1', None)
        meow2 = FacePic(np.array([3, 3]), 'f1', None)
        meow3 = FacePic(np.array([4, 4]), 'f2', None)
        meow4 = FacePic(np.array([8, 8]), 'f2', None)
        meow5 = FacePic(np.array([12, 12]), 'f2', None)
        c1 = FaceCluster(set([meow1, meow2]))
        c4 = FaceCluster(set([meow1]))
        print (len(c1))
        c2 = FaceCluster(set([meow3, meow4, meow5]))
        print (c1.mean())
        print (c2.mean())
        print (c4.mean())
        d12 = c1.distance(c2)
        d21 = c2.distance(c1)
        d42 = c2.distance(c4)
        print (d12, d21, d42)
        c3 = c1.merge(c2)

    def testClusterSpace(self):
        meow1 = FacePic(np.array([1, 1]), 'f1', None)
        meow2 = FacePic(np.array([3, 3]), 'f1', None)
        meow3 = FacePic(np.array([4, 4]), 'f4', None)
        meow4 = FacePic(np.array([8, 8]), 'f2', None)
        meow5 = FacePic(np.array([12, 12]), 'f2', None)
        c0 = FaceCluster(set([meow1, meow2]))
        c1 = FaceCluster(set([meow3]))
        c2 = FaceCluster(set([meow4, meow5]))
        space = Clusterspace([c0, c1, c2])
        print (space.distances)
        print (space.clusters_space)
        print (c0.distance(c1))
        print (c1.distance(c2))
        print (c0.distance(c2))
        assert c0.distance(c1) == space.distances[0,1]
        assert c1.distance(c2) == space.distances[1,2]
        assert c0.distance(c2) == space.distances[0,2]

    def testClusterSpace2(self):
        meow1 = FacePic(np.array([1, 1]), 'f1', None)
        meow2 = FacePic(np.array([3, 3]), 'f2', None)
        meow3 = FacePic(np.array([4, 4]), 'f3', None)
        meow4 = FacePic(np.array([8, 8]), 'f4', None)
        meow5 = FacePic(np.array([12, 12]), 'f5', None)
        meow6 = FacePic(np.array([10, 10]), 'f6', None)
        meow7 = FacePic(np.array([15, 15]), 'f7', None)
        meow8 = FacePic(np.array([22, 23]), 'f8', None)
        c0 = FaceCluster(set([meow1, meow2]))
        c1 = FaceCluster(set([meow3]))
        c2 = FaceCluster(set([meow4, meow5]))
        c3 = FaceCluster(set([meow6,meow7,meow8]))
        space = Clusterspace([c0, c1, c2])
        print (space.distances)
        print (space.clusters_space)
        print (c0.distance(c1))
        print (c1.distance(c2))
        print (c0.distance(c2))
        assert c0.distance(c1) == space.distances[0,1]
        assert c1.distance(c2) == space.distances[1,2]
        assert c0.distance(c2) == space.distances[0,2]
        space.add_clusters([c3])
        assert c3.distance(c0) == space.distances[0, 3]
        assert c3.distance(c1) == space.distances[1, 3]
        assert c3.distance(c2) == space.distances[2, 3]

    def testMergeAble(self):
        meow1 = FacePic(np.array([1, 1]), 'f1', None)
        meow2 = FacePic(np.array([3, 3]), 'f1', None)
        meow3 = FacePic(np.array([4, 4]), 'f1', None)
        meow4 = FacePic(np.array([8, 8]), 'f1', None)
        meow9 = FacePic(np.array([21, 25]), 'f1', None)
        meow10 = FacePic(np.array([31, 25]), 'f1', None)
        meow11 = FacePic(np.array([41, 25]), 'f1', None)
        meow12 = FacePic(np.array([51, 25]), 'f1', None)
        meow5 = FacePic(np.array([12, 12]), 'f2', None)
        meow6 = FacePic(np.array([10, 10]), 'f2', None)
        meow7 = FacePic(np.array([15, 15]), 'f2', None)
        meow8 = FacePic(np.array([22, 23]), 'f2', None)
        meow13 = FacePic(np.array([22, 33]), 'f2', None)
        meowf3_1 = FacePic(np.array([100, 101]), 'f3', None)
        meowf3_2 = FacePic(np.array([101, 101]), 'f3', None)
        meowf3_3 = FacePic(np.array([102, 101]), 'f3', None)
        meowf3_4 = FacePic(np.array([103, 101]), 'f3', None)
        meowf3_5 = FacePic(np.array([104, 101]), 'f3', None)
        meowf3_6 = FacePic(np.array([105, 101]), 'f3', None)

        c0 = FaceCluster(set([meow1, meow2])) # f1 = 2
        c1 = FaceCluster(set([meow3])) # f1 = 3
        c3 = FaceCluster(set([meow6,meow7,meow8])) # f2 = 3
        c4 = FaceCluster(set([meow1,meow2,meow3,meow4]))  # f2 = 4
        c5 = FaceCluster(set([meow5,meow6,meow7,meow8, meow9, meow13])) # f1 =1 ,f2 = 5
        c6 = FaceCluster(set([meow5,meow6,meow7,meow8, meow9, meow10, meow11,meow12])) # f2 = 4, f1 = 4
        c7 = FaceCluster(set([meowf3_1, meowf3_2, meowf3_3, meowf3_4, meowf3_5, meowf3_6, meow1])) # f1 = 1, f3 = 6
        assert c0.is_mergeable(c1) == False
        assert c3.is_mergeable(c1) == True
        assert c4.is_mergeable(c5) == False
        assert c4.is_mergeable(c6) == False
        assert c7.is_mergeable(c1) == False
        assert c7.is_mergeable(c5) == True

    def test_clusterspace(self):
        meow1 = FacePic(np.array([1, 1]), 'f1', None)
        meow2 = FacePic(np.array([3, 3]), 'f1', None)
        meow3 = FacePic(np.array([4, 4]), 'f1', None)
        meow4 = FacePic(np.array([8, 8]), 'f1', None)
        meow9 = FacePic(np.array([21, 25]), 'f1', None)
        meow10 = FacePic(np.array([31, 25]), 'f1', None)
        meow11 = FacePic(np.array([41, 25]), 'f1', None)
        meow12 = FacePic(np.array([51, 25]), 'f1', None)
        meow5 = FacePic(np.array([12, 12]), 'f2', None)
        meow6 = FacePic(np.array([10, 10]), 'f2', None)
        meow7 = FacePic(np.array([15, 15]), 'f2', None)
        meow8 = FacePic(np.array([22, 23]), 'f2', None)
        meow13 = FacePic(np.array([22, 33]), 'f2', None)
        meowf3_1 = FacePic(np.array([100, 101]), 'f3', None)
        meowf3_2 = FacePic(np.array([101, 101]), 'f3', None)
        meowf3_3 = FacePic(np.array([102, 101]), 'f3', None)
        meowf3_4 = FacePic(np.array([103, 101]), 'f3', None)
        meowf3_5 = FacePic(np.array([104, 101]), 'f3', None)
        meowf3_6 = FacePic(np.array([105, 101]), 'f3', None)

        c0 = FaceCluster(set([meow1, meow2])) # f1 = 2
        c1 = FaceCluster(set([meow3])) # f1 = 3
        c2 = FaceCluster(set([meow4]))
        c3 = FaceCluster(set([meow6,meow7,meow8])) # f2 = 3
        c4 = FaceCluster(set([meow1,meow2,meow3,meow4]))  # f2 = 4
        c5 = FaceCluster(set([meow5,meow6,meow7,meow8, meow9, meow13])) # f1 =1 ,f2 = 5
        c6 = FaceCluster(set([meow5,meow6,meow7,meow8, meow9, meow10, meow11,meow12])) # f2 = 4, f1 = 4
        c7 = FaceCluster(set([meowf3_1, meowf3_2, meowf3_3, meowf3_4, meowf3_5, meowf3_6, meow1])) # f1 = 1, f3 = 6
        clusters = [c0,c1,c2,c3,c4,c5,c6,c7]
        clusterspace = Clusterspace(clusters)
        clusterspace.calculate_distance()

        for idx_a, c_a in enumerate(clusters):
            for idx_b, c_b in enumerate(clusters):
                if (idx_a < idx_b):
                    if (c_a.is_mergeable(c_b)):
                        distance = c_a.distance(c_b)
                        assert distance == clusterspace.distances[idx_a, idx_b]
                else:
                    assert not (idx_a, idx_b) in clusterspace.distances.keys()


    def test_merge(self):
        point1 = []

        root1 = (1,1)
        root = root1
        for i in range(root[0],3+root[0]):
            for j in range(root[1], 3+root[1]):
                point1.append((i,j))

        point2 = []
        root2 = (6,1)
        root = root2

        for i in range(root[0], 3 + root[0]):
            for j in range(root[1], 3 + root[1]):
                point2.append((i, j))

        point3 = []
        root3 = (4, 5)
        root = root2

        for i in range(root[0], 3 + root[0]):
            for j in range(root[1], 3 + root[1]):
                point3.append((i, j))

        point = []
        point = point1 + point2 + point3

        faces = []
        clusters = []
        for idx, p in enumerate(point):
            f = FacePic(np.array(list(p)),str(idx),"Meow")
            c = FaceCluster(set([f]))
            faces.append(f)
            clusters.append(c)

        clusterspace = Clusterspace(clusters)
        clusterspace.merge_closest(4)
        working_cluster = clusterspace.getWorkingCluster()
        assert len(working_cluster) == 3

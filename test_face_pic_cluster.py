from face_cluster import FaceCluster
from face_pic import FacePic
from cluster_space import Clusterspace
import numpy as np

def t1():
    meow1 = FacePic(np.array([1,1]),'f1',None)
    meow2 = FacePic(np.array([3,3]), 'f1', None)
    meow3 = FacePic(np.array([4,4]), 'f2', None)
    meow4 = FacePic(np.array([8,8]), 'f2', None)
    meow5 = FacePic(np.array([12, 12]), 'f2', None)
    c1 = FaceCluster(set([meow1, meow2]))
    c4 = FaceCluster(set([meow1]))
    print (len(c1))
    c2 = FaceCluster(set([meow3,meow4, meow5]))
    print (c1.mean())
    print (c2.mean())
    print (c4.mean())
    d12 = c1.distance(c2)
    d21 = c2.distance(c1)
    d42 = c2.distance(c4)
    print (d12, d21, d42)
    c3 = c1.merge(c2)
    print ('breakpoint')

def t2():
    meow1 = FacePic(np.array([1,1]),'f1',None)
    meow2 = FacePic(np.array([3,3]), 'f1', None)
    meow3 = FacePic(np.array([4,4]), 'f2', None)
    meow4 = FacePic(np.array([8,8]), 'f2', None)
    meow5 = FacePic(np.array([12, 12]), 'f2', None)
    c0 = FaceCluster(set([meow1, meow2]))
    c1 = FaceCluster(set([meow3]))
    c2 = FaceCluster(set([meow4, meow5]))
    space = Clusterspace([c0,c1,c2])
    space.calculate_distance()
    print (space.distances)
    print (space.clusters_space)
    print (c0.distance(c1))
    print (c1.distance(c2))
    print (c0.distance(c2))



def test_merge():
    point1 = []

    root1 = (1, 1)
    root = root1
    for i in range(root[0], 3 + root[0]):
        for j in range(root[1], 3 + root[1]):
            point1.append((i, j))

    point2 = []
    root2 = (6, 1)
    root = root2

    for i in range(root[0], 3 + root[0]):
        for j in range(root[1], 3 + root[1]):
            point2.append((i, j))

    point3 = []
    root3 = (4, 5)
    root = root3

    for i in range(root[0], 3 + root[0]):
        for j in range(root[1], 3 + root[1]):
            point3.append((i, j))

    point = []
    point = point1 + point2 + point3

    faces = []
    clusters = []
    for idx, p in enumerate(point):
        f = FacePic(np.array(list(p)), str(idx), "Meow")
        c = FaceCluster(set([f]))
        faces.append(f)
        clusters.append(c)

    clusterspace = Clusterspace(clusters)
    clusterspace.merge_closest(4)
    working_cluster = clusterspace.getWorkingCluster()
    assert len(working_cluster) == 3
    print len(working_cluster)

test_merge()
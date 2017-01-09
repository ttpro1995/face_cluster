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


t2()
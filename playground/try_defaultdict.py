from collections import *
import cluster_space

def t():
    a = defaultdict()
    for i in range(0,100):
        a[i, i+1] = i+5

    a_val = a.values()
    a_item = a.items()

    # find for 92
    idx = a_val.index(92)
    keys = a_item[idx][0]
    print (keys)
    print (a[keys])

    keys =  cluster_space._find_key(a,85)
    print (a[keys])

    print ('breakpoint')

t()
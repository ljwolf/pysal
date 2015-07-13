import pysal as ps
import pr_build as prb
reload(prb)
import pr_classes as prc
reload(prc)
import compactness as com
reload(com)
import os
import numpy as np
import pandas as pd

def vartarget(atoms, varfield):
    v = [atom.record[varfield] for atom in atoms]
    #target = 702905 according to the redistricting committee but...
    target = 527526 #total population over 18 divided by 53
    #target = 350 #columbus target
    result = np.absolute(sum(v) - target)
    return result

def main(pid):
    
    A = prc.Area('/home/ljw/school/dists/CTs/ca113_MIa.shp')
    #A = prc.Area(ps.examples.get_path('columbus.shp'))

    q = prc.Objective()
    q.add_objective(vartarget, 'total')
    
    print 'building regions for', pid
    encs = prb.initialize(A, q, seeds = np.random.randint(0, len(A.Atoms) - 1,53), verbose = True)
    changes = prb.tabu_search(A, verbose = True)
    
    d = pd.DataFrame()
    
    print 'building polys for', pid
    for region in A.Regions:
        region.build_polygon()
        if not region.polygon.is_valid:
            region.polygon = region.polygon.buffer(0) #fixing digitizing

    pps = []
    reos = []
    schs = []
    conhulls = []
    
    print 'computing compactness for', pid
    for region in A.Regions:
        pps.append(com.polsby_popper(region))
        reos.append(com.reock(region))
        schs.append(com.schwartzberg(region))
        conhulls.append(com.convex_hull(region))
    
    d['pp'] = pd.Series(pps)
    d['reo'] = pd.Series(reos)
    d['sch'] = pd.Series(schs)
    d['conhull'] = pd.Series(conhulls)

    print 'writing', pid
    with open('./reock_test/' + str(pid) + '.csv', 'w') as f:
        d.to_csv(f, header=False)
    
    return True

if __name__ == '__main__':
    
    #main(1)
    import multiprocessing as mp

    pool = mp.Pool(4)

    pool.map(main, range(1000))


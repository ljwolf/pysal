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
    #target = 527526 #total population over 18 divided by 53
    target = 350 #columbus target
    result = np.absolute(sum(v) - target)
    return result

def main(pid):
    
    #A = prc.Area('/home/ljw/school/dists/CTs/ca113_MIa.shp')
    A = prc.Area(ps.examples.get_path('columbus.shp'))

    q = prc.Objective()
    q.add_objective(vartarget, 'HOVAL')
    
    encs = prb.initialize(A, q, seeds = np.random.randint(0, len(A.Atoms) - 1,5), verbose = True)
    changes = prb.tabu_search(A, verbose = True)
    
    #jarpath = './miniball.jar'

    d = pd.DataFrame()
    
    print 'building polys'
    for region in A.Regions:
        region.build_polygon()
        if not region.polygon.is_valid:
            region.polygon = region.polygon.buffer(0) #fixing digitizing

 
    print 'polsby-popper'
    d['pp'] = pd.Series([com.polsby_popper(region) for region in A.Regions])
    print 'reock'
    d['reo']= pd.Series([com.reock(region) for region in A.Regions])
    print 'schwartzberg'
    d['sch']= pd.Series([com.schwartzberg(region) for region in A.Regions])
    print 'convex hull'
    d['con']= pd.Series([com.convex_hull(region) for region in A.Regions])
    print 'writing'
    with open('./reock_test/' + str(pid) + '.csv', 'w') as f:
        d.to_csv(f, header=False)
    
    return True

if __name__ == '__main__':
    import multiprocessing as mp

    A = prc.Area(ps.examples.get_path('columbus.shp'))

    q = prc.Objective()
    q.add_objective(vartarget, 'HOVAL')
    encs = prb.initialize(A, q, seeds = np.random.randint(0, len(A.Atoms) - 1, 5), verbose = True)
    changes = prb.tabu_search(A, verbose = True)
        
    pool = mp.Pool(processes=mp.cpu_count() -1)

    pool.map(main, range(100))


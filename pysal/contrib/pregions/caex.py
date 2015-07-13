import pysal as ps
import pr_build as prb
reload(prb)
import pr_classes as prc
reload(prc)
import numpy as np

A = prc.Area('/home/ljw/school/districting/CTs/ca113_MIa.shp')



def vartarget(atoms, varfield):
    v = [atom.record[varfield] for atom in atoms]
    #target = 702905 according to the redistricting committee but...
    target = 527526 #total population over 18 divided by 53
    result = np.absolute(sum(v) - target)
    return result

q = prc.Objective()
q.add_objective(vartarget, 'total')

encs = prb.initialize(A, q, seeds = np.random.randint(0, len(A.Atoms) - 1, 53), verbose = True)
changes = prb.tabu_search(A, verbose = True) 

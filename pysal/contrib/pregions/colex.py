import pysal as ps
import pr_build as prb
reload(prb)
import pr_classes as prc
reload(prc)
import numpy as np

A = prc.Area(ps.examples.get_path('columbus.shp'))



def vartarget(atoms, varfield):
    v = [atom.record[varfield] for atom in atoms]
    target = 350
    result = np.absolute(sum(v) - target)
    return result

q = prc.Objective()
q.add_objective(vartarget, 'HOVAL')

encs = prb.initialize(A, q, seeds = np.random.randint(0, len(A.Atoms) - 1, 5), verbose = True)
changes = prb.tabu_search(A, verbose = True) 

import pysal as ps
import shapely.geometry as geom
import shapely.ops as geoproc
import shapely as sp
import numpy as np
import pr_classes as prc
import scipy.stats as st
import copy

ATOL = .0001

def initialize(Area, obj, seeds = [], MAX_ITERS = 5000, verbose = False, rseed = 0):
    if rseed:
        np.random.seed(rseed)
    candidates = [i for i in Area.Atoms.keys() if i not in seeds]
    candidates = [i for i in candidates if i not in Area.W.islands]
    if len(seeds) == 0:
        seeds = np.random.choice(candidates, len(candidates)/2)
    for seed in seeds:
        Area.add_region({seed:Area.Atoms[seed]}, obj = copy.deepcopy(obj), construct_poly = False)
    solving = True
    its = 0
    lcan_prior = 0
    while solving and its <  MAX_ITERS:
        r = 0
        if verbose == 1 and its%100 == 0:
            print 'Iteration:', its, 'Candidates Remaining:', len(candidates)
        for region in Area.Regions:
            
            ns = region.neighbors()
             
            print 'Neighbors:', ns
            ns = [i for i in ns if i in candidates]
            print 'Neighbors available:', ns
            if len(ns) > 0:

                in_id = np.random.choice(ns)
                region.update_atoms(atoms_in = {in_id:Area.Atoms[in_id]}, update_poly = False)
                candidates.remove(in_id)
                if verbose == 2: #I know I know, but is this better than "very verbose?"
                    print 'Added ', in_id, 'to', r
            else:
                if verbose == 2:
                    print 'Added nothing to', r
                pass
            r += 1
        its += 1
        if len(candidates) == 0 or lcan_prior - len(candidates) == 0 :
            solving = False
        lcan_prior = len(candidates)
    encs = {key:Area.Atoms[key] for key in candidates}
    Area.Enclaves = encs
    return encs

def swap(A, R1idx, R2idx, swaplist = [], it = 0, objs = None, decay = 0, dry_run = False):
    """
    decay is just the multiplicative factor governing when inferior solutions should be accepted.
    decay * np.random.random() < 1 is the test to accept an inferior solution for one side only
    the 
    
    if you want to accept an inferior solution 40% of the time, set decay to .4
    if you want to never accept inferior solutions, set decay to 0

    dry_run is like the rsync dry_run. Do all the calculations, but don't swap anything. 
    """
    R1 = A.Regions[R1idx]
    R2 = A.Regions[R2idx]

    if objs == None:
        objs = R1.obj.functions
    if np.any([obj['geo'] for obj in objs.values()]):
        poly = True
    else:
        poly = False
    
    #this is costly if shapes are involved :(
    
    R1_in  = {key: R2.Atoms[key] for key in swaplist if key in R2.Atoms.keys()}
    R1_out = {key: R1.Atoms[key] for key in swaplist if key in R1.Atoms.keys()}
    
    R1_in = dict({key:R1.Atoms[key] for key in R1.Atoms.keys() if key not in R1_out}, **R1_in)

    R1_new = prc.Region(A, atoms_in = R1_in, construct_poly = poly)

    
    R2_in = {key: R1.Atoms[key] for key in swaplist if key in R1.Atoms.keys()}
    R2_out = {key:R2.Atoms[key] for key in swaplist if key in R2.Atoms.keys()}

    R2_in = dict({key:R2.Atoms[key] for key in R2.Atoms.keys() if key not in R2_out}, **R2_in)
    
    R2_new = prc.Region(A, atoms_in = R2_in, construct_poly = poly)
    
    R1_new.obj.check_constraints()
    R1_new.obj.check_objective(R1_new.Atoms.values())
    if not hasattr(R1.obj, 'soln'):
        R1.obj.check_constraints()
        R1.obj.check_objective(R1.Atoms.values())

    R2_new.obj.check_constraints()
    R2_new.obj.check_objective(R2_new.Atoms.values())
    if not hasattr(R2.obj, 'soln'):
        R2.obj.check_constraints()
        R2.obj.check_objective(R2.Atoms.values())
    
    #print hasattr(R1.obj, 'soln'), hasattr(R2.obj, 'soln'), hasattr(R1_new.obj, 'soln'), hasattr(R2_new.obj, 'soln')

    #now compare
    delta1 = R1_new.obj.soln - R1.obj.soln
    delta2 = R2_new.obj.soln - R2.obj.soln
    feasible = R1_new.obj.feasible and R2_new.obj.feasible

    if feasible and not dry_run:
        if delta1 + delta2 >= 0 and np.random.random() > decay: #to imporove a min obj, only take negative delts
            pass
        else: #only take improvements or shifts if decay allows.
            A.Regions[R1idx] = R1_new
            A.Regions[R2idx] = R2_new
    
    return delta1 + delta2

def tabu_swap(A, R1idx, R2idx, rate = 1, take = 1, it = 0, objs = None, decay = 0, dry_run = False):
    #print R1idx, R2idx, A.Regions[R1idx]
    R1 = A.Regions[R1idx]
    R2 = A.Regions[R2idx]

    tabulist = [(atom.shpid, atom.last_look) for atom in R1.Atoms.values()]
    tabulist.extend([(atom.shpid, atom.last_look) for atom in R2.Atoms.values()])\
    
    tabmean = np.mean([tup[1] for tup in tabulist])
    probs = [st.geom.pmf(tabu[1], 1/(tabmean + ATOL)) for tabu in tabulist] #weight the times geometrically
    probs = [x/sum(probs) for x in probs]
    probs = probs.append(1 - sum(probs))
    
    adjtabu = [tabu[0] for tabu in tabulist] + [None]
    swaplist = np.random.choice(adjtabu, size=take, p=probs)

    change = swap(A, R1idx, R2idx, swaplist = swaplist, it = it, objs = objs, decay = decay, dry_run = dry_run)
    
    for idx in swaplist:
        if idx in R1.Atoms.keys():
            R1.Atoms[idx].last_look = it
        elif idx in R2.Atoms.keys():
            R2.Atoms[idx].last_look = it

    return change

def tabu_search(A, rate = 1, take = 1, decay = 0, MAX_ITERS = np.nan, dry_run = False, verbose = False):    
     
    tabu_it = 0
    pair_it = 0
    duds = 0
    changes = []
    tabuing = True
    
    for region in A.Regions:
        region.obj.check_objective(region.Atoms.values())
        
    while tabuing:
        outer_soln = sum([reg.obj.soln for reg in A.Regions])
        
        pair = np.random.choice(len(A.Regions), 2, replace = False)
        pairwise_improving = True
        clist = [(outer_soln, 0, tabu_it)]
        
        while pairwise_improving:
            curr_soln = sum([reg.obj.soln for reg in A.Regions])
            change = tabu_swap(A, pair[0], pair[1], rate = rate, take = take, \
                    decay = decay, dry_run = dry_run)
            if change < 0:
                clist.append((curr_soln, change, pair_it))
                if verbose == 2:
                    print 'improving...'
                    duds -= 1
            elif pair_it - clist[-1][-1] > len(A.Regions[pair[0]].Atoms)/2: #leave when you haven't had a swap for a while
                pairwise_improving = False
                changes.append(clist)
                if verbose == 2:
                    print 'not improving, switching pairs...'
            pair_it += 1
        
        if len(changes[-1]) == 1:
            duds += 1
        if duds > len(A.Regions):
            tabuing = False
        if tabu_it > MAX_ITERS:
            break
        tabu_it += 1
        if verbose:
            print 'One pairwise cycle done'
    
    return changes
        

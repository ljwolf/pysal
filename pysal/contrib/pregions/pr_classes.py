import pysal as ps
import shapely.geometry as geom
import shapely.ops as geoproc
import shapely as sp
import numpy as np
import copy as cp


class Area:
    """
    The top level map upon which Regions are constructed from Atoms
    """
    def __init__(self, fname, contiguity='queen'):
        if fname.split('.')[-1] == 'shp':
            fname = '.'.join(fname.split('.')[:-1]) #foo.bar.shp but not foo.tar.gz
        
        ##Data IO
        self.dataObj = fname
        self.df = ps.open(fname+'.dbf')
        self.atts = self.df.header
        self.chains = ps.open(fname + '.shp').read() #how do I do this without two reads?
        
        self.Enclaves = {} 
        self.Atoms = {}
        self.Regions = []

        for idx, chain in enumerate(self.chains):
            self.Atoms[idx] = Atom(idx, chain, self.df)
        
        self.lattice = geom.MultiPolygon([x.polygon for x in self.Atoms.values()])
        

        ##Weights
        self.moves = contiguity
        if self.moves == 'queen' or 'Queen' or 'QUEEN':
            self.W = ps.queen_from_shapefile(fname + '.shp')
            
        elif self.moves == 'rook' or 'Rook' or 'ROOK':
            self.W = ps.rook_from_shapefile(fname + '.shp')
        else:
            raise UserWarning('no contiguity selected, defaulting to rook')
            self.W = ps.rook_from_shapefile(fname + '.shp')
    
    def add_region(self, atoms, obj = None, fns = {}, weights = {}, constrs = {}, construct_poly = True):
        self.Regions.append(Region(self, atoms, construct_poly))
        try:
            self.Regions[-1].obj = cp.copy(self.Regions[0].obj)
        except AttributeError:
            if obj:
                self.Regions[-1].obj = cp.deepcopy(obj)
            elif fns:
                self.Regions[-1].obj = Objective(fns = fns, weights = weights, constrs = constrs)
            else:
                pass
        self.Regions[-1].id = len(self.Regions)
            
    def set_obj_fn(self, obj = None, fns = {}, weights = {}, constrs = {}):
        if obj:
            for region in self.Regions:
                region.obj_fn = cp.deepcopy(obj)
                obj.in_region = region
        else:
            obj = Objective(fns, weights, constrs)
            for region in self.Regions:
                region.obj_fn = obj
 
class Region:
    """
    Aggregations of Atoms in an Area
    """
    
    def __init__(self, A, atoms_in, construct_poly = True):
        self.id = None
        self.Atoms = {key:A.Atoms[key] for key in atoms_in}
        self.in_Area = A
        if len(A.Regions) > 0:
            self.obj = cp.deepcopy(A.Regions[0].obj)
            self.obj.check_objective()
            self.obj.check_constraints()
        else:
            self.obj = Objective()
        self.W = A.W
        
        if construct_poly:
            self.build_polygon()
            self._geom = self.polygon
        else:
            self.polygon = self._geom = None

    def neighbors(self, W = None):
        if W == None:
            W = self.W
        neighbs = [W[atom].keys() for atom in self.Atoms]
        neighbs = [atom for nlist in neighbs for atom in nlist]
        return list(set(neighbs))
    
    def update_atoms(self, atoms_in = {}, atoms_out = {}, update_poly = True):
        for atom in atoms_out:
            self.Atoms[atom].in_Region = None
            self.Atoms.pop(atom)
        for atom in atoms_in:
            self.Atoms[atom] = atoms_in[atom]
            self.Atoms[atom].in_Region = self.in_Region = self.id
        if update_poly:
            self.update_polygon(atoms_in, atoms_out, update_atoms = False)
    
    def build_polygon(self):
        self.polygon = geoproc.cascaded_union([atom.polygon for atom in self.Atoms.values()])

    def update_polygon(self, atoms_in = {}, atoms_out = {}, update_atoms = False):
        if atoms_out:
            for atom in atoms_out.values():
                self.polygon = self.polygon.difference(atom.polygon)
        if atoms_in:
            geoproc.cascaded_union([self.polygon].extend(atoms_in.values()))
        if update_atoms:
            self.update_atoms(atoms_in, atoms_out, update_polygon = False)

class Atom:
    """
    Smallest elements in an Area that compose Regions. 
    """
    def __init__(self, idx, chain, df):
        self.dfid = idx
        self.record = {name:value for name,value in zip(df.header, df.read_record(idx))}
        self.shpid = chain.id
        self.in_Region = None
        self.chain = chain
        self.polygon = self._geom = geom.shape(self.chain)
        self.last_look = 0
    
    def neighbors(self, W):
        self.neighbors = W[self.shpid]
        return self.neighbors

class Objective:
    """
    A sufficiently arbitrary objective function, to allow for users to specify
    their own objective (or objectives)

    To do this, we need to pass it something a little bit more complex, and the
    design of this class will resemble the "model" object in Gurobi's python
    interface.

    All "fns" and "constrs" should apply to an arbitrary dict Atoms. Objectives need to be some additive function of components, like

    z = obj(x1, x2, x3, x4) = x1^2 + k * log(x2 + x3) e^x4

    All objectives should be to minimize the fns and keep the constrs above 0, like

    Min obj(x) = z
    s.t.  f(x) >= 0

    You can transform constraints to get here, too consider the following transforms:
    
    1. Objective: Max -> Min:
    
    Minimizing the negative is exactly the same as maximizing the positive.
    
        Max obj(x) = z --> Min -obj(x) = -z

    2. Constraints: lt to gt:

    keeping an f(x) below zero is the same as keeping -f(x) above zero. Also, having f(x) > k is the same as having f(x) - k > 0.

        f(x) =< k --> -f(x) + k >= 0

    3. Constraints: equality

    to get an equality constraint, do both f(x) - k >= 0 and f(x) + k >= 0. The only time f(x) and -f(x) satisfy this constraint is when f(x) = k. This will turn 1 equality constraint into two inequality constraints.
        f(x) = k --> f(x) - k >= 0 & - f(x) + k >= 0

    """
    def __init__(self, fns = {}, constrs = {}, region = None):
        
        self.functions = fns
        self.terms = {key:None for key in self.functions.keys()}
        self.constrs = constrs
        self.feasible = None
        
    def add_objective(self, fn, args, weight = 1, name = None, spatial = False):
        if name == None:
            if fn.__name__ != '<lambda>':
                name = fn.__name__    
            else:
                name = 'obj' + str(len(self.constrs) + 1)
        self.functions.update({name:dict()})
        self.functions[name]['fn'] = fn
        self.functions[name]['args'] = args
        self.functions[name]['geo'] = spatial
        self.functions[name]['weight'] = weight

    def check_objective(self, atoms = []):
        for key,term in self.functions.iteritems():
            obj = term['fn']
            args = term['args']
            term['soln'] = obj(atoms, args)
        self.soln = sum([fn['soln'] * fn['weight'] for fn in self.functions.values()])
        return self.soln
    
    def relax_objective(self, name):
        del self.functions[name]
    
    def add_constraint(self, fn, args, name = None, spatial = False):
        if name == None:
            name = 'c' + str(len(self.constrs) + 1)
        self.constrs[name]['fn'] = fn
        self.constrs[name]['args'] = args
        self.constrs[name]['geo'] = geo

    def relax_constraints(self, name):
        del self.constrs[name]

    def check_constraints(self, atoms = []):
        sats = True
        for key,constraint in self.constrs:
            constr = indict['fn']
            args = indict['args']
            constraint['slack'] = constr(atoms, args)
            sats = sats and constraint['slack'] >= 0
        self.feasible = sats
        return self.feasible
    
    #def update_objective(self):
    #    obj = {} 
    #    for key,fn in self.fns.iteritems():
    #        obj[key] = self.weights[key] * fn(self.args[key])
    #    return obj

from . import shapes as s
import copy
import numpy as np
from warnings import warn

def boundary(shape):
    """
    This should correctly compute an OGC-spec boundary for the given
    shape. In set terms, this is the closure of the point set defining
    the shape

    Polygon -> [Ring for ring in vertices U holes]
    LineSegment -> Point(head), Point(tail)
    Chain -> [head/tail of each part, if point has odd order]
    Ray -> Point(origin)
    Rectangle -> Ring(Rectangle.vertices) 
    [Geometry, Line, Point, Ring, VerticalLine] -> None
    """
    if isinstance(shape, s.Polygon): 
        if shape.holes == [[]]: #is fails here
            holerings = []
        else:
            holerings = list(map(s.Ring, shape.holes))
        return tuple(list(map(s.Ring, shape.parts)) + holerings)
    elif isinstance(shape, s.LineSegment):
        return s.Point(shape.p1), s.Point(shape.p2)
    elif isinstance(shape, s.Chain):
        if len(shape.parts) == 1:
            if shape.vertices[-1] == shape.vertices[0]:
                warn('The boundary of {} is empty!'.format(shape), UserWarning)
                return None
            else:
                return s.Point(shape.vertices[0]), s.Point(shape.vertices[-1])
        boundary_set = set()
        for part in shape.parts:
            head, tail = part[0], part[-1] #I wish py2 could head, *_, tail
            # head/tail are only in boundary if they have odd order. So,
            # alternating insert/del gives us only odd-ordered head/tail
            if head in boundary_set:
                boundary_set.discard(head)
            else:
                boundary_set.update((head,))
            if tail in boundary_set:
                boundary_set.discard(tail)
            else:
                boundary_set.update((tail,))
        if len(boundary_set) == 1:
            return s.Point(boundary_set)
        return tuple(map(s.Point, boundary_set))
    elif isinstance(shape, s.Ray):
        return s.Point(shape.o)
    elif isinstance(shape, s.Rectangle):
        return s.Ring([shape.left, shape.lower, shape.right, shape.upper])
    else:
        return None

def _chain_exterior_indices(chain):
    """
    This is helper function that returns a list of tuples describing whether or
    not (head,tail) of the corresponding part is in the boundary or not
    """
    boundaries = set([ch._Point__loc for ch in boundary(chain)])
    indices = [(part[0] in boundaries, part[-1] in boundaries)
               for i, part in enumerate(chain.parts)]
    return indices
    
def interior(shape):
    """
    This should return an OGC-spec interior for the given shape

    LineSegment -> LineSegment
    Ray -> Ray
    Polygon -> Polygon
    Chain -> (Chain OR None[, Chain OR None, ...]) None when Chain is Ring
    Rectangle -> Rectangle

    [Point, Ring, Line, VerticalLine] -> None
    """
    if isinstance(shape, s.Polygon):
        out = s.Polygon(shape.parts, shape.holes)
        out._closed = False #the interior of a polygon is an open set
        return out
    elif isinstance(shape, s.Chain):
        is_actually_ring = [part[0] == part[-1] for part in shape.parts]
        out = []
        for part, ring in zip(shape.parts, is_actually_ring):
            part = None if ring else s.Chain(part)
            if part is not None:
                part._closed = False
            out.append(part)
        return out
    elif isinstance(shape, s.LineSegment):
        out = s.LineSegment(shape.p1, shape.p2)
        out._closed = False
        return out
    elif isinstance(shape, s.Ray):
        out = s.Ray(shape.o, shape.p)
        out._origin_inclusive = False
        return out
    else:
        return None

def _point_ints_point(a,b):
    return np.array_equal(a._Point__loc, b._Point__loc)

def _point_ints_chain(a,b):
    searching = True
    segs = b.segments
    while segs:
        this_part = segs.pop()
        collinear = [seg.sw_ccw(a) == 0 for seg in this_part]
        if any(collinear):
            return True
    return False
        
    raise NotImplementedError
def _point_ints_polygon(a,b):
    if not b.is_closed:
        return intersects(a, boundary(b))
    else:
        return b.contains_point(a)

def _chain_ints_point(a,b):
    return _point_ints_chain(b,a)
def _chain_ints_chain(a,b):
    """
    If this is slow, we do a bentley-ottman
    """
    cache = set()
    for a_part in a.segments:
        for b_part in b.segments:
            if (a_part, b_part) in set():
                continue
            elif a_part.intersect(b_part):
                return True
    return False
def _chain_ints_polygon(a,b):
    raise NotImplementedError

def _polygon_ints_point(a,b):
    return _point_ints_polygon(b,a)
def _polygon_ints_chain(a,b):
    return _chain_ints_polygon(b,a)
def _polygon_ints_polygon(a,b):
    raise NotImplementedError

# This is where I think multiple dispatch would be nice. 
# these standalones will get bound on first arg anyway, though
_ints_dispatch ={(s.Point, s.Point):     _point_ints_point, 
                 (s.Point, s.Chain):     _point_ints_chain,
                 (s.Point, s.Polygon):   _point_ints_polygon,
                 (s.Chain, s.Point):     _chain_ints_point, 
                 (s.Chain, s.Chain):     _chain_ints_chain,
                 (s.Chain, s.Polygon):   _chain_ints_polygon,
                 (s.Polygon, s.Point):   _polygon_ints_point, 
                 (s.Polygon, s.Chain):   _polygon_ints_chain,
                 (s.Polygon, s.Polygon): _polygon_ints_polygon}

def intersects(a,b):
    """
    Returns a boolean whether or not a intersects b
    """
    a = coerce(a)
    b = coerce(b)
    return _ints_dispatch[(type(a), type(b))](a,b)

def _point_intn_point(a,b):
    raise NotImplementedError
def _point_intn_chain(a,b):
    raise NotImplementedError
def _point_intn_polygon(a,b):
    raise NotImplementedError

def _chain_intn_point(a,b):
    return _point_intn_chain(b,a)
def _chain_intn_chain(a,b):
    raise NotImplementedError
def _chain_intn_polygon(a,b):
    raise NotImplementedError

def _polygon_intn_point(a,b):
    return _point_intn_polygon(b,a)
def _polygon_intn_chain(a,b):
    return _chain_intn_polygon(b,a)
def _polygon_intn_polygon(a,b):
    raise NotImplementedError

# This is where I think multiple dispatch would be nice. 
# these standalones will get bound on first arg anyway, though
_intn_dispatch ={(s.Point, s.Point):     _point_intn_point, 
                 (s.Point, s.Chain):     _point_intn_chain,
                 (s.Point, s.Polygon):   _point_intn_polygon,
                 (s.Chain, s.Point):     _chain_intn_point, 
                 (s.Chain, s.Chain):     _chain_intn_chain,
                 (s.Chain, s.Polygon):   _chain_intn_polygon,
                 (s.Polygon, s.Point):   _polygon_intn_point, 
                 (s.Polygon, s.Chain):   _polygon_intn_chain,
                 (s.Polygon, s.Polygon): _polygon_intn_polygon}

def intersection(a,b):
    """
    returns the intersection of a and b
    """
    a = coerce(a)
    b = coerce(b)
    
    return _intn_dispatch[(type(a), type(b))](a,b)

def coerce(shape):
    """
    coerce an arbitrary shape to one of the shapes in PySAL that are conformal
    to OGC primitives:

    Ring -> Chain
    Chain -> Chain
    Polygon -> Polygon
    Point -> Point
    LineSegment -> Chain
    Rectangle -> Polygon or Chain, depending on if it's closed
    [Line, VerticalLine, Ray] -> None
    """
    if isinstance(shape, (s.Polygon, s.Chain, s.Point)):
        return shape
    if isinstance(shape, s.Ring):
        out = s.Chain(s.Ring.vertices)
    if isinstance(shape, s.LineSegment):
        out = s.Chain([s.p1, s.p2])
    if isinstance(shape, s.Rectangle):
        if shape.is_closed:
            return s.Polygon([(s.left, s.lower),(s.left, s.upper),
                              (s.right, s.upper),(s.right, s.lower)])
        else:
            return s.Ring([(s.left, s.lower), (s.left, s.upper),
                           (s.right, s.upper),(s.right, s.lower)])
    elif isinstance(shape, tuple):
        try:
            return s.Point(shape)
        except:
            return None
    
    

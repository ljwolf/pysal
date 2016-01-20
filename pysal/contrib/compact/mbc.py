from math import pi as PI
from scipy.spatial import ConvexHull
from pysal.cg import get_angle_between, Ray, is_clockwise
import copy 
import numpy as np
import scipy.spatial.distance as dist

not_clockwise = lambda x: not is_clockwise(x)

def minimum_bounding_circle(points, not_hull=True):
    """
    Implements Skyum (1990)'s algorithm for the minimum bounding circle in R^2. 

    0. Store points clockwise. 
    1. Find p in S that maximizes angle(prec(p), p, succ(p) THEN radius(prec(p),
    p, succ(p)). This is also called the lexicographic maximum, and is the last
    entry of a list of (radius, angle) in lexicographical order. 
    2a. If angle(prec(p), p, succ(p)) <= 90 degrees, then finish. 
    2b. If not, remove p from set. 
    """
    points = [points[i] for i in ConvexHull(points).vertices]
    points.reverse() #shift from ccw to cw
    POINTS = copy.deepcopy(points)
    removed = []
    i=0
    while True:
        angles = [_angle(*_neighb(p, points)) for p in points]
        circles = [_circle(*_neighb(p, points)) for p in points]
        radii = [c[0] for c in circles]
        lexord = np.lexsort((radii, angles)) #confusing as hell defaults...
        lexmax = lexord[-1]
        candidate = (_prec(points[lexmax], points), 
                     points[lexmax],
                     _succ(points[lexmax], points))
        if angles[lexmax] <= PI/2.0:
            return circles[lexmax]#, points, removed, candidate
        else:
            try:
                removed.append((points.pop(lexmax), i))
            except IndexError:
                raise Exception("Construction of Minimum Bounding Circle failed!")
        i+=1

def _angle(p,q,r):
    """
    compute the positive angle formed by PQR
    """
    return np.abs(get_angle_between(Ray(q,p),Ray(q,r)))

def _prec(p,l):
    """
    retrieve the predecessor of p in list l
    """
    pos = l.index(p)
    if pos-1 < 0:
        return l[-1]
    else:
        return l[pos-1]

def _succ(p,l):
    """
    retrieve the successor of p in list l
    """
    pos = l.index(p)
    if pos+1 >= len(l):
        return l[0]
    else:
        return l[pos+1]

def _neighb(p,l):
    """
    sandwich p with the predecessor and successor of p in cycle l
    """
    return _prec(p, l), p, _succ(p,l)

def _circle(p,q,r, dmetric=dist.euclidean):
    """
    Returns (radius, (center_x, center_y)) of the circumscribed circle by the
    triangle pqr.

    note, this does not assume that p!=q!=r
    """
    px,py = p
    qx,qy = q
    rx,ry = r
    if np.allclose(p, q) or np.allclose(q,r):
        #print p,q,r
        raise Exception('conditions for algorithm not met')
    #elif np.allclose(p,r):
        #print p,q,r, "p,r are the same"
    #else:
        #print p,q,r, "all are different"

    if np.allclose(np.abs(_angle(p,q,r)), 0):
        radius = dmetric(p,q)/2.
        center_x = (px + qx)/2.
        center_y = (py + qy)/2.
    elif np.allclose((qy - py) * (rx - qx), (ry - qy) * (qx - px)):
        center_x = center_y = radius = -np.inf
    else:
        D = 2*(px*(qy - ry) + qx*(ry - py) + rx*(py - qy))
        center_x = ((px**2 + py**2)*(qy-ry) + (qx**2 + qy**2)*(ry-py) 
              + (rx**2 + ry**2)*(py-qy)) / float(D)
        center_y = ((px**2 + py**2)*(rx-qx) + (qx**2 + qy**2)*(px-rx) 
              + (rx**2 + ry**2)*(qx-px)) / float(D)
        radius = dmetric((center_x, center_y), p)
    return radius, (center_x, center_y)

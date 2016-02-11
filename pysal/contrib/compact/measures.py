from __future__ import division
from math import pi as PI
from pysal import cg
from .mbc import minimum_bounding_circle
import numpy as np

def ipq(chain):
    """
    The Isoperimetric quotient, defined as the ratio of a chain's area to the 
    area of the equi-perimeter circle. 

    Construction:
    --------------
    let:
    p_d = perimeter of district
    a_d = area of district
    
    a_c = area of the constructed circle
    r = radius of constructed circle

    then the relationship between the constructed radius and the district
    perimeter is:
    p_d = 2 \pi r
    p_d / (2 \pi) = r
    
    meaning the area of the circle can be expressed as:
    a_c = \pi r^2
    a_c = \pi (p_d / (2\pi))^2
    
    implying finally that the IPQ is:

    pp = (a_d) / (a_c) = (a_d) / ((p_d / (2*\pi))^2 * \pi) = (a_d) / (p_d**2 / (4\PI))
    """
    return (4 * PI * chain.area) / (chain.perimeter**2)

def iaq(chain):
    """
    The Isoareal quotient, defined as the ratio of a chain's perimeter to the
    perimeter of the equi-areal circle
    """
    return (2 * PI * np.sqrt(chain.area/PI)) / chain.perimeter

def polsby_popper(chain):
    """
    Alternative name for the Isoperimetric Quotient
    """
    return ipq(chain)

def schwartzberg(chain):
    """
    Alterantive name for the Isoareal Quotient
    """
    return iaq(chain)

def convex_hull(chain):
    """
    ratio of the convex hull area to the area of the shape itself
    """
    pointset = [point for part in chain.parts for point in part] 
    chull = cg.Polygon(cg.convex_hull(pointset))
    return chain.area / chull.area

def reock(chain):
    """
    The Reock compactness measure, defined by the ratio of areas between the
    minimum bounding/containing circle of a shape and the shape itself. 
    """
    radius, (cx, cy) = minimum_bounding_circle(pointset)
    return chain.area / (PI * radius ** 2)


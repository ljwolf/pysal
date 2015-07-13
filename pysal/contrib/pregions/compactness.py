import pysal as ps
import shapely.geometry as geom
import shapely.ops as geoproc
import shapely as sp
import numpy as np
import math
import scipy as sci

#THIS MUST BE IN THIS ORDER TO WORK
import os
os.environ['CLASSPATH'] = '/usr/local/bin/miniball.jar'

#from jnius import autoclass
#Mini_Ball = autoclass('com.dreizak.miniball.highdim.Miniball')
#Array_Point_Set = autoclass('com.dreizak.miniball.model.ArrayPointSet')
#DONT KNOW WHY, JNIUS IS FRUSTRATING

def polsby_popper(Region):
    """
    ratio of district area to the area of the equi-perimeter circle
    see that:
    p_d = 2 \pi r
    p_d / (2 \pi ) = r
    a_c = (p_c / 2 \pi)^2 * \pi
    pp = (a_d)/(a_c) = (a_d) / ((p_d / 2 \pi)^2 * \pi) = (a_d) / (p_d^2 / (4\pi))
    """
    return (Region.polygon.area) / (Region.polygon.boundary.length**2 / (4 * math.pi))

#def reock(Region):
#    """
#    ratio of district area to the area of the smallest enclosing circle
#    
#    because there's not a good implementation of
#    the minimum bounding circle in python that I could access, 
#    I'll just ship the miniball jar and use it using jnius.
#    of course, this requires the JDK and jnius, which is bollocks
#    I'll provide the compiled version I used for interested 
#    parties at: 
#    http://www.public.asu.edu/~lwolf2/data/miniball.tar.gz
#    then just point your jarpath at the 
#    miniball-1.0.4-SNAPSHOT.jar file.
#
#    So, all you'd have to do if you want to run this test is to install the jdk and jnius.
#    """
#
#    #os.environ['CLASSPATH'] = '/usr/local/bin/miniball.jar'
#    
#    chull = Region.polygon.convex_hull
#    pts = [pt for pt in chull.boundary.coords]
#
#    JAVA_pts = Array_Point_Set(2, len(pts))
#    for i, pt in enumerate(pts):
#        for j, co in enumerate(pt):
#            JAVA_pts.set(i, j, co)
#        
#    JAVA_mb = Mini_Ball(JAVA_pts)
#
#    mball = geom.Point(JAVA_mb.center()).buffer(JAVA_mb.radius())
#
#    return Region.polygon.area / mball.area

def schwartzberg(Region):
    """
    ratio of district perimeter to the perimeter of the equi-areal circle
    shouldn't this one be inverted? longer perimeter will indicate less compactness?
    """
    area = Region.polygon.area
    circumference = np.sqrt(area/math.pi)*2*math.pi
    return circumference/ Region.polygon.boundary.length


def convex_hull(Region):
    """
    ratio of district area to the convex hull area
    """
    return (Region.polygon.area)/(Region.polygon.convex_hull.area)

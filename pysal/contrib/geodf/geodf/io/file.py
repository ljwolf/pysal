import os
import pysal as ps
import numpy as np
from shapely.geometry import mapping

from six import iteritems
from geodf import GeoDataFrame


def read_shapefile(filename, **kwargs):
    """
    Returns a GeoDataFrame from a shapefile/dbf pair.

    Slight modification of the upstream GeoPandas repo to 
    use PySAL readers for shapefiles

    *filename* is either the absolute or relative path to the file to be
    opened and *kwargs* are keyword args to be passed to the method when
    opening the file.
    """
    shp = ps.open(os.path.splitext(filename)[0] + '.shp')
    dbf = ps.open(os.path.splitext(filename)[0] + '.dbf')
    feats = []

    for i, poly in enumerate(shp):
        row = dbf[i][0]
        prop = {field:element for field,element in zip(dbf.header, row)}
        geom = poly.__geo_interface__
        feat = {'geometry':geom, 'properties':prop, 'bbox':poly.bbox}
        feats.append(feat)

    gdf = GeoDataFrame.from_features(feats)
    shp.close()
    dbf.close()

    return gdf


def to_file(df, filename, driver="ESRI Shapefile", schema=None,
            **kwargs):
    """
    Write this GeoDataFrame to a shapefile

    Parameters
    ----------
    df : GeoDataFrame to be written
    filename : string
        File path or file handle to write to.
    schema : dict, default None
        If specified, the schema dictionary is passed to Fiona to
        better control how the file is written. If None, GeoPandas
        will determine the schema based on each column's dtype

    The *kwargs* are passed to fiona.open and can be used to write
    to multi-layer data, store data within archives (zip files), etc.
    """
    fpath = os.path.splitext(filename)[0]
 
    if schema != None:
         specs = schemas
    else:
        type2spec = {int: ('N', 20, 0),
                     np.int64: ('N', 20, 0),
                     float: ('N', 36, 15),
                     np.float64: ('N', 36, 15),
                     str: ('C', 14, 0)
                     }
        types = [type(df[i].iloc[0]) for i in df.columns.drop('geometry')]
        specs = [type2spec[t] for t in types]
    db = ps.open(fpath + '.dbf' , 'w')
    db.header = list(df.columns)
    db.field_spec = specs
    for i, row in df.T.iteritems():
        db.write(row)
    db.close()
    
    shp = ps.open(fpath + '.shp', 'w')
    for poly in df['geometry']:
        shp.write(ps.cg.shapes.asShape(poly))
    shp.close()
    return fpath + '.dbf', fpath + '.shp'
    
    
def _common_geom_type(df):
    # Need to check geom_types before we write to file...
    # Some (most?) providers expect a single geometry type:
    # Point, LineString, or Polygon
    geom_types = df.geometry.geom_type.unique()

    from os.path import commonprefix   # To find longest common prefix
    geom_type = commonprefix([g[::-1] for g in geom_types if g])[::-1]  # Reverse
    if not geom_type:
        geom_type = None

    return geom_type

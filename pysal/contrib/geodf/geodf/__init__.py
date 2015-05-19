try:
    from geodf.version import version as __version__
except ImportError:
    __version__ = '0.2.0.dev-unknown'

from geodf.geoseries import GeoSeries
from geodf.geodataframe import GeoDataFrame

from geodf.io.file import read_shapefile
from geodf.io.sql import read_postgis

# make the interactive namespace easier to use
# for `from geopandas import *` demos.
import geodf as gdf
import pandas as pd
import numpy as np

CREATE OR REPLACE 
         FUNCTION COM_IPQ(geom Geometry)
          RETURNS double precision AS 
               $$ SELECT (4 * pi() * ST_Area(geom)) / (ST_Perimeter(geom)^2);$$ 
         LANGUAGE SQL;

CREATE OR REPLACE 
         FUNCTION COM_PolsbyPopper(geom Geometry)
          RETURNS double precision AS
               $$ SELECT COM_IPQ(geom); $$
         LANGUAGE SQL;

CREATE OR REPLACE 
         FUNCTION COM_ConvexHull(geom Geometry)
          RETURNS double precision AS
               $$ SELECT ST_Area(geom) / ST_Area(ST_ConvexHull(geom));$$ 
         LANGUAGE SQL;

CREATE OR REPLACE 
         FUNCTION COM_IAQ(geom Geometry)
          RETURNS double precision AS
               $$ SELECT (2 * pi() * |/(ST_Area(geom)/pi())) / ST_Perimeter(geom);$$ 
         LANGUAGE SQL;

CREATE OR REPLACE 
         FUNCTION COM_Schwartzberg(geom Geometry)
          RETURNS double precision AS
               $$ SELECT COM_IAQ(geom); $$
         LANGUAGE SQL;

CREATE OR REPLACE
         FUNCTION COM_Reock(geom Geometry)
          RETURNS double precision AS
               $$ SELECT ST_Area(geom) / ST_Area(ST_MinimumBoundingCircle(geom)); $$
         LANGUAGE SQL;

CREATE OR REPLACE 
         FUNCTION COM_py_Reock(geom Geometry)
          RETURNS double precision AS
               $$ 
from compact import reock
from shapely import wkb
from binascii import unhexlify
from pysal.cg import asShape
pgon = wkb.loads(unhexlify(geom))
return reock(asShape(pgon))
                $$
          LANGUAGE plpythonu;

CREATE OR REPLACE
         FUNCTION COM_OS_3(geom Geometry)
          RETURNS double precision AS
               $$ SELECT |/(ST_Area(geom)/pi()) 
                         / (|/(ST_Area(ST_MinimumBoundingCircle(geom))/pi())); $$
         LANGUAGE SQL;

CREATE OR REPLACE
         FUNCTION COM_FlahertyCrumplin(geom Geometry)
          RETURNS double precision AS
               $$ SELECT COM_OS_3(geom) $$
         LANGUAGE SQL;

CREATE OR REPLACE
         FUNCTION COM_LW_5(geom Geometry)
          RETURNS double precision AS
               $$ SELECT (ST_Xmax(geom) - ST_Xmin(geom)) 
                         - (ST_Ymax(geom) - ST_Ymin(geom)); $$
         LANGUAGE SQL;

CREATE OR REPLACE 
         FUNCTION COM_EigSeitzinger(geom Geometry)
          RETURNS double precision AS
               $$ SELECT COM_LW_5(geom); $$
         LANGUAGE SQL;

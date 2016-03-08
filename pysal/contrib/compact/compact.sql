CREATE OR REPLACE FUNCTION
    Polsby_popper(geom Geometry)
RETURNS real AS $$
    SELECT (4 * pi() * ST_Area(geom)) / (ST_Perimeter(geom)^2)
$$ LANGUAGE SQL

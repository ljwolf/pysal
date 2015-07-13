__all__ = ['geopandas', 'pandas', 'fiona']

def geopandas():
    try:
        import geopandas
        return True
    except:
        return False

def pandas():
    try:
        import geopandas
        return True
    except:
        return False

def fiona():
    try:
        import fiona
        return True
    except:
        return False

from __future__ import division
from model import Linear_Model
from solvers import ols_solver

class OLS(Linear_Model):
    """
    The lightweight preconfiguration class of the linear model superclass,
    overriding relevant detail.
    """
    def __init__(self, y, X, data = None, **kwargs):
        super(OLS, self).__init__(y, X, w=None, data=data, **kwargs)
        
        self._solver = ols_solver
        
        if type(self) == OLS:
            self._validate()

    def fit(self, inplace = True, **kwargs):
        if inplace:
            self._solver(self)
        else:
            newmod = self.__deepcopy__()
            return newmod.fit(inplace=True, **kwargs)

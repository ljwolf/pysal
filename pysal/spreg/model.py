from __future__ import division

import pandas
import copy

from collections import defaultdict
from numpy import array, hstack, allclose, newaxis

class Linear_Model(object):
    """
    A Linear model has at least:
    
    y           : str or iterable
                    dataframe key or iterable denoting the response variable
    X           : str or iterable
                     dataframe key or iterable denoting the design matrix
    w           : W
                    spatial weights object
    data        : dataframe
                    pandas dataframe supporting column retrieval
    **kwargs    : dict
                    variable length dictionary that gets flattened into model properties
                    while checking against the current dictionary to avoid clobbering
    """
    def __init__(self, y, X, w=None, data=None, **kwargs):
        """
        Setup model, independent from solution. To fit, call Model.fit()
        """
        #stick basic attributes inside of model
        self.y = y
        self.X = X # need to check for constant vector 
        self.w = w
        self.n = self.y.shape[0]
        
        if data is None:
            self.data = hstack((y, X))
        else:
            self.data = data
        
        ####### idea: store the large # of possible arguments in defaultdicts
        #config -> strings describing choices about model properties
        #options -> boolean flags about to indicate model properties
        self._options = defaultdict(bool)
        #self._options.update({k:v for k,v in kwargs.iteritems() if type(v) == bool})
        self._configs = defaultdict(str)
        #self._configs.update({k:v for k,v in kwargs.iteritems() if type(v) == str})
        if 'robust' in kwargs:
            if type(kwargs['robust']) == str:
                self.robust = self._options['robust'] = True #link the entries
                self._config['robust'] = kwargs['robust'].lower()
            elif kwargs['robust']:
                self.robust = self._options['robust'] = kwargs['robust']
                self._config['robust'] = 'hac' ##DEFAULTING ROBUSTNESS HERE FOR BOOL
            else:
                self.robust = self._options['robust'] = False
        
        #only add to toplevel if it won't clobber
        no_collision = [not hasattr(self, kwa) for kwa in kwargs]
        if all(no_collision):
            self.__dict__.update(kwargs)
        else:
            collided = kwargs.keys().index(True)
            raise Exception('Duplicate argument encountered:{}'.format(collided))
        
        #use pandas if provided
        papi = [str, list, pandas.DataFrame]
        argcheck = [type(arg) == t for t,arg in zip(papi, [y,X,data])]
        if all(argcheck):
            y = self.data[[y]].values
            if len(x) == 1:
                X = self.data[[X]].values
            else:
                X = self.data[X].values

        self._validate()

    #############
    #   PUBLIC  #
    #############
    def fit(self, inplace=True, **kwargs):
        """
        Fit a model using parameters passed.
        """
        ##execute solver-specific code here
        if inplace:
            pass
        else:
            newmod = self.__deepcopy__()
            newmod.fit(inplace, **kwargs)
    
    def center(self, inplace=True):
        Xcolmeans = self.X
        ymean = self.y.mean(axis=0)
        if inplace:
            self.X = self.X - self.X.mean(axis=0)[newaxis,:]
            self.y = self.y - self.y.mean()
            self._centered = True
        else:
            newmod = self.__deepcopy__()
            newmod.X = newmod.X - newmod.X.mean(axis=0)[newaxis,:]
            newmod.y = newmod.y - newmod.y.mean()
            newmod._centered = True
            return newmod

    def normalize(self, inplace=True, center=True):
        if inplace and center:
            self.center()
            self.X = self.X / self.X.std(axis=0)
            self.y = self.y / self.y.std()
            self._normalized = True
        elif center:
            newmod = self.center(inplace=False)
            newmod.X = newmod.X / newmod.X.std(axis=0)
            newmod.y = newmod.y / newmod.y.std()
            newmod._normalized=True
            return newmod
    
    ##############
    #   PRIVATE  #
    ##############
    def _validate(self):
        confs = self._check_conformal(self.y, self.x)
        ws = self._check_weights()
        if self._config['robust'] == hac:
            hacs = self._check_hac()
        return self._check_conformal(self.y,self.X) and self._check_weights()
    
    def _check_weights(self):
        if self.w is not None:     
            return self.w.n == self.n
        else:
            return True
 
    def _check_conformal(self, *args):
        return all([args[i].shape[0] == args[i+1].shape[0] for i in range(len(args)-1)])

    def _check_hac(self):
        try:
            t1 = self.gwk is not None
            t2 = np.diagonal(self.gwk) == np.eye(self.gwk.shape)
            return all([t1,t2])
        except:
            return False
    
    #############
    #   MAGIC   #
    #############
    def __deepcopy__(self):
        # must walk down all attributes, methods, and settings.
        # return independent model object with identical configuration.
        
        # an admittedly naive implementation
        newmod = Linear_Model(self.y, self.X, self.w, self.data)
        newmod.__dict__.update(copy.deepcopy(self.__dict__))
        return newmod

    def __copy__(self):
        # must walk down all attributes, methods, and settings, 
        # return model object with identical configuration, but points to original.

        # another naive implementation
        newmod = Linear_Model(self.y, self.X, self.w, self.data)
        newmod.__dict.update(self.__dict__)
        return newmod

    def __eq__(self, other):
        if type(other) != type(self):
            return False
        elif allclose(self,y, other.y) and allclose(self.X, self.X): # add to this list?
            return True

    def __ne__(self, other):
        return not self.__eq__(self, other)

    def __str__(self):
        try:
            return self.formula # with patsy?
        except:
            pass

    def __hash__(self):
        try:
            return hash([y,X]) # why would this be needed??
        except:
            pass

    def __len__(self):
        return self.Xs

    def __call__(self, inplace=True, **kwargs):
        self.fit(self, inplace, **kwargs) #allows for Model(method='ols') to fit the model

    ##it'd be cool to implement model pickling...

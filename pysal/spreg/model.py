from __future__ import division

import copy
import numpy as np

from pysal.core import check
if check.pandas():
    import pandas
from collections import defaultdict
from functools import partial

#just for fun, let's prebake some column reductions:
_colmedian = partial(np.median, axis=0)
_colmean = partial(np.mean, axis=0)
_colstd = partial(np.std, axis=0)
_colvar = partial(np.var, axis=0)
_colmax = partial(np.max, axis=0)
_colmin = partial(np.min, axis=0)

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
                    variable length dictionary that gets flattened into model
                    properties.
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
            self.data = np.hstack((y, X))
        else:
            self.data = data
        
        ####### idea: store the large # of possible arguments in defaultdicts
        # Later checks against these defaultdicts won't KeyError out
        # model subclasses can override defaults as needed
        # config -> strings describing choices about model properties, '' default
        # options -> boolean flags about to indicate model state, False default
        
        self._options = defaultdict(bool)
        #self._options.update({k:v for k,v in kwargs.iteritems() if type(v) == bool})
        self._configs = defaultdict(str)
        #self._configs.update({k:v for k,v in kwargs.iteritems() if type(v) == str})

        # For example, robust can be both an option and a config. Setting a
        # config should set an option to True, but setting an option, if it
        # requires a config, needs to set a default config. 
        
        # Here, if robust = 'white' or robust = 'hac' is passed, we set the
        # option to true and the config to the type of robustness requested.
        # If the user only passes robust=True, we default to 'hac'
        if 'robust' in kwargs:
            robust = kwargs.pop('robust')
            if type(robust) == str:
                self.robust = self._options['robust'] = True #link the entries
                self._configs['robust'] = robust.lower()
            elif robust:
                self.robust = self._options['robust'] = robust
                self._configs['robust'] = 'hac' ##DEFAULTING ROBUSTNESS HERE FOR BOOL
            else:
                self.robust = self._options['robust'] = False
        
        # Then, for other things, we can add them to the toplevel while
        # preventing them from clobbering each other. Python checks kwargs
        # for duplicates by default. At this point in init, this checks for:
        # X,y,w,data,_configs, _options
        no_collision = [not hasattr(self, kwa) for kwa in kwargs]
        if all(no_collision):
            self.__dict__.update(kwargs)
        else:
            collided = no_collision.index(True)
            raise Exception('Duplicate argument encountered:{}'.format(kwargs[collided]))
        
        # construct from pandas if the 'str', 'str', dataframe pattern is used.
        # need to think about how to also accomodate patsy spec
        papi = [str, list, pandas.DataFrame]
        argcheck = [type(arg) == t for t,arg in zip(papi, [y,X,data])]
        if all(argcheck):
            y = self.data[[y]].values
            if len(X) == 1:
                X = self.data[[X]].values
            else:
                X = self.data[X].values
        
        self._fitted = False
        # This should always end the init, and should get modified if 
        # subclasses need new tests. So, like, _check_regimes can get added to a
        # regime regression. I've written validate to inspect the object's
        # __dir__ and dynamically pull from all private methods starting with
        # "_check". So, all inherited models get a few _check* from here, but
        # _validate by adding new _check* methods.

        # Another thing is that the model classes should only validate
        # ONCE. So, for instance, if we call the generic Linear_Model.__init__()
        # from withina subclass, we want to _validate() THERE, not in the parent.
        
        if type(self) ==Linear_Model:
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
            self._fitted = True
            return None
        else:
            newmod = self.__deepcopy__()
            newmod.fit(inplace=True, **kwargs)
            return newmod
    
    def center(self, how=_colmean, inplace=True):
        """
        Center the data around some column reduction.
        """
        if inplace:
            self.X = self.X - how(self.X)[np.newaxis,:]
            self.y = self.y - how(self.y)
            self._centered = True
            return None
        else:
            newmod = self.__deepcopy__()
            newmod.center(how = how, inplace=True)
            newmod._centered = True
            return newmod

    def rescale(self, how=_colstd, inplace=True):
        """
        Scale the data by a column reduction.
        """
        if inplace:
            self.X = self.X / how(self.X)[np.newaxis,:]
            self.y = self.y / how(self.y)
            self._rescaled = True
            return None
        else:
            newmod = self.__deepcopy__()
            newmod.rescale(how=how, inplace=True)
            newmod._rescaled = True
            return newmod


    def normalize(self, inplace=True, **kwargs):
        """
        Normalize the data using a scale reduction and a centering reduction
        """
        if 'center_how' in kwargs:
            center_how = kwargs.pop('scale_how')
        else:
            center_how = _colmean

        if 'scale_how' in kwargs:
            scale_how = kwargs.pop('scale_how')
        else:
            scale_how = _colstd

        if inplace:
            self.center(how=center_how, inplace=True)
            self.rescale(how=scale_how, inplace=True)
            self._normalized = True
            return None
        else:
            newmod = self.center(how=center_how, inplace=False)
            newmod.rescale(how=scale_how)
            newmod._normalized=True
            return newmod
    
    ##############
    #   PRIVATE  #
    ##############
    def _validate(self):
        """
        This function searches for attributes of the model that start with
        "_check" and runs them to validate the model. Any "_check" method
        should return a tuple containing:

        (Test status, Error message)

        Then, if the test status is false, error message is raised by the
        validation function immediately.

        Think of this as the error handler for model validation, what the long
        USER.check_* statement chains were in old spreg inits. 
        
        If we wanted to do REAL error handling, we could define these errors and
        return them from the checks, but this is alright if we just want to use
        a generic Exception.
        """
        selfchecks = [fn for fn in dir(self) if fn.startswith('_check')]
        
        for check in selfchecks:
            status, msg = eval('self.{}()'.format(check))
            if not status:
                raise Exception(msg)
        return None

    def _check_weights(self):
        """
        if weights are passed, they should match the n in dimension
        """
        if self.w is not None:     
            if self.w.n == self.n:
                return True, None
            else:
                return False, "Weights are not the same dimension as data"
        else:
            return True, None
    
    def _check_design(self):
        """
        y and X must be passed, and they need to be conformal.
        """
        if self.__check_conformal(self.y.T, self.X):
            return True, None
        else:
            return False, "y and X vector are not conformal"
 
    def _check_hac(self):
        """
        if we are using hac robust estimators, the gwk matrix needs
        strictly ones in the diagonal
        """
        if self._configs['robust'] == 'hac':
            if hasattr(self, 'gwk'):
                t1 = self.gwk is not None
                t2 = np.diagonal(self.gwk).unique() == np.array([1])
                if all([t1,t2]):
                    return True, None
                else:
                    return False, "HAC Robustness requires ones on diagonal of gwt matrix"
            else:
                return False, "No gwk matrix provided to estimate HAC-robust estimators"
        else:
            return True, None

    #def _check_white(self):

    
    # since we will be checking for conformal matrices a lot, this might be
    # useful as a "meta" check
    def __check_conformal(self, *args, **kwargs):
        """
        check if a list of 2d matrices is conformal to an inner product,
        looking at the inner dimension. 

        n X m (dot) m X n (dot) n X t (dot) t x 1 -> conformal
        n X m (dot) n X m (dot) n X t (dot) t x y -> not conformal

        not sure what this would look like yet for 3-d alignment
        """
        if hasattr(kwargs, 'axis'):
            axis = axis
        else:
            axis = 0
        return all([args[i].shape[1] == args[i+1].shape[0] for i in range(len(args)-1)])
    
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
        elif np.allclose(self,y, other.y) and np.allclose(self.X, self.X): 
            # add to this list, check parameters? This is a design discussion.
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
            return hash([y,X]) # Maybe don't need this?
        except:
            pass

    def __len__(self):
        return self.X

    def __call__(self, inplace=True, **kwargs):
        self.fit(self, inplace, **kwargs) #allows for Model(method='ols') to fit the model

    ## it'd be cool to implement model pickling and maybe some numpy/patsy/pandas
    ## magic methods.

if __name__ == '__main__':
    import pysal
    
    db = pysal.open(pysal.examples.get_path('columbus.dbf'))

    X = db.by_col_array(['INC', 'CRIME'])
    y = db.by_col_array(['HOVAL'])

    mod = Linear_Model(y, X)

    #can transform model in place or create copy
    mod2 = mod.center(inplace=False)

    # Can pass robust as either str or bool, setting a sane default if bool 
    mod3 = Linear_Model(y, X, robust='white')
    print('Robust:', mod3._options['robust'], ' using', mod3._configs['robust'])

    # As an exmaple, I've implemented a test that checks if you've provided
    # the right parameters to conduct a hac-robust regression. These kinds of
    # tests can get extended.

    # right now, this just passes, but once linked, it would modify the model in
    # in place or return a copy of the model after being fit.

    mod_fit = mod.fit(inplace=False)
    print('mod_fit is fitted?:', mod_fit._fitted)
    
    print('mod is fitted?', mod._fitted)
    mod.fit()
    print('mod is fitted?:', mod._fitted)

    print('Our new model will fail out because we dont have a gwk matrix')
    print('    and robust=True defaults to hac')
    mod_fail = Linear_Model(y, X, robust = True) #defaults to 'hac', remember?

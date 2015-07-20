from __future__ import division
from model import Linear_Model
from utils import spdot, sphstack, spbroadcast
import numpy as np

def vcov(model, method):
    """
    Estimation of the variance-covariance matrix. Estimated nonrobust (default),
    by White (if model is roubst, type 'white') or HAC (if model is robust, type
    'hac'). 

    Parameters
    ----------

    model            : Regression object (OLS or TSLS)
                      output instance from a regression model

    gwk             : PySAL weights object
                      Optional. Spatial weights based on kernel functions
                      If provided, returns the HAC variance estimation
    sig2n_k         : boolean
                      If True, then use n-k to rescale the vc matrix.
                      If False, use n. (White only)

    Returns
    --------

    psi             : kxk array
                      Robust estimation of the variance-covariance

    """
    if method == 'tsls':
        xu = spbroadcast(model.h, model.u) #model.h is defined if an h param is passed
    else:
        xu = spbroadcast(model.x, model.u)
    if not model._options['robust']:  #check to see if we need to do robust estimation:
        return np.dot(model.sig2, model.xtxi)
    elif model._config['robust'] == 'hac':
        psi0 = spdot(xu.T, lag_spatial(model.gwk, xu))
    else:
        psi0 = spdot(xu.T, xu)
        if sig2n_k:
            psi0 = psi0 * (1. * model.n / (model.n - model.k))

    if tsls:
        psi1 = spdot(model.varb, model.zthhthi)
        psi = spdot(psi1, np.dot(psi0, psi1.T))
    else:
        psi = spdot(model.xtxi, np.dot(psi0, model.xtxi))

    return psi

def ols_solver(mod, inplace=True):
    """
    Solve the simplest linear model
    """
    if inplace:
        model = mod
        model.xty = spdot(model.x.T, mod.y)
        model.xtx = np.dot(model.x.T, model.x)

        model.xtxi = np.linalg.inv(np.dot(model.x.T, model.x))
        model.betas = np.dot(model.xtxi, model.xty)
        model.predy = spdot(model.x, model.betas)
        
        model.u = model.y - model.predy
        model.utu = np.sum(model.u **2)
        
        # So, with this model using n or n-k for the variance, we should decide
        # if .fit() should let someone override initial model options. So, say
        # we default to sig2n. But, the user should have some way of applying
        # which model variance they want at solve time, so like
        # model.fit('sig2n_k' = False). We'd probably have to let this rebind
        # and revalidate the model before proceeding with the fitting...
        if model._options['sig2n_k']:
            model.sig2 = model.utu / model.n
        else:
            model.sig2 = model.utu / (model.n - model.k)

        model.vm = vcov(model, 'ols')
        
        return model
    else:
        newmod = mod.__deepcopy__()
        # if we allow .fit()-time changes in options/configs, we need to
        # revalidate here
        return ols_solver(newmod, inplace=True)


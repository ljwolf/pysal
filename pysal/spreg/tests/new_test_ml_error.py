import unittest as ut
import pysal as ps
import numpy as np
import scipy as sci
from pysal.spreg.ml_error import ML_Error
from pysal.spreg import utils
from pysal.common import RTOL, ATOL

@ut.skipIf(int(scipy.__version__.split('.')[1] < 11), 
        "Maximum Likelihood requires SciPy version 11 or newer")
class TestMLError(ut.TestCase):
    def setUp(self):
        db =  pysal.open(pysal.examples.get_path("baltim.dbf"),'r')
        self.ds_name = "baltim.dbf"
        self.y_name = "PRICE"
        self.y = np.array(db.by_col(self.y_name)).T
        self.y.shape = (len(self.y),1)
        self.x_names = ["NROOM","AGE","SQFT"]
        self.x = np.array([db.by_col(var) for var in self.x_names]).T
        ww = pysal.open(pysal.examples.get_path("baltim_q.gal"))
        self.w = ww.read()
        ww.close()
        self.w_name = "baltim_q.gal"
        self.w.transform = 'r'

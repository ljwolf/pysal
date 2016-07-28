import unittest as ut
import pysal as ps
import numpy as np
import scipy as sci
from pysal.spreg.ml_error import ML_Error
from pysal.spreg import utils
from pysal.common import RTOL
ATOL = 1e-7
import os

TEST_DIR = os.path.dirname(__file__)

@ut.skipIf(int(sci.__version__.split('.')[1] < 11), 
        "Maximum Likelihood requires SciPy version 11 or newer")
class TestMLError(ut.TestCase):
    def setUp(self):
        db =  ps.open(ps.examples.get_path("baltim.dbf"),'r')
        self.ds_name = "baltim.dbf"
        self.y_name = "PRICE"
        self.y = np.array(db.by_col(self.y_name)).T
        self.y.shape = (len(self.y),1)
        self.x_names = ["LOTSZ", "SQFT", "AGE", "NROOM"]
        self.x = np.array([db.by_col(var) for var in self.x_names]).T
        ww = ps.open(ps.examples.get_path("baltim_q.gal"))
        self.w = ww.read()
        ww.close()
        self.w_name = "baltim_q.gal"
        self.w.transform = 'r'

    def _estimate_and_compare(self, method):
        reg = ML_Error(self.y, self.x, w=self.w,
                       name_y=self.y_name, name_x=self.x_names,
                       name_w=self.w_name, method=method)
        known = dict()
        known['aic'] = 1739.4233280274384
        known['e_filtered'] = np.load(TEST_DIR + '/data/ml_error_e_filtered.np')
        known['betas'] = np.array([[ 19.126389049 ],
                                   [  0.0982715236],
                                   [  0.3544517925],
                                   [ -0.2259779991],
                                   [  3.6848651245],
                                   [  0.6283378426]]) 
        known['logll'] = -864.7116640137192
        known['pr2'] = 0.5316089589864738
        known['predy'] = np.load(TEST_DIR + '/data/ml_error_predy.np')
        known['schwarz'] = 1756.1826186948188
        known['sig2'] = np.array([[ 194.14015907]])
        known['std_err'] = np.array([  5.35880832,  
                                       0.01588682,  
                                       0.16652636,
                                       0.06184595,
                                       1.09883335,
                                       0.06974104])
        known['utu'] = 59354.153717404341
        known['z_stat'] = np.array([[3.569149690865502,0.00035814175767763122],
                                    [6.1857254637717096, 6.1817510902973726e-10],
                                    [2.1285025537815772, 0.033295438443247442],
                                    [-3.6538852940111886, 0.0002583016991258926],
                                    [3.3534340139369272, 0.00079815442582055289],
                                    [9.00958575185882, 2.0683628014998281e-19]])
        if method is 'ORD':
            known['std_err'][-1] =  0.070645
            known['vm'] = np.load(TEST_DIR + '/data/ml_error_ord_vm.np')
            known['z_stat'][-1] =  [8.8943477782974032, 5.8763397738549035e-19]
        else:
            known['vm'] = np.load(TEST_DIR + '/data/ml_error_vm.np')
        for key in known.keys():
            errmsg = '{} does not match the reference value!'.format(key)
            np.testing.assert_allclose(known[key], getattr(reg, key), 
                                       rtol=RTOL, atol=ATOL, err_msg = errmsg)

    def test_full(self):
        self._estimate_and_compare('FULL')

    def test_lu(self):
        self._estimate_and_compare('LU')

    def test_ord(self):
        self._estimate_and_compare('ORD')

if __name__ == '__main__':
    ut.main()

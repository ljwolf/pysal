"""Unit test for Wsets module."""
import unittest
import pysal


class TestWsets(unittest.TestCase):
    """Unit test for Wsets module."""

    def test_w_union(self):
        """Unit test"""
        w1 = pysal.lat2W(4, 4)
        w2 = pysal.lat2W(6, 4)
        w3 = pysal.weights.Wsets.w_union(w1, w2)
        self.assertEqual(w1[0], w3[0])
        self.assertEqual(set(w1.neighbors[15]), set([11, 14]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14, 19]))
        self.assertEqual(set(w3.neighbors[15]), set([19, 11, 14]))

    def test_w_intersection(self):
        """Unit test"""
        w1 = pysal.lat2W(4, 4)
        w2 = pysal.lat2W(6, 4)
        w3 = pysal.weights.Wsets.w_union(w1, w2)
        self.assertEqual(w1[0], w3[0])
        self.assertEqual(set(w1.neighbors[15]), set([11, 14]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14, 19]))
        self.assertEqual(set(w3.neighbors[15]), set([19, 11, 14]))

    def test_w_difference(self):
        """Unit test"""
        w1 = pysal.lat2W(4, 4, rook=False)
        w2 = pysal.lat2W(4, 4, rook=True)
        w3 = pysal.weights.Wsets.w_difference(w1, w2, constrained=False)
        self.assertNotEqual(w1[0], w3[0])
        self.assertEqual(set(w1.neighbors[15]), set([10, 11, 14]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14]))
        self.assertEqual(set(w3.neighbors[15]), set([10]))

    def test_w_symmetric_difference(self):
        """Unit test"""
        w1 = pysal.lat2W(4, 4, rook=False)
        w2 = pysal.lat2W(6, 4, rook=True)
        w3 = pysal.weights.Wsets.w_symmetric_difference(
            w1, w2, constrained=False)
        self.assertNotEqual(w1[0], w3[0])
        self.assertEqual(set(w1.neighbors[15]), set([10, 11, 14]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14, 19]))
        self.assertEqual(set(w3.neighbors[15]), set([10, 19]))

    def test_w_subset(self):
        """Unit test"""
        w1 = pysal.lat2W(6, 4)
        ids = range(16)
        w2 = pysal.weights.Wsets.w_subset(w1, ids)
        self.assertEqual(w1[0], w2[0])
        self.assertEqual(set(w1.neighbors[15]), set([11, 14, 19]))
        self.assertEqual(set(w2.neighbors[15]), set([11, 14]))
    
    def test_w_stitch(self):
        w = pysal.lat2W(3,3)
        w_stitched = pysal.weights.Wsets.w_stitch([w]*3, back=1, forth=1)
        middle_0 = {'{t}-{ob}'.format(t=t, ob=ob):1 for t in range(3) 
                                                    for ob in (1,3)  }
        assert_dict_equal(middle_0, w_stitched['1-0'])
    
    def test_w_stack(self):
        w = pysal.lat2W(3,3)
        w_stacked = pysal.weights.Wsets.w_stack([w]*4)
        self.assertEquals(w_stacked.n, 4*9)
        
        #generate neighbor dictionary for 0 at arbitrary time
        template = lambda t: {'{t}-{obj}'.format(t=t, obj=obj):1.0 for obj in (1,3)}
        for time in range(4):
            assert_dict_equal(template(time), w_stacked['{}-0'.format(time)])

    def test_w_stitch_single(self):
        w = pysal.lat2W(3,3)
        w_stitched = pysal.weights.Wsets.w_stitch_single(w, 3)
        self.assertEqual(w_stitched.n, 9*3)

        #generate neighbor dictionary for 0 at arbitrary time
        known0 =  lambda t: {'{}-1'.format(t):1.0,'{}-3'.format(t):1.0}
        for t in range(3):
            assert_dict_equal(w_stitched['{}-0'.format(t)], known0(t))
        w_stitched =  pysal.weights.Wsets.w_stitch_single(w, 3, back=1)

        #bind arbitrary time dict to t=0 for these comparisons
        known0 = known0(0)
        assert_dict_equal(w_stitched['0-0'], known0)
        known0.update({'1-{}'.format(obj):1.0 for obj in (1,3)})
        assert_dict_equal(w_stitched['1-0'], known0)
        _ = [known0.pop('0-{}'.format(i)) for i in (1,3)]
        known0.update({'2-{}'.format(obj):1.0 for obj in (1,3)})
        assert_dict_equal(w_stitched['2-0'], known0)

        w_stitchedfb = pysal.weights.Wsets.w_stitch_single(w, 3, back=1, forth=1)

        first = {'{}-{}'.format(t,obj):1.0 for t in (0,1) for obj in (1,3)}
        assert_dict_equal(first, w_stitchedfb['0-0'])

        middle = {'{}-{}'.format(t,obj):1.0 for t in range(3) for obj in (1,3)}
        assert_dict_equal(w_stitchedfb['1-0'], middle)
        
        #last is the same as no forwards, since we run out of timesteps
        assert_dict_equal(w_stitched['2-0'], w_stitchedfb['2-0'])

def assert_dict_equal(d1, d2):
    assert set(d1.keys()) == set(d2.keys())
    v_d1 = [d2[k] == v for k,v in d1.items() ]
    assert all(v_d1)


suite = unittest.TestLoader().loadTestsFromTestCase(TestWsets)

if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    runner.run(suite)

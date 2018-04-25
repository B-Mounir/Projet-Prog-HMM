import unittest
import numpy as np
from HMM_class import *

HMM_tuple = (2, 2, np.array([[0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([[0.5, 0.5], [0.7, 0.3]]))
HMM_test = HMM(*HMM_tuple)

class hmmTest(unittest.TestCase):

    def test_nbl(self):
        HMM_test.nbl = 10
        self.assertEqual(str(HMM_test), str((10, 2, np.array([[0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.9]]), np.array([[0.5, 0.5], [0.7, 0.3]]))))

    def test_error_nbl(self):
        with self.assertRaises(ValueError):
            HMM_test.nbl = 2.5


TEST = hmmTest()
TEST.test_nbl()
TEST.test_error_nbl()

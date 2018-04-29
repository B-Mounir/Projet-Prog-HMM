"""
This module contains the tests for the HMM class
"""

import unittest
import math
import numpy as np

import HMM_class as HMM


class HMMTest(unittest.TestCase):
    def setUp(self):
        self.A = HMM.HMM(2, 2, np.array([[0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                         np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.B = HMM.HMM(2, 2, np.array([[0.741, 0.259]]), np.array([[0.0115, 0.9885], [0.5084, 0.4916]]),
                         np.array([[0.4547, 0.5453], [0.2089, 0.7911]]))


    def test_HMM(self):
        self.assertRaises(ValueError, HMM.HMM, 0, 2, np.array([[0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                            np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.assertRaises(ValueError, HMM.HMM, 2, 0, np.array([[0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                          np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.assertRaises(TypeError, HMM.HMM, 2, 2, [[0.5, 0.5]], np.array([[0.9, 0.1], [0.1, 0.9]]),
                          np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.assertRaises(ValueError, HMM.HMM, 2, 2, np.array([[-0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                          np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.assertRaises(ValueError, HMM.HMM, 2, 2, np.array([[0.5, 1.5]]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                          np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.assertRaises(ValueError, HMM.HMM, 2, 2, np.array([[0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.8]]),
                          np.array([[0.5, 0.5], [0.7, 0.3]]))
        self.assertRaises(ValueError, HMM.HMM, 2, 2, np.array([[0.5, 0.5]]), np.array([[0.9, 0.1], [0.1, 0.9]]),
                          np.array([[0.5, 0.5], [0.75, 0.3]]))

    def test_save_load(self):
        h = self.A
        h.save("./temp")
        h = HMM.HMM.load("./temp")
        self.assertEqual(h.nbl, 2)
        self.assertEqual(h.nbs, 2)
        np.testing.assert_array_equal(h.initial, np.array([[0.5, 0.5]]))
        np.testing.assert_array_equal(h.transitions, np.array([[0.9, 0.1], [0.1, 0.9]]))
        np.testing.assert_array_equal(h.emissions, np.array([[0.5, 0.5], [0.7, 0.3]]))

    def test_PFw_PBw(self):
        h = self.A
        self.assertEqual(h.pfw((0,)), 0.6)
        self.assertEqual(h.pfw((1,)), 0.4)
        for i in range(100):
            w = h.gen_rand(10)
            self.assertAlmostEqual(h.pfw(w), h.pbw(w))

    def test_predit(self):
        for i in range(100):
            h = HMM.HMM.gen_HMM(5, 2)
            w = h.gen_rand(h,10)
            w0 = w + [0]
            w1 = w + [1]
            x = h.predit(h, w)
            if h.pfw(h, w0) > h.pfw(h, w1):
                self.assertEqual(0, x)
            else:
                self.assertEqual(1, x)

    def test_Viterbi(self):
        h = self.B
        w = [1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        (lc, p) = h.viterbi(w)
        #self.assertEqual(lc, [0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        self.assertAlmostEqual(math.log2(p), -15.816435284201352)

if __name__ == "__main__":
    unittest.main()

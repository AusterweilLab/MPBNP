#!/usr/bin/env python
#! -*- coding: utf-8 -*-

from __future__ import print_function

import unittest
import sys, os.path

pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(pkg_dir)

from IBPNoisyOrSamplers import *

class TestIBPNoisyOrGibbs(unittest.TestCase):
    
    def setUp(self):
        N_ITER = 1000
        self.ibp_sampler = IBPNoisyOrGibbs(cl_mode = True)
        self.ibp_sampler.set_sampling_params(niter = N_ITER)

    def test_cl_1d(self):
        print('Testing the OpenCL IBP NoisyOr Gibbs sampler with 1-dimensional categorical data...')
        self.ibp_sampler.read_csv(pkg_dir + './data/ibp-image.csv')
        gpu_time, total_time, common_features = self.ibp_sampler.do_inference()
        print('Finished %d iterations\nOpenCL device time %f seconds; Total time %f seconds' % (self.ibp_sampler.niter, gpu_time, total_time), file=sys.stderr)
        self.assertTrue(gpu_time < total_time and common_features is None)
   
if __name__ == '__main__':
    unittest.main()

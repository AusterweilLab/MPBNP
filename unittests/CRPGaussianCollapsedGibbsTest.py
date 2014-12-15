#!/usr/bin/env python
#! -*- coding: utf-8 -*-

from __future__ import print_function

import unittest
import sys, os.path

pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(pkg_dir)

from CRPGaussianSamplers import *

class TestCRPGaussianCollapsedGibbs(unittest.TestCase):
    

    def setUp(self):
        N_ITER = 1000
        self.crp_sampler = CRPGaussianCollapsedGibbs(cl_mode = True)
        self.crp_sampler.set_sampling_params(niter = N_ITER)

    def test_cl_1d(self):
        print('Testing the OpenCL Gaussian Collapsed Gibbs sampler with 1-dimensional Gaussian data...')
        obs = np.hstack((np.random.normal(1, 1, 81), np.random.normal(20, 1, 81), np.random.normal(10,1,300)))
        self.crp_sampler.direct_read_obs(obs)
        gpu_time, total_time, common_clusters = self.crp_sampler.do_inference()
        print('Finished %d iterations\nOpenCL device time %f seconds; Total time %f seconds' % (self.crp_sampler.niter, gpu_time, total_time), file=sys.stderr)
        self.assertTrue(gpu_time < total_time and len(common_clusters) > 0)

    def test_cl_2d(self):
        print('Testing the OpenCL Gaussian Collapsed Gibbs sampler with 1-dimensional Gaussian data...')
        obs = np.vstack((np.random.multivariate_normal(mean = [1., 1.], cov = [[.1, 0], [0, .1]], size = 12),
                         np.random.multivariate_normal(mean = [55., 52.], cov = [[0.1, 0], [0, .1]], size = 12)))
        self.crp_sampler.direct_read_obs(obs)
        gpu_time, total_time, common_clusters = self.crp_sampler.do_inference()
        print('Finished %d iterations\nOpenCL device time %f seconds; Total time %f seconds' % (self.crp_sampler.niter, gpu_time, total_time), file=sys.stderr)
        self.assertTrue(gpu_time < total_time and len(common_clusters) > 0)

    
if __name__ == '__main__':
    unittest.main()

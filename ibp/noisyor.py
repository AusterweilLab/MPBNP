#!/usr/bin/env python2
#-*- coding: utf-8 -*-

from __future__ import print_function, division
import sys, os.path, itertools, cPickle
pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(pkg_dir)

from fractions import gcd
from scipy.stats import poisson
from MPBNP import *
from MPBNP import BaseSampler, BasePredictor
from numpy import matlib
from scipy.special import gammaln
from scipy.misc import comb
#import pandas as pd
    
np.set_printoptions(suppress=True)

#TODO: optimize buffer creation/when data really needs to be initialized
#TODO: and how to get uninitiailized data allocated on a device

class Gibbs(BaseSampler):

    def __init__(self, cl_mode = True, cl_device = None, record_best = True,
                 alpha = None, lam = 0.99, theta = 0.25, epislon = 0.01,
                 sim_annealP = True, init_k = 10, T_init=40., anneal_rate = .99,
                 splitProb = 0.8, split_mergeP = True, split_steps=20,
                 splitAdj = 1/150):
        """Initialize the class.
        """
        BaseSampler.__init__(self, cl_mode = cl_mode, cl_device = cl_device, record_best = record_best)

        np.set_printoptions(precision=3)

        if cl_mode:
            program_str = open(pkg_dir + 'MPBNP/ibp/kernels/ibp_noisyor_cl.c', 'r').read()
            self.prg = cl.Program(self.ctx, program_str).build()

            self.p_mul_logprob_z_data = cl.Kernel(self.prg, 'logprob_z_data').\
                        get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, self.device)
            self.p_mul_sample_y = cl.Kernel(self.prg, 'sample_y').\
                        get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, self.device)
            self.p_mul_sample_z = cl.Kernel(self.prg, 'sample_z').\
                        get_work_group_info(cl.kernel_work_group_info.PREFERRED_WORK_GROUP_SIZE_MULTIPLE, self.device)
            self.k_max_new = 4


        self.alpha = alpha # tendency to generate new features
        self.k = init_k    # initial number of features
        self.theta = theta # prior probability that a pixel is on in a feature image
        self.lam = lam # effecacy of a feature
        self.epislon = epislon # probability that a pixel is on by change in an actual image
        self.samples = {'z': [], 'y': []} # sample storage, to be pickled
        self.sim_annealP = sim_annealP
        self.splitProb = splitProb
        if sim_annealP:
            self.anneal_rate = anneal_rate
            self.T = T_init #current simulated annealing temperature
        else:
            self.T = 1
            self.anneal_rate = 1.
        self.split_mergeP = split_mergeP
        self.split_steps = split_steps
        self.splitAdj = splitAdj
        self.float_size = np.array(1,dtype=np.float32).nbytes
        self.int_size = np.array(1,dtype=np.int32).nbytes

    def read_csv(self, filepath, header=True):
        """Read the data from a csv file.
        """
        if use_pandas:
            BaseSampler.read_csv(self, filepath, header=False)
            self.img_w, self.img_h = None, None
            self.img_w = self.obs[0,0]
            self.img_h = self.obs.shape[1] // self.img_w

            if (self.obs.shape[1]-1) % self.img_w != 0:
                raise Exception('The sampler does not understand the format of the data. Did you forget to specify image width in the data file?')
            self.obs = np.require(self.obs[:,1:], dtype=np.int32, requirements=['C','A'])
            if self.cl_mode:
                self.d_obs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.obs.astype(np.int32))

        else:
            BaseSampler.read_csv(self, filepath, header)
            # convert the data to floats
            for row in self.obs:
                if self.img_w is None:
                    self.img_w = int(row[0])
                    if self.img_w == 0 or (len(row)-1) % self.img_w != 0:
                        raise Exception('The sampler does not understand the format of the data. Did you forget to specify image width in the data file?')
                self.new_obs.append([int(_) for _ in row])

                self.obs = np.array(self.new_obs)[:,1:]
            if self.cl_mode:
                self.d_obs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.obs.astype(np.int32))

        self.d = self.obs.shape[1]
        self.img_h = int(self.d / self.img_w)
        #self.alpha = float(self.N) * 5
        self.alpha = np.log(float(self.N))
        return

    def direct_read_obs(self, obs,img_w=None):
        #BaseSampler.read_csv(self, obs)
        BaseSampler.direct_read_obs(self,obs)
        self.N = self.obs.shape[0]
        self.d = self.obs.shape[1]
        if img_w is None:
            self.img_w = self.d
        else:
            self.img_w = img_w
        self.img_h = int(self.d/self.img_w)
        #self.alpha = float(self.N) * 5
        self.alpha = np.log(float(self.N)) 
        #Adding Ting's conversation juju
        self.new_obs = []
        for row in self.obs:
            self.new_obs.append([int(_) for _ in row])
            self.obs = np.array(self.new_obs)[:,1:]
        if self.cl_mode:
            self.d_obs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.obs.astype(np.int32))


        
    def do_inference(self, init_y = None, init_z = None, output_file = None):
        """Perform inference on the given observations assuming data are generated by an IBP model
        with noisy-or as the likelihood function.
        @param init_y: An initial feature image matrix, where values are 0 or 1
        @param init_z: An initial feature ownership matrix, where values are 0 or 1
        """
        BaseSampler.do_inference(self, output_file=None)
        #self.k = 2
        if init_y is None:
          #  init_y = np.array([[1, 1, 0, 0], [0,0,1,1]]) 
            init_y = np.random.randint(0, 2, (self.k, self.d))
        else:
            assert(type(init_y) is np.ndarray)
            assert(init_y.shape == (self.k, self.d))
        if init_z is None:
           # init_z = np.array([[0,1],[0,1],[1,0],[1,0],[0,1],[0,1],[1,0],[1,0]])
            init_z = np.hstack((np.ones(shape=(self.N,1),dtype=np.int32),
                                np.random.randint(0, 2, (len(self.obs), self.k-1))))
        else:
            assert(type(init_z) is np.ndarray)
            assert(init_z.shape == (len(self.obs), self.k))

        if self.cl_mode:
            timing_stats = self._cl_infer_yz(init_y, init_z, output_file)
        else:
            timing_stats = self._infer_yz(init_y, init_z, output_file)

        # report the results
        if output_file is sys.stdout:
            if self.record_best:
                final_y, final_z = self.best_sample[0]
                num_of_feats = final_z.shape[1]
                print('parameter,value',
                      'alpha,%f' % self.alpha, 'lambda,%f' % self.lam, 'theta,%f' % self.theta,
                      'epislon,%f' % self.epislon, 'inferred_K,%d' % num_of_feats,
                      'gpu_time,%f' % timing_stats[0], 'total_time,%f' % timing_stats[1],
                      file = output_file, sep='\n')

                np.savetxt(output_file, final_z, fmt="%d", comments='', delimiter=',',
                           header=','.join(['feature%d' % _ for _ in range(num_of_feats)]))

                for k in np.arange(num_of_feats):
                    print('Feature %d\n---------' % k, file = output_file)
                    np.savetxt(output_file, final_y[k].reshape(self.img_w, self.img_h),
                               fmt="%d", delimiter=',')

                print("input objs:", file=output_file)
                print(self.obs, file=output_file)
                print("outputting reconstructed objects:",file=output_file)
                obj_recon = np.dot(final_z, final_y)
                np.savetxt(output_file, obj_recon, fmt="%d", comments="", delimiter=",")
        elif output_file is not None:
            if self.record_best:
                final_y, final_z = self.best_sample[0]
                num_of_feats = final_z.shape[1]
                try: os.mkdir(output_file)
                except: pass
                print('parameter,value',
                      'alpha,%f' % self.alpha, 'lambda,%f' % self.lam, 'theta,%f' % self.theta,
                      'epislon,%f' % self.epislon, 'inferred_K,%d' % num_of_feats,
                      'gpu_time,%f' % timing_stats[0], 'total_time,%f' % timing_stats[1],
                      file = gzip.open(output_file + 'parameters.csv.gz', 'w'), sep = '\n')
                
                np.savetxt(gzip.open(output_file + 'feature_ownership.csv.gz', 'w'), final_z,
                           fmt="%d", comments='', delimiter=',',
                           header=','.join(['feature%d' % _ for _ in range(num_of_feats)]))

                for k in np.arange(num_of_feats):
                    np.savetxt(gzip.open(output_file + 'feature_%d_image.csv.gz' % k, 'w'),
                               final_y[k].reshape(self.img_w, self.img_h), fmt="%d", delimiter=',')
            else:
                try: os.mkdir(output_file)
                except: pass
                print('parameter,value',
                      'alpha,%f' % self.alpha, 'lambda,%f' % self.lam, 'theta,%f' % self.theta,
                      'epislon,%f' % self.epislon,
                      'gpu_time,%f' % timing_stats[0], 'total_time,%f' % timing_stats[1],
                      file = gzip.open(output_file + 'parameters.csv.gz', 'w'), sep = '\n')
                np.savez_compressed(output_file + 'feature_ownership.npz', self.samples['z'])
                np.savez_compressed(output_file + 'feature_images.npz', self.samples['y'])

        return timing_stats
                
    def _infer_yz(self, init_y, init_z, output_file):
        """Wrapper function to start the inference on y and z.
        This function is not supposed to directly invoked by an end user.
        @param init_y: Passed in from do_inference()
        @param init_z: Passed in from do_inference()
        """
        cur_y = init_y
        cur_z = init_z

        a_time = time()
        self.auto_save_sample(sample = (cur_y, cur_z))
        for i in xrange(self.niter):
            temp_cur_y = self._infer_y(cur_y, cur_z)
            temp_cur_y, temp_cur_z = self._infer_z(temp_cur_y, cur_z)
            #self._sample_lam(cur_y, cur_z)

            if self.record_best:
                if self.auto_save_sample(sample = (temp_cur_y, temp_cur_z)):
                    cur_y, cur_z = temp_cur_y, temp_cur_z
                if self.no_improvement():
                    break
                
            elif i >= self.burnin:
                cur_y, cur_z = temp_cur_y, temp_cur_z
                self.samples['z'].append(cur_z)
                self.samples['y'].append(cur_y)

        self.total_time += time() - a_time
        return self.gpu_time, self.total_time, None

    def _infer_y(self, cur_y, cur_z):
        """Infer feature images
        """
        # calculate the prior probability that a pixel is on
        y_on_log_prob = np.log(self.theta) * np.ones(cur_y.shape)
        y_off_log_prob = np.log(1. - self.theta) * np.ones(cur_y.shape)

        # calculate the likelihood
        on_loglik = np.empty(cur_y.shape)
        off_loglik = np.empty(cur_y.shape)
        for row in xrange(cur_y.shape[0]):
            affected_data_index = np.where(cur_z[:,row] == 1)
            for col in xrange(cur_y.shape[1]):
                old_value = cur_y[row, col]
                cur_y[row, col] = 1
                on_loglik[row, col] = self._loglik_nth(cur_y, cur_z, n = affected_data_index)
                cur_y[row, col] = 0
                off_loglik[row, col] = self._loglik_nth(cur_y, cur_z, n = affected_data_index)
                cur_y[row, col] = old_value

        # add to the prior
        y_on_log_prob += on_loglik
        y_off_log_prob += off_loglik

        ew_max = np.maximum(y_on_log_prob, y_off_log_prob)
        y_on_log_prob -= ew_max
        y_off_log_prob -= ew_max
        
        # normalize
        y_on_prob = np.exp(y_on_log_prob) / (np.exp(y_on_log_prob) + np.exp(y_off_log_prob))
        cur_y = np.random.binomial(1, y_on_prob)

        return cur_y

    def _infer_z(self, cur_y, cur_z):
        """Infer feature ownership
        """
        N = float(len(self.obs))
        z_col_sum = cur_z.sum(axis = 0)

        # calculate the IBP prior on feature ownership for existing features
        m_minus = z_col_sum - cur_z
        on_prob = m_minus / N
        off_prob = 1 - m_minus / N
        
        # add loglikelihood of data
        for row in xrange(cur_z.shape[0]):
            for col in xrange(cur_z.shape[1]):
                old_value = cur_z[row, col]
                cur_z[row, col] = 1
                on_prob[row, col] = on_prob[row, col] * np.exp(self._loglik_nth(cur_y, cur_z, n = row))
                cur_z[row, col] = 0
                off_prob[row, col] = off_prob[row, col] * np.exp(self._loglik_nth(cur_y, cur_z, n = row))
                cur_z[row, col] = old_value

        # normalize the probability
        on_prob = on_prob / (on_prob + off_prob)

        # sample the values
        cur_z = np.random.binomial(1, on_prob)

        # sample new features use importance sampling
        k_new = self._sample_k_new(cur_y, cur_z)
        if k_new:
            cur_y, cur_z = k_new
        
        # delete empty feature images
        non_empty_feat_img = np.where(cur_y.sum(axis = 1) > 0)
        cur_y = cur_y[non_empty_feat_img[0],:]
        cur_z = cur_z[:,non_empty_feat_img[0]]

        # delete null features
        active_feat_col = np.where(cur_z.sum(axis = 0) > 0)
        cur_z = cur_z[:,active_feat_col[0]]
        cur_y = cur_y[active_feat_col[0],:]

        # the above two steps need to be done before sampling new features
        # because new features are initialized randomly
        
        # update self.k
        self.k = cur_z.shape[1]
        
        return cur_y, cur_z

    def _sample_k_new(self, cur_y, cur_z):
        """Sample new features for all rows using Metropolis hastings.
        (This is a heuristic strategy aiming for easy parallelization in an 
        equivalent GPU implementation. We here have effectively treated the
        current Z as a snapshot frozen in time, and each new k is based on
        this frozen snapshot of Z. In a more correct procedure, we should
        go through the rows and sample k new for each row given all previously
        sampled new ks.)
        """
        N = float(len(self.obs))
        #old_loglik = self._loglik(cur_y, cur_z)

        k_new_count = np.random.poisson(self.alpha / N)
        if k_new_count == 0: return False
            
        # modify the feature ownership matrix
        cur_z_new = np.hstack((cur_z, np.random.randint(0, 2, size = (cur_z.shape[0], k_new_count))))
        #cur_z_new[:, [xrange(-k_new_count,0)]] = 1
        # propose feature images by sampling from the prior distribution
        cur_y_new = np.vstack((cur_y, np.random.binomial(1, self.theta, (k_new_count, self.d))))
        
        return cur_y_new.astype(np.int32), cur_z_new.astype(np.int32)

    def _sample_lam(self, cur_y, cur_z):
        """Resample the value of lambda.
        """
        old_loglik = self._loglik(cur_y, cur_z)
        old_lam = self.lam
    
        # modify the feature ownership matrix
        self.lam = np.random.beta(1,1)
        new_loglik = self._loglik(cur_y, cur_z)
        move_prob = 1 / (1 + np.exp(old_loglik - new_loglik));
        if random.random() < move_prob:
            pass
        else:
            self.lam = old_lam

    def _sample_epislon(self, cur_y, cur_z):
        """Resample the value of epislon
        """
        old_loglik = self._loglik(cur_y, cur_z)
        old_epislon = self.epislon
    
        # modify the feature ownership matrix
        self.epislon = np.random.beta(1,1)
        new_loglik = self._loglik(cur_y, cur_z)
        move_prob = 1 / (1 + np.exp(old_loglik - new_loglik));
        if random.random() < move_prob:
            pass
        else:
            self.epislon = old_epislon

    def _loglik_nth(self, cur_y, cur_z, n):
        """Calculate the loglikelihood of the nth data point
        given Y and Z.
        """
        assert(cur_z.shape[1] == cur_y.shape[0])
                
        not_on_p = np.power(1. - self.lam, np.dot(cur_z[n], cur_y)) * (1. - self.epislon)
        loglik = np.log(np.abs(self.obs[n] - not_on_p)).sum()
        return loglik

    def _loglik(self, cur_y, cur_z):
        """Calculate the loglikelihood of data given Y and Z.
        """
        assert(cur_z.shape[1] == cur_y.shape[0])

        n_by_d = np.dot(cur_z, cur_y)
        not_on_p = np.power(1. - self.lam, n_by_d) * (1. - self.epislon)
        loglik_mat = np.log(np.abs(self.obs - not_on_p))
        return loglik_mat.sum()

    def _cl_infer_yz(self, init_y, init_z, output_file = None):
        """Wrapper function to start the inference on y and z.
        This function is not supposed to directly invoked by an end user.
        @param init_y: Passed in from do_inference()
        @param init_z: Passed in from do_inference()
        """
        cur_y = init_y.astype(np.int32)
        cur_z = init_z.astype(np.int32)
        # cur_y = np.require(np.array([[0, 0, 1, 1],[1,1,0,0]], dtype=np.int32),
        #                    dtype=np.int32, requirements=['C','A'])
        # cur_z = np.require(np.array([[1, 0], [1, 0],
        #                              [0, 1], [0, 1],
        #                              [1, 0], [1, 0],
        #                              [0, 1], [0, 1],
        #                              [1, 0], [1, 0],
        #                              [0, 1], [0, 1],
        #                              [1, 0], [1, 0],
        #                              [0, 1], [0, 1],
        #                              [1, 0], [1, 0],
        #                              [0, 1], [0, 1],
        #                              [1, 0], [1, 0],
        #                              [0, 1], [0, 1],
        #                              [1, 0], [1, 0],
        #                              [0, 1], [0, 1],
        #                              [1, 0], [1, 0],
        #                              [0, 1], [0, 1]], dtype=np.int32),
        #                    dtype=np.int32, requirements=['C','A'])
        # self.k = 2
        self.tmp_y = cur_y
        self.tmp_z = cur_z

        debugP = False

        d_cur_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = cur_y.astype(np.int32))
        d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = cur_z.astype(np.int32))

        self.auto_save_sample(sample = (d_cur_y, d_cur_z))

        
        for i in xrange(self.niter):
            self.T = max(self.T*self.anneal_rate,1.)
            if (i % (self.niter/10)) == 0:
                print("%d sweeps of %d with T %f and K %d" % (i,self.niter,self.T, self.k))
            a_time = time()

            if self.split_mergeP and ((i % self.split_steps) == 0):
                #split/merge step!
                #print("trying a split-merge step")
                d_cur_y, d_cur_z = self._cl_split_merge_anneal_step(d_cur_y, d_cur_z)
            else:
                if debugP:
                    cl.enqueue_copy(self.queue, self.tmp_z, d_cur_z)
                    cl.enqueue_copy(self.queue, self.tmp_y, d_cur_y)

                    print("pre-infer_y")
                    print("cur_y:")
                    print(self.tmp_y)
                    print("cur_z")
                    print(self.tmp_z)
                if self.N < 201:
                    d_cur_y = self._cl_infer_y(d_cur_y, d_cur_z)
                else:
                    d_cur_y = self._cl_infer_y2(d_cur_y, d_cur_z)
                if debugP:
                    cl.enqueue_copy(self.queue, self.tmp_z, d_cur_z)
                    cl.enqueue_copy(self.queue, self.tmp_y, d_cur_y)

                    print("pre-infer_z")
                    print("cur_y:")
                    print(self.tmp_y)
                    print("cur_z")
                    print(self.tmp_z)
                d_cur_z = self._cl_infer_z(d_cur_y, d_cur_z)
                self.gpu_time += time() - a_time
                cl.enqueue_copy(self.queue, self.tmp_z, d_cur_z)
                cl.enqueue_copy(self.queue, self.tmp_y, d_cur_y)
            
                if debugP:
                    print("pre- create new feats")
                    print("cur_y:")
                    print(self.tmp_y)
                    print("cur_z")
                    print(self.tmp_z)
                d_cur_y, d_cur_z = self._cl_infer_k_newJLA2(self.tmp_y, self.tmp_z)
 
            if self.record_best:
                # if self.k != cur_z.shape[1]:
                #     cur_z = np.require(np.zeros(shape = (self.obs.shape[0], self.k), dtype=np.int32),
                #                                  dtype = np.int32, requirements=['C','A'])
                #     cur_y = np.require(np.zeros(shape = (self.k, self.obs.shape[1]), dtype=np.int32),
                #                                  dtype = np.int32, requirements=['C','A'])
                # cl.enqueue_copy(self.queue, cur_z, d_cur_z)
                # cl.enqueue_copy(self.queue, cur_y, d_cur_y)
                ignoreVal = None
                if self.cl_mode:
                    ignoreVal = self.auto_save_sample(sample=(d_cur_y,d_cur_z))
                else:
                    ignoreVal = self.auto_save_sample(sample = (cur_y, cur_z))
                #if self.no_improvement(1000):
                #    break                    
            elif i >= self.burnin:
                #cur_y, cur_z = temp_cur_y, temp_cur_z
                cl.enqueue_copy(self.queue, cur_z, d_cur_z)
                cl.enqueue_copy(self.queue, cur_y, d_cur_y)
                self.samples['z'].append(cur_z)
                self.samples['y'].append(cur_y)
            
            self.total_time += time() - a_time

        return self.gpu_time, self.total_time, None

    def _cl_infer_y(self, d_cur_y, d_cur_z):
        """Infer feature images
        """

        K = self.k

        if K > 0:
        
            N = self.obs.shape[0]
            D = self.obs.shape[1]

            # d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                         hostbuf = np.require(np.zeros(shape = self.obs.shape, dtype=np.int32),
            #                                              dtype = np.int32, requirements=['C','A']))

            d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(N*D*np.dtype('int32').itemsize))

            self.prg.init_bufferInt(self.queue, (N*D,1), None, d_obj_recon, np.int32(N*D), np.int32(0))


            self.prg.compute_recon_objs(self.queue, (N, K, D), None,
                                    d_cur_y, d_cur_z, d_obj_recon,
                                    np.int32(N), np.int32(D), np.int32(K))

            d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                               hostbuf=np.random.random(size = (K, D)).astype(np.float32))

            d_lp_off = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(K*D*np.dtype('float32').itemsize))
            # hostbuf = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                      dtype = np.float32, requirements=['C','A']))
            d_lp_on = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(K*D*np.dtype('float32').itemsize))

            self.prg.init_2bufferFlt(self.queue, (K*D,1), None, d_lp_off, d_lp_on, np.int32(K*D), np.float32(0.))

            # hostbuf = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                     dtype = np.float32, requirements=['C','A']))

            # d_lp_off2 = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                      hostbuf = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                                           dtype = np.float32, requirements=['C','A']))
            # d_lp_on2 = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                     hostbuf = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                                         dtype = np.float32, requirements=['C','A']))
            
            maxWorkGroupSize = np.int32(self.device.max_work_group_size/2)
            maxLocalMem = self.device.local_mem_size

            workGroupSize = int(min(maxLocalMem//np.dtype(np.float32).itemsize, maxWorkGroupSize))
            tmpLocMemNPFlt = np.empty(workGroupSize,dtype=np.float32)
            d_locmemFlt = cl.LocalMemory(tmpLocMemNPFlt.nbytes)
            curNumToRun = min(int(workGroupSize/2), N)
            numToRun = N
            numPrevRun = 0
            #TODO: this could be made faster... 
            while numToRun > 0:
                self.prg.calc_y_lps_old(self.queue, (curNumToRun,K,D), (curNumToRun, 1,1), d_locmemFlt, d_cur_y, d_cur_z,
                                    d_obj_recon, self.d_obs, d_lp_off, d_lp_on,
                                    np.int32(N), np.int32(D), np.int32(K), np.int32(numPrevRun),
                                    np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))
                # self.prg.calc_y_lps_noLcl(self.queue, (curNumToRun,K,D), (curNumToRun, 1,1), d_locmemFlt, d_cur_y, d_cur_z,
                #                     d_obj_recon, self.d_obs, d_lp_off2, d_lp_on2,
                #                     np.int32(N), np.int32(D), np.int32(K), np.int32(numPrevRun),
                #                     np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))
                numToRun -= curNumToRun
                numPrevRun += curNumToRun



            # check_y_lps_on = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                                            dtype = np.float32, requirements=['C','A'])
            # check_y_lps_off = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                                            dtype = np.float32, requirements=['C','A'])
            # cl.enqueue_copy(self.queue,check_y_lps_off, d_lp_off)
            # cl.enqueue_copy(self.queue,check_y_lps_on, d_lp_on)
            # #
            # check_y_lps_on2 = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                                           dtype = np.float32, requirements=['C','A'])
            # check_y_lps_off2 = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                                           dtype = np.float32, requirements=['C','A'])
            # cl.enqueue_copy(self.queue,check_y_lps_off2, d_lp_off2)
            # cl.enqueue_copy(self.queue,check_y_lps_on2, d_lp_on2)
            self.prg.sample_y_pre_calc(self.queue, (K,D, 1), None, d_cur_y, d_lp_off, d_lp_on,
                                       d_rand, np.int32(K), np.int32(D))
        return d_cur_y


    def _cl_infer_y2(self, d_cur_y, d_cur_z):
        """Infer feature images
        """

        K = self.k

        if K > 0:
        
            N = self.obs.shape[0]
            D = self.obs.shape[1]

            # d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                         hostbuf = np.require(np.zeros(shape = self.obs.shape, dtype=np.int32),
            #                                              dtype = np.int32, requirements=['C','A']))

            d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=(N*D*np.dtype('int32').itemsize))

            self.prg.init_bufferInt(self.queue, (N*D,1), None, d_obj_recon, np.int32(N*D), np.int32(0))

            self.prg.compute_recon_objs(self.queue, (N, K, D), None,
                                    d_cur_y, d_cur_z, d_obj_recon,
                                    np.int32(N), np.int32(D), np.int32(K))

            d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                               hostbuf=np.random.random(size = (K, D)).astype(np.float32))
            #buff_size = (self.tmp_y.astype(np.float32)).nbytes
            d_lp_off = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(K*D*np.dtype('float32').itemsize))
            # hostbuf = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                      dtype = np.float32, requirements=['C','A']))
            d_lp_on = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(K*D*np.dtype('float32').itemsize))

            self.prg.init_2bufferFlt(self.queue, (K*D,1), None, d_lp_off,d_lp_on, np.int32(K*D), np.float32(0.))
            # hostbuf = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                     dtype = np.float32, requirements=['C','A']))

            # d_lp_off2 = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                      hostbuf = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                                           dtype = np.float32, requirements=['C','A']))
            # d_lp_on2 = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                     hostbuf = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                                         dtype = np.float32, requirements=['C','A']))
            
            maxWorkGroupSize = np.int32(self.device.max_work_group_size/2)
            maxLocalMem = self.device.local_mem_size

            workGroupSize = int(min(maxLocalMem//np.dtype(np.float32).itemsize, maxWorkGroupSize))
            tmpLocMemNPFlt = np.empty(workGroupSize,dtype=np.float32)
            d_locmemFlt = cl.LocalMemory(tmpLocMemNPFlt.nbytes)
            curNumToRun = min(int(workGroupSize/2), N)
            numToRun = N
            numPrevRun = 0
            #TODO: this could be made faster... 
            while numToRun > 0:
                #issues when I don't split them....
                self.prg.calc_y_lp_off(self.queue, (curNumToRun,K,D), (curNumToRun, 1,1), d_locmemFlt, d_cur_y, d_cur_z,
                                    d_obj_recon, self.d_obs, d_lp_off, 
                                    np.int32(N), np.int32(D), np.int32(K), np.int32(numPrevRun),
                                    np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))
                self.prg.calc_y_lp_on(self.queue, (curNumToRun,K,D), (curNumToRun, 1,1), d_locmemFlt, d_cur_y, d_cur_z,
                                    d_obj_recon, self.d_obs, d_lp_on,
                                    np.int32(N), np.int32(D), np.int32(K), np.int32(numPrevRun),
                                    np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

                # self.prg.calc_y_lps_noLcl(self.queue, (curNumToRun,K,D), (curNumToRun, 1,1), d_locmemFlt, d_cur_y, d_cur_z,
                #                     d_obj_recon, self.d_obs, d_lp_off2, d_lp_on2,
                #                     np.int32(N), np.int32(D), np.int32(K), np.int32(numPrevRun),
                #                     np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))
                numToRun -= curNumToRun
                numPrevRun += curNumToRun



            # check_y_lps_on = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                                            dtype = np.float32, requirements=['C','A'])
            # check_y_lps_off = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                                            dtype = np.float32, requirements=['C','A'])
            # cl.enqueue_copy(self.queue,check_y_lps_off, d_lp_off)
            # cl.enqueue_copy(self.queue,check_y_lps_on, d_lp_on)
            # #
            # check_y_lps_on2 = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                                           dtype = np.float32, requirements=['C','A'])
            # check_y_lps_off2 = np.require(np.zeros(shape = (K, D), dtype=np.float32),
            #                                           dtype = np.float32, requirements=['C','A'])
            # cl.enqueue_copy(self.queue,check_y_lps_off2, d_lp_off2)
            # cl.enqueue_copy(self.queue,check_y_lps_on2, d_lp_on2)
            self.prg.sample_y_pre_calc(self.queue, (K,D, 1), None, d_cur_y, d_lp_off, d_lp_on,
                                       d_rand, np.int32(K), np.int32(D))
        return d_cur_y

        # self.prg.sample_yJLA(self.queue, (N,K,D), (N, 1,1), d_locmemFlt, d_cur_y, d_cur_z,
        #                      d_z_by_y, self.d_obs, d_rand, np.int32(N), np.int32(D), np.int32(K),
        #                      np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        # wgx = gcd(cur_y.shape[0], self.p_mul_sample_y)
        # wgy = gcd(cur_y.shape[1], self.p_mul_sample_y)
        
        # # calculate the prior probability that a pixel is on
        # self.prg.sample_y(self.queue, cur_y.shape, (wgx, wgy),
        #                   d_cur_y, d_cur_z, d_z_by_y, self.d_obs, d_rand, 
        #                   np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
        #                   np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        # cl.enqueue_copy(self.queue, cur_y, d_cur_y)
        # return cur_y

    def _cl_infer_z(self, d_cur_y, d_cur_z):
        """Infer feature ownership
        """
        #d_cur_y = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_y.astype(np.int32))
        #d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = cur_z.astype(np.int32))

        N = self.obs.shape[0]
        D = self.obs.shape[1]
        K = self.k

        if K > 0:
            # d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                         hostbuf = np.require(np.zeros(shape = self.obs.shape, dtype=np.int32),
            #                                              dtype = np.int32, requirements=['C','A']))
            d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(N*D*np.dtype('int32').itemsize))

            self.prg.init_bufferInt(self.queue, (N*D,1), None, d_obj_recon, np.int32(N*D), np.int32(0))

            #d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=(N*D*np.dtype('int32').itemsize))

            d_z_col_sum = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=(K*np.dtype('int32').itemsize))        
            self.prg.init_bufferInt(self.queue, (K,1), None, d_z_col_sum, np.int32(K), np.int32(0))
            self.prg.compute_recon_objs_andzsums(self.queue, (N, K, D), None,
                                                 d_cur_y, d_cur_z, d_obj_recon, d_z_col_sum,
                                                 np.int32(N), np.int32(D), np.int32(K))


            # to_get_size = np.require(np.zeros(shape = (N,K,D), dtype=np.float32),
            #                          dtype=np.float32, requirements=['C','A'])
            #d_lp_off = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = to_get_size.nbytes)
            #d_lp_on = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = to_get_size.nbytes)
            # d_lp_off = cl.Buffer(self.ctx, self.mf.READ_WRITE |self.mf.COPY_HOST_PTR, hostbuf=to_get_size)
            # d_lp_on = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=to_get_size)

            d_lp_off = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=(N*K*D*np.dtype('float32').itemsize))
            d_lp_on = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=(N*K*D*np.dtype('float32').itemsize))

            self.prg.init_2bufferFlt(self.queue, (N*K*D,1), None, d_lp_off,d_lp_on, np.int32(N*K*D), np.float32(0.))

            # lpoffdebug = np.require(np.zeros(shape = (N,K,D), dtype=np.float32),
            #                                           dtype = np.float32, requirements=['C','A'])
            # lpondebug = np.require(np.zeros(shape = (N,K,D), dtype=np.float32),
            #                                           dtype = np.float32, requirements=['C','A'])
            maxWorkGroupSize = np.int32(self.device.max_work_group_size/2)
            maxLocalMem = self.device.local_mem_size

            workGroupSize = int(min(maxLocalMem//np.dtype(np.float32).itemsize, maxWorkGroupSize))
            tmpLocMemNPFlt = np.empty(workGroupSize,dtype=np.float32)
            d_locmemFlt = cl.LocalMemory(tmpLocMemNPFlt.nbytes)

            curNumToRun = min(workGroupSize, N)
            numToRun = N
            numPrevRun = 0
            #TODO: this could be made faster... 
            while numToRun > 0:
                self.prg.calc_z_lps(self.queue, (curNumToRun,K,D), (curNumToRun, 1,1), d_locmemFlt, d_cur_y, d_cur_z,
                                    d_obj_recon, d_z_col_sum, self.d_obs, d_lp_off, d_lp_on,
                                    np.int32(N), np.int32(D), np.int32(K), np.int32(numPrevRun),
                                    np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))
                numToRun -= curNumToRun
                numPrevRun += curNumToRun

            # lp_off = np.zeros(shape=(N,K, D), dtype=np.float32)
            # lp_on = np.zeros(shape=(N,K, D), dtype=np.float32)
            # cl.enqueue_copy(self.queue, lp_off, d_lp_off)
            # cl.enqueue_copy(self.queue,lp_on,d_lp_on)

            # cl.enqueue_copy(self.queue, lpoffdebug, d_lp_off)
            # cl.enqueue_copy(self.queue, lpondebug, d_lp_on)


            #TODO: rewrite for float4s
            workGroupSize = int(min(maxLocalMem//np.dtype(np.float32).itemsize, maxWorkGroupSize))
            tmpLocMemNPFlt = np.empty(workGroupSize,dtype=np.float32)
            d_locmemFlt = cl.LocalMemory(tmpLocMemNPFlt.nbytes)

            curNumToReduce = max(D, workGroupSize)

            #this is due to the weird pyopencl/intel cpu bug where # of workgroups must be a multiple of workgroup size
            #no still doesn't work
            #num_d_workers = int(math.ceil(D/workGroupSize) * workGroupSize)

            num_d_workers = workGroupSize
            #
            #

            num_loop = int(math.ceil(D/workGroupSize))
            d_lp_off_tmp = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(N*K*D*np.dtype('float32').itemsize))
            
            d_lp_on_tmp = cl.Buffer(self.ctx, self.mf.READ_WRITE, size=(N*K*D*np.dtype('float32').itemsize))
            cl.enqueue_copy(self.queue, d_lp_off_tmp, d_lp_off)
            cl.enqueue_copy(self.queue,d_lp_on_tmp, d_lp_on)


            # self.prg.init_2bufferFlt(self.queue, (N*K*D,1), None,
            #                          d_lp_off_tmp,d_lp_on_tmp, np.int32(N*K*D), np.float32(0.))

            # lpoffdebug2 = np.require(np.zeros(shape = (N,K,D), dtype=np.float32),
            #                                            dtype = np.float32, requirements=['C','A'])
            # lpondebug2 = np.require(np.zeros(shape = (N,K,D), dtype=np.float32),
            #                                            dtype = np.float32, requirements=['C','A'])
            # lpoffdebug3 = np.require(np.zeros(shape = (N,K,D), dtype=np.float32),
            #                                           dtype = np.float32, requirements=['C','A'])
            # lpondebug3 = np.require(np.zeros(shape = (N,K,D), dtype=np.float32),
            #                                           dtype = np.float32, requirements=['C','A'])

            # # cl.enqueue_copy(self.queue, lpoffdebug2, d_lp_off_tmp)
            # # cl.enqueue_copy(self.queue, lpondebug2, d_lp_on_tmp)

            while (curNumToReduce > workGroupSize):
                 for i in np.arange(num_loop):
                     self.prg.reduc_vec_sum3d(self.queue, (num_d_workers,N,K),
                                      (workGroupSize,1,1),
                                      d_locmemFlt, d_lp_off, d_lp_off_tmp,
                                      np.int32(N), np.int32(K), np.int32(D), np.int32(i*num_d_workers))
                     self.prg.reduc_vec_sum3d(self.queue, (num_d_workers,N,K),
                                      (workGroupSize,1,1),
                                      d_locmemFlt, d_lp_on, d_lp_on_tmp,
                                      np.int32(N), np.int32(K), np.int32(D), np.int32(i*num_d_workers))

                 curNumToReduce = curNumToReduce // workGroupSize
                 if curNumToReduce > workGroupSize:
                     d_Old_lp_off_tmp = d_lp_off_tmp
                     d_lp_off_tmp = d_lp_off
                     d_lp_off = d_Old_lp_off_tmp

                     d_Old_lp_on_tmp = d_lp_on_tmp
                     d_lp_on_tmp = d_lp_on
                     d_lp_on = d_Old_lp_on_tmp

                 # cl.enqueue_copy(self.queue, lpoffdebug3, d_lp_off)
                 # cl.enqueue_copy(self.queue, lpoffdebug2, d_lp_off_tmp)
                 # cl.enqueue_copy(self.queue, lpondebug3, d_lp_on)
                 # cl.enqueue_copy(self.queue, lpondebug2, d_lp_on_tmp)


          # #  quit()
          #   print("starting complete kernel")
            #cl.enqueue_copy(self.queue, lpoffdebug, d_lp_off)
            #cl.enqueue_copy(self.queue, lpondebug, d_lp_on)


            #d_lp_off_sum = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                    hostbuf = np.require(np.zeros(shape = (N,K,D), dtype=np.float32),
            #                                         dtype = np.float32, requirements=['C','A']))
            #d_lp_on_sum = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                    hostbuf = np.require(np.zeros(shape = (N,K,D), dtype=np.float32),
            #                                         dtype = np.float32, requirements=['C','A']))

            self.prg.finish_reduc_vec_sum3d(self.queue, (workGroupSize,N,K),
                                             (workGroupSize,1,1),
                                             d_locmemFlt, d_lp_off_tmp, d_lp_off,
                                             np.int32(N), np.int32(K), np.int32(D),
                                             np.int32(min(D,curNumToReduce)))
            self.prg.finish_reduc_vec_sum3d(self.queue, (workGroupSize,N,K),
                                           (workGroupSize,1,1),
                                            d_locmemFlt, d_lp_on_tmp, d_lp_on,
                                            np.int32(N), np.int32(K), np.int32(D),
                                            np.int32(min(D,curNumToReduce)))

            # cl.enqueue_copy(self.queue, lpoffdebug3, d_lp_off)
            # cl.enqueue_copy(self.queue, lpondebug3, d_lp_on)


            d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf=np.random.random(size = (N,K)).astype(np.float32))
            #now d=0 of lp_off and lp_on for all nk has the answer
            self.prg.sample_z_pre_calc(self.queue, (N, K), None, d_cur_z, d_lp_off, d_lp_on,
                                           d_rand, np.int32(N), np.int32(K), np.int32(D), np.float32(self.T))

        return d_cur_z

    def _cl_infer_k_new(self, cur_y, cur_z):

        # sample new features use importance sampling
        k_new = self._sample_k_new(cur_y, cur_z)
        if k_new:
            cur_y, cur_z = k_new

        # delete empty feature images
        non_empty_feat_img = np.where(cur_y.sum(axis = 1) > 0)
        cur_y = cur_y[non_empty_feat_img[0],:].astype(np.int32)
        cur_z = cur_z[:,non_empty_feat_img[0]].astype(np.int32)
            
        # delete null features
        inactive_feat_col = np.where(cur_z.sum(axis = 0) == 0)
        cur_z = np.delete(cur_z, inactive_feat_col[0], axis=1).astype(np.int32)
        cur_y = np.delete(cur_y, inactive_feat_col[0], axis=0).astype(np.int32)
            
        z_s0, z_s1 = cur_z.shape
        cur_z = cur_z.reshape((z_s0 * z_s1, 1))
        cur_z = cur_z.reshape((z_s0, z_s1))

        y_s0, y_s1 = cur_y.shape
        cur_y = cur_y.reshape((y_s0 * y_s1, 1))
        cur_y = cur_y.reshape((y_s0, y_s1))

        # update self.k
        self.k = cur_z.shape[1]
        
        return cur_y, cur_z

    def _cl_infer_k_newJLA(self, cur_y, cur_z):
        """Written by Joe Austerweil on 07/20/15
            Infer new features (bounded by self.k_max_new). Cleans (removes unused features) first
        :param cur_y: feature images on host
        :param cur_z: feature ownership on host
        :return: cur_z, cur_y"""
        a_time = time()
        
        # delete empty feature images
        non_empty_feat_img = np.where(cur_y.sum(axis = 1) > 0)
        cur_y = cur_y[non_empty_feat_img[0],:].astype(np.int32)
        cur_z = cur_z[:,non_empty_feat_img[0]].astype(np.int32)

        # delete null features
        active_feat_col = np.where(cur_z.sum(axis = 0) > 0)
        cur_z = cur_z[:,active_feat_col[0]].astype(np.int32)
        cur_y = cur_y[active_feat_col[0],:].astype(np.int32)

        # update self.k
        self.k = cur_z.shape[1]

        z_s0, z_s1 = cur_z.shape
        cur_z = cur_z.reshape((z_s0 * z_s1, 1))
        cur_z = np.require(cur_z.reshape((z_s0, z_s1)), dtype=np.int32, requirements=['C','A'])

        y_s0, y_s1 = cur_y.shape
        cur_y = cur_y.reshape((y_s0 * y_s1, 1))
        cur_y = np.require(cur_y.reshape((y_s0, y_s1)), dtype=np.int32, requirements=['C','A'])


        N = self.obs.shape[0]
        K = cur_z.shape[1]
        D = self.obs.shape[1]


        #print("x:")
        #print(self.obs)
        #print("cur z:")
        #print(cur_z)
        #print("cur_y:")
        #print(cur_y)
        obj_recon = None
        #sample new features. 1st recon!

        # d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
        #                         hostbuf = np.require(np.zeros(shape = self.obs.shape, dtype=np.int32),
        #                                              dtype = np.int32, requirements=['C','A']))

        d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(N*D*np.dtype('int32').itemsize))

        self.prg.init_bufferInt(self.queue, (N*D,1), None, d_obj_recon, np.int32(N*D), np.int32(0))

        if cur_z.size is 0:
            obj_recon = np.zeros(self.obs.shape)
        else:
            
            d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf=cur_z.astype(np.int32))
            d_cur_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf = cur_y.astype(np.int32))
            
            self.prg.compute_recon_objs(self.queue, (N, K, D), None,
                                    d_cur_y, d_cur_z, d_obj_recon,
                                    np.int32(N), np.int32(D), np.int32(K))
       
        
            # obj_recon = np.require(np.zeros(shape = self.obs.shape, dtype=np.int32),
            #                    dtype = np.int32, requirements=['C','A'])
            #cl.enqueue_copy(self.queue, obj_recon,d_obj_recon)
        #print("Current Object Reconstruction:")
        #print(obj_recon)

        #TODO: gpuify lpx calc
        lps = np.require(np.zeros((N,D,self.k_max_new+1), dtype=np.float32), dtype=np.float32, requirements=['C','A'])
        d_lps = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                          hostbuf=lps.astype(np.float32))
        #TODO: localify k_max_new to share some values
        self.prg.calc_lp_fornew(self.queue, (N, D, self.k_max_new+1), None,
                                d_obj_recon, self.d_obs, d_lps,
                                np.int32(N), np.int32(D), np.int32(K), np.int32(self.k_max_new+1),
                                np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        cl.enqueue_copy(self.queue, lps, d_lps)
        #print("LPs: ")
        #print(lps)
        #quit()
        KNewRng =  np.arange(K, self.k_max_new+1+K)
        #print("KNewRng")
        #print(KNewRng)
        #kNewLPs = np.sum(lps,axis=1) - np.matlib.repmat(self.alpha/N + KNewRng.T * np.log(self.alpha/N) - gammaln(KNewRng.T+1),N,1)
        kNewLPs = np.sum(lps,axis=1) - self.alpha/N + KNewRng.T * np.log(self.alpha/N) - gammaln(KNewRng+1)
        #kNewLPs = self.alpha/N + KNewRng.T * np.log(self.alpha/N) - gammaln(KNewRng+1)
        #print("kNewLPs")
        #print(kNewLPs)
        #quit()
        #kNewLPs = np.sum(kNewLPs,axis=1)
        #print(kNewLPs)
        #print(kNewLPs.shape)
        #quit()
        logmaxs = np.amax(kNewLPs,axis=1)
       # print("logmaxs:")
       # print(logmaxs)
        logmaxs = logmaxs[:,np.newaxis]
        pdfs = np.exp(kNewLPs-logmaxs)
        #print("pdfs:")
        #print(pdfs)
        #quit()
        sums = np.sum(pdfs, axis=1)
        sums = sums[:,np.newaxis]
        #sums = np.expand_dims(sums, axis=1)
        pdfs = np.divide(pdfs, sums)
        #pdfs = np.divide(np.rollaxis(pdfs,2), sums)
        #pdfs = np.swapaxes(np.swapaxes(pdfs, 2, 0),1,0)
        #print(pdfs)
        #print(pdfs.shape)
        
        knews = np.apply_along_axis(lambda x: np.random.choice(KNewRng-K,p=x), axis=1, arr=pdfs)
        #knews = np.require(knews,dtype=np.int32, requirements=['C', 'A'])
        #print("knews")
        #print(knews)
        #print(knews.shape)
        #quit()
        partSum = 0
        totNewK = 0
        
        totNewK = min(np.sum(knews), self.k_max_new)
        samps = np.random.choice(np.arange(N), p=knews/np.sum(knews), size=totNewK)

        knews = np.zeros(N)
        for n in np.arange(totNewK):
            knews[samps[n]]+=1


        #        for i in np.arange(knews.size):
 #           if (partSum +knews[i]) > self.k_max_new: 
 #               knews[i] = max(self.k_max_new-partSum,0)
 #           partSum += knews[i]
        
        #print("knews")
        #print(knews)
        #totNewK = 1
        #knews = np.array([1, 0, 0, 0, 0, 0, 0, 0])
        
        new_z = np.zeros(shape=(N,totNewK),dtype=np.int32)
        for k_ind in np.arange(totNewK):
            new_z[:,k_ind] = knews > k_ind

        #print("new z")
        #print(new_z)
       # print("cur_z before comb:")
       # print(cur_z)
        #update z,y,r to prep for creation
        cur_z = np.require(np.hstack((cur_z, new_z)), dtype=np.int32, requirements=['C','A'])

        #print("cur_z after comb:")
        #print(cur_z)
 
        #print("cur_y pre comb:")
        #print(cur_y)
        cur_y = np.require(np.vstack((cur_y, np.zeros(shape=(totNewK,D), dtype=np.int32))),
                           dtype=np.int32, requirements=['C', 'A'])
        #print("cur_y post comb:")
        #print(cur_y)
        oldK = K
        K += totNewK
        #print("oldK %d, K %d, totNewK %d" %(oldK, K, totNewK))
        d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf = cur_z.astype(np.int32))
        d_cur_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf = cur_y.astype(np.int32))
        d_knews = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                  hostbuf = knews.astype(np.int32))

        comb_vec = np.require(comb(totNewK, np.arange(totNewK+1)),
                              dtype=np.int32, requirements=['C','A'])
       # print("comb_vec:")
      #  print(comb_vec)
        #quit()
        d_comb_vec = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                               hostbuf=comb_vec.astype(np.int32))
        new_y_val_probs = np.require(np.empty(shape=(totNewK+1,N,D), dtype=np.float32),
                                     dtype=np.float32, requirements=['C','A'])
        d_new_y_val_probs = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                      hostbuf = new_y_val_probs.astype(np.float32))
        # call gpu-based function to create the new features...
        maxWorkGroupSize = np.int32(self.device.max_work_group_size/2)
        maxLocalMem = self.device.local_mem_size

        workGroupSize = int(min(maxLocalMem//np.dtype(np.float32).itemsize, maxWorkGroupSize))
        tmpLocMemNPInt = np.empty(workGroupSize,dtype=np.int32)
        d_locmemInt = cl.LocalMemory(tmpLocMemNPInt.nbytes)
        curNumToRun = min(D, workGroupSize)
        num_d_workers = workGroupSize

        num_loop = int(math.ceil(D/workGroupSize))
        for i in np.arange(num_loop):
            self.prg.new_y_val_probs(self.queue, (num_d_workers, N, totNewK+1), (num_d_workers, 1,1), d_locmemInt,
                                     d_cur_z, d_cur_y,  d_comb_vec, d_obj_recon, self.d_obs,
                                     d_new_y_val_probs, np.int32(N), np.int32(K), np.int32(D), np.int32(totNewK+1),
                                     np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))
        
        cl.enqueue_copy(self.queue, new_y_val_probs, d_new_y_val_probs) 
        # print("new_y_val_probs: ")
       #  print(new_y_val_probs)
       #  print(new_y_val_probs.shape)
       # #
        #norms = np.sum(new_y_val_probs, axis=0)
        #print(new_y_val_probs/norms)
        #quit()

        #new_y_on = np.require(np.zeros(shape=(N,D),dtype=np.int32),
        #                      dtype=np.int32, requirements=['C','A'])
        #d_new_y_on = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
        #                       hostbuf=new_y_on.astype(np.int32))

        y_rand_vals = np.require(np.random.random((N,D)), dtype=np.float32,
                                 requirements=['C', 'A'])
        d_y_rand_vals = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                  hostbuf=y_rand_vals.astype(np.float32))
        tmpLocMemNPFlt = np.empty(workGroupSize,dtype=np.float32)
        d_locmemFlt = cl.LocalMemory(tmpLocMemNPFlt.nbytes)

        self.prg.y_new_samp(self.queue, (totNewK+1,N,D), (totNewK+1,1,1), d_locmemFlt,
                            d_new_y_val_probs,  d_cur_y, d_knews, d_y_rand_vals,
                            np.int32(totNewK+1), np.int32(N), np.int32(D), np.int32(oldK))

        #cl.enqueue_copy(self.queue, new_y_on,d_new_y_on)
        cl.enqueue_copy(self.queue, cur_y, d_cur_y)
        #print("new_y_on:")
        #print(new_y_on)
        #print(new_y_on.shape)
        
        #print("cur_y:")
        #print(cur_y)
        #print(cur_y.shape)

        if totNewK > 0:
            cur_z = self._cl_infer_z(cur_y, cur_z)
        
       # quit()
        # new_ys = cur_y[oldK:,:]
        # possKNew = np.arange(totNewK+1)
        # #perhaps faster with just new_ys[oldK:,:] = new_y_on? i don't think that'd be right...
        #curNewI = 0
        #TODO: FIX THIS
#        for i in np.arange(N):
#            numNew = knews[i]
#            possNew = np.arange(numNew+1)
#            newInds = possNew > numNew
#            poss_new_y = cur_y[(i+oldK):(i+oldK+numNew-1),:]
#            poss_new_y[newInds,:] = (np.atleast_2d(new_y_on[i,:]).T > np.atleast_2d(newInds)).T
        
        # Delete empty feature images
        non_empty_feat_img = np.where(cur_y.sum(axis = 1) > 0)
        cur_y = cur_y[non_empty_feat_img[0],:].astype(np.int32)
        cur_z = cur_z[:,non_empty_feat_img[0]].astype(np.int32)
        
        #print("new sampled cur_y: ")
        #print(cur_y)
       # print("new sampled z: ")
       # print(cur_z)
       # print("cur_y.shape:%d, %d, cur_z.shape %d, %d" %(cur_y.shape[0], cur_y.shape[1], cur_z.shape[0], cur_z.shape[1]))
       # print(np.dot(cur_z,cur_y))
    
        self.gpu_time += time() - a_time
        return cur_y, cur_z
    
    def _cl_infer_k_newJLA2(self, cur_y, cur_z):
        """Written by Joe Austerweil on 07/20/15
            Infer new features (bounded by self.k_max_new). Cleans (removes unused features) first.
            Only one object can get new features per run
        :param cur_y: feature images on host
        :param cur_z: feature ownership on host
        :return: cur_z, cur_y"""
        a_time = time()

        startK = self.k
        # delete empty feature images
        non_empty_feat_img = np.where(cur_y.sum(axis = 1) > 0)
        cur_y = cur_y[non_empty_feat_img[0],:].astype(np.int32)
        cur_z = cur_z[:,non_empty_feat_img[0]].astype(np.int32)

        # delete null features
        active_feat_col = np.where(cur_z.sum(axis = 0) > 0)
        cur_z = cur_z[:,active_feat_col[0]].astype(np.int32)
        cur_y = cur_y[active_feat_col[0],:].astype(np.int32)

        z_s0, z_s1 = cur_z.shape
        cur_z = cur_z.reshape((z_s0 * z_s1, 1))
        cur_z = np.require(cur_z.reshape((z_s0, z_s1)), dtype=np.int32, requirements=['C','A'])

        y_s0, y_s1 = cur_y.shape
        cur_y = cur_y.reshape((y_s0 * y_s1, 1))
        cur_y = np.require(cur_y.reshape((y_s0, y_s1)), dtype=np.int32, requirements=['C','A'])

        # update self.k
        self.k = cur_z.shape[1]

        N = self.obs.shape[0]
        K = cur_z.shape[1]
        D = self.obs.shape[1]

        obj_recon = None
        #sample new features. 1st recon!

        # d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
        #                         hostbuf = np.require(np.zeros(shape = self.obs.shape, dtype=np.int32),
        #                                              dtype = np.int32, requirements=['C','A']))
        d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(N*D*np.dtype('int32').itemsize))

        self.prg.init_bufferInt(self.queue, (N*D,1), None, d_obj_recon, np.int32(N*D), np.int32(0))

        if cur_z.size is 0:
            obj_recon = np.zeros(self.obs.shape)
        else:
            d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf=cur_z.astype(np.int32))
            d_cur_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf = cur_y.astype(np.int32))
            
            self.prg.compute_recon_objs(self.queue, (N, K, D), None,
                                    d_cur_y, d_cur_z, d_obj_recon,
                                    np.int32(N), np.int32(D), np.int32(K))
              
            # obj_recon = np.require(np.zeros(shape = self.obs.shape, dtype=np.int32),
            #                    dtype = np.int32, requirements=['C','A'])
            # cl.enqueue_copy(self.queue, obj_recon,d_obj_recon)
        
        lps = np.require(np.empty((N,D,self.k_max_new+1), dtype=np.float32),
                         dtype=np.float32, requirements=['C','A'])
        d_lps = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(N*D*(self.k_max_new+1)*np.dtype('float32').itemsize))
        # self.prg.init_bufferFlt(self.queue, (N*D*(self.k_max_new+1),1), None, d_lps,
        #                         np.int32(N*D*(self.k_max_new+1)), np.float32(0.))
        #                  hostbuf=lps.astype(np.float32))
        #TODO: localify k_max_new to share some values
        self.prg.calc_lp_fornew(self.queue, (N, D, self.k_max_new+1), None,
                                d_obj_recon, self.d_obs, d_lps,
                                np.int32(N), np.int32(D), np.int32(K), np.int32(self.k_max_new+1),
                                np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        cl.enqueue_copy(self.queue, lps, d_lps)
        #TODO: GPUify these parts
        KNewRng =  np.arange(K, self.k_max_new+1+K)
        kNewLPs = np.sum(lps,axis=1) - self.alpha/N + KNewRng.T * np.log(self.alpha/N) - gammaln(KNewRng+1)
        logmaxs = np.amax(kNewLPs,axis=1)
        logmaxs = logmaxs[:,np.newaxis]
        pdfs = np.exp(kNewLPs-logmaxs)
        sums = np.sum(pdfs, axis=1)
        sums = sums[:,np.newaxis]
        pdfs = np.divide(pdfs, sums)
        
        knews = np.apply_along_axis(lambda x: np.random.choice(KNewRng-K,p=x), axis=1, arr=pdfs)
        partSum = 0

        if np.nonzero(knews)[0].shape[0] > 0:
            new_k_ind = np.random.choice(np.arange(N), size=1, p= knews/np.sum(knews))[0]
#            new_k_ind = new_k_ind[0]
            new_k = min(knews[new_k_ind], self.k_max_new)
            
            new_z = np.zeros(shape=(N,new_k),dtype=np.int32)
            new_z[new_k_ind,:] = 1
            
            cur_z = np.require(np.hstack((cur_z, new_z)), dtype=np.int32, requirements=['C','A'])

            cur_y = np.require(np.vstack((cur_y, np.zeros(shape=(new_k,D), dtype=np.int32))),
                               dtype=np.int32, requirements=['C', 'A'])

            oldK = K
            K += new_k

            self.k = K
            d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                hostbuf = cur_z.astype(np.int32))
            d_cur_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                hostbuf = cur_y.astype(np.int32))
            # d_knews = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                     hostbuf = knews.astype(np.int32))
            #for ensuring z & y are proper size later, but only reallocating as necessary
            self.tmp_z = cur_z
            self.tmp_y = cur_y
            comb_vec = np.require(comb(new_k, np.arange(new_k+1)),
                                  dtype=np.int32, requirements=['C','A'])
            d_comb_vec = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf=comb_vec.astype(np.int32))
            # new_y_val_probs = np.require(np.empty(shape=(new_k+1,D), dtype=np.float32),
            #                              dtype=np.float32, requirements=['C','A'])
            d_new_y_val_probs = cl.Buffer(self.ctx, self.mf.READ_WRITE,
                                          size=((new_k+1)*D*np.dtype('float32').itemsize))
            #hostbuf = new_y_val_probs.astype(np.float32))

            # call gpu-based function to create the new features...
            maxWorkGroupSize = np.int32(self.device.max_work_group_size/2)
            maxLocalMem = self.device.local_mem_size

            workGroupSize = int(min(maxLocalMem//np.dtype(np.float32).itemsize, maxWorkGroupSize))
            tmpLocMemNPInt = np.empty(workGroupSize,dtype=np.int32)
            d_locmemInt = cl.LocalMemory(tmpLocMemNPInt.nbytes)
            num_d_workers = min(workGroupSize,D)
            self.prg.new_y_val_probs2(self.queue, (num_d_workers, int(math.ceil((1.*D)/num_d_workers)), new_k+1),
                                      (num_d_workers, 1,1), d_locmemInt,
                                      d_cur_z, d_cur_y,  d_comb_vec, d_obj_recon, self.d_obs,
                                      d_new_y_val_probs, np.int32(new_k_ind), np.int32(new_k),
                                      np.int32(N), np.int32(D), np.int32(K),
                                      np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))
        

            #cl.enqueue_copy(self.queue, new_y_val_probs, d_new_y_val_probs)

            y_rand_vals = np.require(np.random.random((D)), dtype=np.float32,
                                 requirements=['C', 'A'])
            d_y_rand_vals = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                      hostbuf=y_rand_vals.astype(np.float32))
            tmpLocMemNPFlt = np.empty(workGroupSize,dtype=np.float32)
            d_locmemFlt = cl.LocalMemory(tmpLocMemNPFlt.nbytes)

            self.prg.y_new_samp2(self.queue, (new_k+1,1,D), (new_k+1,1,1), d_locmemFlt,
                                 d_new_y_val_probs,  d_cur_y,  d_y_rand_vals,
                                np.int32(new_k_ind), np.int32(new_k),
                                np.int32(N), np.int32(D), np.int32(oldK))
            
            #cl.enqueue_copy(self.queue, cur_y, d_cur_y)
            #
            # print("cur_y:")
            # print(cur_y)
            # print(cur_y.shape)
            #
            # cl.enqueue_copy(self.queue, cur_z, d_cur_z)
            #
            # print("cur_z:")
            # print(cur_z)
            # print(cur_z.shape)            #print("inferring zs for new features: ")

            d_cur_z = self._cl_infer_z(d_cur_y, d_cur_z)
        
        
            # # Delete empty feature images
            # non_empty_feat_img = np.where(cur_y.sum(axis = 1) > 0)
            # cur_y = cur_y[non_empty_feat_img[0],:].astype(np.int32)
            # cur_z = cur_z[:,non_empty_feat_img[0]].astype(np.int32)
        
        if self.k is not startK:
            self.tmp_z = np.require(np.zeros(shape=(N,self.k), dtype=np.int32),
                                             dtype=np.int32, requirements=['C','A'])
            self.tmp_y = np.require(np.zeros(shape=(self.k,D), dtype=np.int32),
                                             dtype=np.int32, requirements=['C','A'])

        self.gpu_time += time() - a_time

        return d_cur_y, d_cur_z

    def _cl_split_merge_anneal_step(self, d_cur_y, d_cur_z):
        # if not self.sim_annealP:
        #     print("warning: implemented split-merge .")
        N = self.obs.shape[0]
        D = self.obs.shape[1]
        K = self.k
        samp = np.random.rand(1)
        if K > 1:
            cur_y = np.require(np.zeros(shape=(K,D),dtype=np.int32), dtype=np.int32,
                               requirements=['C', 'A'])
            cur_z = np.require(np.zeros(shape=(N,K),dtype=np.int32), dtype=np.int32,
                               requirements=['C', 'A'])
            cl.enqueue_copy(self.queue, cur_y, d_cur_y)
            cl.enqueue_copy(self.queue, cur_z, d_cur_z)
            ms = np.sum(cur_z,0)
            lp_cur = self._logprob((d_cur_y,d_cur_z))
            p_ms = ms / np.sum(ms)
            if samp[0] < self.splitProb:
                #split
                k_ind = np.random.choice(np.arange(K),size=1,p=p_ms)
                nks = cur_z[:,k_ind]
                poss_ns = np.where(nks==1)
                pick2 = np.random.permutation(poss_ns)
                n1 = pick2[0]
                n2 = pick2[1]
                #form a split proposal give it n2
                z_samp = np.copy(cur_z)
                y_samp = np.copy(cur_y)
                z_zeros = np.require(np.zeros(shape=(N,1),dtype=np.int32), dtype=np.int32,
                                   requirements=['C', 'A'])
                y_zeros = np.require(np.zeros(shape=(1,D),dtype=np.int32), dtype=np.int32,
                                   requirements=['C', 'A'])
                #print(z_samp.shape)
                z_samp = np.concatenate((z_samp, z_zeros),axis=1)
                y_samp = np.concatenate((y_samp, y_zeros),axis=0)
                #print(z_samp.shape)
                z_samp[n2,K] = 1
                z_samp[n2,k_ind] = 0
                z_samp[pick2[2:],k_ind] = 0
                d_samp_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                          hostbuf = z_samp.astype(np.int32))
                d_samp_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                          hostbuf = y_samp.astype(np.int32))
                self.k = K+1 #remember to set k back to K if we reject
                if self.N <= 200:
                    d_samp_y = self._cl_infer_y(d_samp_y, d_samp_z)
                else:
                    d_samp_y = self._cl_infer_y2(d_samp_y, d_samp_z)
                d_samp_z = self._cl_infer_z(d_samp_y, d_samp_z)
                #cl.enqueue_copy(self.queue, z_samp, d_samp_z)
                #cl.enqueue_copy(self.queue, y_samp, d_samp_y)
                self.tmp_y = np.require(np.zeros(shape=(self.k,D),dtype=np.int32), dtype=np.int32,
                                        requirements=['C', 'A'])
                self.tmp_z = np.require(np.zeros(shape=(N,self.k),dtype=np.int32), dtype=np.int32,
                                        requirements=['C', 'A'])
                lp_prop = self._logprob((d_samp_y,d_samp_z))

                expVal = self.T*self.splitAdj*(lp_prop - lp_cur)
                ap = None
                if expVal > 1:
                    ap = 1
                elif expVal < -100:
                    ap = 0
                else:
                    ap = min(1, math.exp(expVal))
                a_samp = np.random.rand(1)[0]
                if a_samp >= ap:
                    #reject
                    self.k = K
                    self.tmp_y = np.require(np.zeros(shape=(self.k,D),dtype=np.int32), dtype=np.int32,
                                            requirements=['C', 'A'])
                    self.tmp_z = np.require(np.zeros(shape=(N,self.k),dtype=np.int32), dtype=np.int32,
                                            requirements=['C', 'A'])

                else:
                    #accept
                    d_cur_y = d_samp_y
                    d_cur_z = d_samp_z
            elif K > 2:
                #merge proposal (figure out why K=2 has problems)
                #pick two features
                #print("trying to merge with K: %d" % K)
                k_inds = np.random.choice(np.arange(K),size=2,p=p_ms,replace=False)
                #form merged
                z_samp = np.copy(cur_z)
                y_samp = np.copy(cur_y)
                z_samp[:,k_inds[0]] = ((z_samp[:,k_inds[0]] + z_samp[:,k_inds[1]]) > 0) + 0
                #remove k_inds[1] and resample y
                z_mask = np.ones(K, dtype=bool)
                z_mask[k_inds[1]] = False
                z_samp = z_samp[:,z_mask]
                y_samp = y_samp[z_mask,:]
                self.k = K-1
                d_samp_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                          hostbuf = z_samp.astype(np.int32))
                d_samp_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                          hostbuf = y_samp.astype(np.int32))

                if self.N <= 200:
                    d_samp_y = self._cl_infer_y(d_samp_y, d_samp_z)
                else:
                    d_samp_y = self._cl_infer_y2(d_samp_y, d_samp_z)
                self.tmp_y = np.require(np.zeros(shape=(self.k,D),dtype=np.int32), dtype=np.int32,
                                        requirements=['C', 'A'])
                self.tmp_z = np.require(np.zeros(shape=(N,self.k),dtype=np.int32), dtype=np.int32,
                                        requirements=['C', 'A'])
                lp_prop = self._logprob((d_samp_y,d_samp_z))
                #lp_cur = self._logprob((d_cur_y,d_cur_z))
                expVal = self.T*self.splitAdj*(lp_prop - lp_cur)
                ap = None
                if expVal > 1:
                    ap = 1
                elif expVal < -100:
                    ap = 0
                else:
                    ap = min(1, math.exp(expVal))
                a_samp = np.random.rand(1)[0]

                if a_samp >= ap:
                    #reject
                    self.k = K
                    self.tmp_y = np.require(np.zeros(shape=(self.k,D),dtype=np.int32), dtype=np.int32,
                                            requirements=['C', 'A'])
                    self.tmp_z = np.require(np.zeros(shape=(N,self.k),dtype=np.int32), dtype=np.int32,
                                            requirements=['C', 'A'])
                else:
                    #accept
                    d_cur_y = d_samp_y
                    d_cur_z = d_samp_z
                    
        return d_cur_y, d_cur_z

    def convert_sample_to_host(self, sample):
        BaseSampler.convert_sample_to_host(self,sample)

        if self.cl_mode:
            d_cur_y, d_cur_z = sample
            N = self.obs.shape[0]
            K = self.k
            D = self.obs.shape[1]
            cur_y = np.require(np.zeros(shape = (K,D), dtype=np.int32),
                             dtype = np.int32, requirements=['C','A'])
            cur_z = np.require(np.zeros(shape = (N,K), dtype=np.int32),
                             dtype = np.int32, requirements=['C','A'])
            cl.enqueue_copy(self.queue, cur_y, d_cur_y)
            cl.enqueue_copy(self.queue, cur_z, d_cur_z)
            return (cur_y, cur_z)
        return sample

    def _logprob(self, sample):
        """Calculate the joint log probability of data and model given a sample.
        """
        log_prior = 0
        log_lik = 0
        #if cur_z.shape[1] == 0: return -99999999.9
        if self.k == 0: return -99999999.9
        if self.cl_mode:
            a_time = time()
            d_cur_y, d_cur_z = sample

            #for debugging on aug 14 2015
            #cl.enqueue_copy(self.queue, self.tmp_y, d_cur_y)
            #cl.enqueue_copy(self.queue, self.tmp_z, d_cur_z)

            N = self.obs.shape[0]
            D = self.obs.shape[1]
            K = self.k
            cur_z = np.require(np.zeros(shape = (N,K), dtype=np.int32),
                             dtype = np.int32, requirements=['C','A'])

            mks = np.require(np.zeros(shape = (K,1), dtype=np.int32),
                             dtype = np.int32, requirements=['C','A'])

            # d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                         hostbuf = np.require(np.zeros(shape = self.obs.shape, dtype=np.int32),
            #                                              dtype = np.int32, requirements=['C','A']))
            d_z_col_sum = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(K*np.dtype('int32').itemsize))

            #TODO: write 2buffer where buffers can have different sizes
            self.prg.init_bufferInt(self.queue, (K,1), None, d_z_col_sum, np.int32(K), np.int32(0))

            d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(N*D*np.dtype('int32').itemsize))

            self.prg.init_bufferInt(self.queue, (N*D,1), None, d_obj_recon, np.int32(N*D), np.int32(0))

            # d_z_col_sum = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                         hostbuf = mks)
            self.prg.compute_recon_objs_andzsums(self.queue, (N,K,D), None,
                                        d_cur_y, d_cur_z, d_obj_recon, d_z_col_sum,
                                        np.int32(N), np.int32(D), np.int32(K))

            # d_lps = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                         hostbuf = np.require(np.zeros(shape = (N, D), dtype=np.float32),
            #                                              dtype = np.float32, requirements=['C','A']))


            d_lps = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(N*D*np.dtype('float32').itemsize))
            # self.prg.init_bufferFlt(self.queue, (N*D,1), None, d_lps, np.int32(N*D), np.float32(0.))

            self.prg.calc_lps(self.queue, (N,D), None,
                              d_obj_recon, self.d_obs, d_lps,
                              np.int32(N), np.int32(D),
                              np.float32(self.lam), np.float32(self.epislon))

            #lps =  np.require(np.zeros(shape = (N, D), dtype=np.float32),
            #                  dtype = np.float32, requirements=['C','A'])
            #cl.enqueue_copy(self.queue,lps,d_lps)
            # call gpu-based function to create the new features...
            maxWorkGroupSize = np.int32(self.device.max_work_group_size/2)
            maxLocalMem = self.device.local_mem_size

            workGroupSize = int(min(maxLocalMem//np.dtype(np.float32).itemsize, maxWorkGroupSize))

            tmpLocMemNPFlt = np.empty(workGroupSize,dtype=np.float32)
            d_locmemFlt = cl.LocalMemory(tmpLocMemNPFlt.nbytes)
            num_d_workers = min(workGroupSize,N*D)

            curNumToReduce = N*D
            #leftOver = int(math.ceil(curNumToReduce/num_d_workers))
            d_sum_tmp = None
            if curNumToReduce > workGroupSize:
                # d_sum_tmp = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                #                       hostbuf = np.require(np.zeros(shape = (N, D), dtype=np.float32),
                #                                            dtype = np.float32, requirements=['C','A']))
                d_sum_tmp = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(N*D*np.dtype('float32').itemsize))
                self.prg.init_bufferFlt(self.queue, (N*D,1), None, d_sum_tmp, np.int32(N*D), np.float32(0.))

                while curNumToReduce > workGroupSize:
                    leftOver = int(math.ceil(curNumToReduce/num_d_workers))
                    self.prg.sum_reduc1d(self.queue, (num_d_workers,leftOver), (num_d_workers,1),
                                     d_locmemFlt, d_lps, d_sum_tmp, np.int32(curNumToReduce))

                    curNumToReduce = curNumToReduce // workGroupSize
                    if curNumToReduce > workGroupSize:
                        d_old_sum_tmp = d_sum_tmp
                        d_sum_tmp = d_lps
                        d_lps = d_old_sum_tmp

            lpX = np.require(np.zeros(shape = (1), dtype=np.float32),
                             dtype = np.float32, requirements=['C','A'])

            d_sum1d = cl.Buffer(self.ctx, self.mf.READ_WRITE, size = (np.dtype('float32').itemsize))

            if (N*D) > workGroupSize:
                self.prg.finish_sum_reduc1d(self.queue, (workGroupSize,1), (workGroupSize,1),
                                            d_locmemFlt, d_sum_tmp, d_sum1d, np.int32(curNumToReduce))
            else:
                self.prg.finish_sum_reduc1d(self.queue, (workGroupSize,1), (workGroupSize,1),
                                            d_locmemFlt, d_lps, d_sum1d, np.int32(curNumToReduce))

            cl.enqueue_copy(self.queue, mks, d_z_col_sum)
            cl.enqueue_copy(self.queue, lpX, d_sum1d)
            cl.enqueue_copy(self.queue, cur_z, d_cur_z)

            #for debugging
            # cur_y = np.require(np.zeros(shape = (K,D), dtype=np.int32),
            #                    dtype = np.int32, requirements=['C','A'])
            # obj_recon = np.require(np.zeros(shape = (K,D), dtype=np.int32),
            #                    dtype = np.int32, requirements=['C','A'])
            # cl.enqueue_copy(self.queue, cur_y, d_cur_y)
            # cl.enqueue_copy(self.queue,obj_recon,d_obj_recon)

            zT = cur_z.T
            _,cts = np.unique(zT.view(np.dtype((np.void, zT.dtype.itemsize *zT.shape[0]))), return_counts=True)
            mk_mask = np.where(mks > 0)
            lpZ = np.sum(gammaln(mks[mk_mask])+gammaln(N-mks[mk_mask]+1)-gammaln(N+1)) -np.sum(gammaln(cts))

            return lpZ + lpX[0]

            #d_logprob = cl.array.empty(self.queue, (cur_z.shape[0],), np.float32, allocator=self.mem_pool)

            # if cur_z.shape[0] % self.p_mul_logprob_z_data == 0: wg = (self.p_mul_logprob_z_data,)
            # else: wg = None
            #
            # self.prg.logprob_z_data(self.queue, (cur_z.shape[0],), wg,
            #                         d_cur_z, d_cur_y, self.d_obs, d_logprob.data, #d_novel_f.data,
            #                         np.int32(self.N), np.int32(cur_y.shape[1]), np.int32(cur_z.shape[1]),
            #                         np.float32(self.alpha), np.float32(self.lam), np.float32(self.epislon))
            # log_lik = d_logprob.get().sum()
            # self.gpu_time += time() - a_time
            #
            # # calculate the prior probability of Y
            # num_on = (cur_y == 1).sum()
            # num_off = (cur_y == 0).sum()
            # log_prior = num_on * np.log(self.theta) + num_off * np.log(1 - self.theta)

        else:
            cur_y, cur_z = sample
            # calculate the prior probability of Z
            for n in xrange(cur_z.shape[0]):
                num_novel = 0
                for k in xrange(cur_z.shape[1]):
                    m = cur_z[:n,k].sum()
                    if m > 0:
                        if cur_z[n,k] == 1: log_prior += np.log(m / (n+1.0))
                        else: log_prior += np.log(1 - m / (n + 1.0))
                    else: 
                        if cur_z[n,k] == 1: num_novel += 1
                if num_novel > 0:
                    log_prior += poisson.logpmf(num_novel, self.alpha / (n+1.0))
            # calculate the prior probability of Y
            #num_on = (cur_y == 1).sum()
            #num_off = (cur_y == 0).sum()
            #log_prior += num_on * np.log(self.theta) + num_off * np.log(1 - self.theta)
            # calculate the logliklihood
            log_lik = self._loglik(cur_y = cur_y, cur_z = cur_z)
        return log_prior + log_lik
            
    
class GibbsPredictor(BasePredictor):

    def __init__(self, cl_mode = True, cl_device = None,
                 alpha = 1.0, lam = 0.98, theta = 0.01, epislon = 0.02, init_k = 4):
        """Initialize the predictor.
        """
        BasePredictor.__init__(self, cl_mode = cl_mode, cl_device = cl_device)
        self.alpha = alpha
        self.lam = lam
        self.theta = theta
        self.epislon = epislon

    def read_test_csv(self, file_path, header=True):
        """Read the test cases and convert values to integer.
        """
        BasePredictor.read_test_csv(self, file_path, header)
        self.obs = np.array(self.obs, dtype=np.int32)
        return

    def read_samples_csv(self, var_name, file_path, header = True):
        """Read samples from a csv file.
        """
        BasePredictor.read_samples_csv(self, var_name, file_path, header)
        new_samples = []
        for sample in self.samples[var_name]:
            if len(sample) > 1: # remove null feature samples
                sample = np.array(sample, dtype=np.int32)
                sample = np.reshape(sample[1:], (-1, sample[0]))
                new_samples.append(sample)
        self.samples[var_name] = new_samples

    def predict(self, thining = 0, burnin = 0, use_iter=None, output_file = None):
        """Predict the test cases
        """
        assert('y' in self.samples and 'z' in self.samples)
        assert(len(self.samples['y']) == len(self.samples['z']))
        
        num_sample = len(self.samples['y'])
        num_obs = len(self.obs)
        logprob_result = np.empty((num_sample, num_obs))

        for i in xrange(num_sample):
            cur_y = self.samples['y'][i]
            cur_z = self.samples['z'][i]
            
            # generate all possible Zs
            num_feature = cur_z.shape[1]
            all_z = []
            for n in xrange(num_feature+1):
                base = [1] * n + [0] * (num_feature - n)
                all_z.extend(list(set(itertools.permutations(base))))
            all_z = np.array(all_z, dtype=np.int32)
            
            # BEGIN p(z|z_inferred) calculation

            # the following lines of code may be a bit tricky to parse
            # first, calculate the probability of features that already exist
            # since features are additive within an image, we can just prod them
            prior_off_prob = 1.0 - cur_z.sum(axis = 0) / float(cur_z.shape[0])
            prior_prob = np.abs(all_z - prior_off_prob)

            # then, locate the novel features in all_z
            mask = np.ones(all_z.shape)
            mask[:,np.where(cur_z.sum(axis = 0) > 0)] = 0
            novel_all_z = all_z * mask
            
            # temporarily mark those cells to have probability 1
            prior_prob[novel_all_z==1] = 1

            # we can safely do row product now, still ignoring new features
            prior_prob = prior_prob.prod(axis = 1)

            # let's count the number of new features for each row
            num_novel = novel_all_z.sum(axis = 1)
            # calculate the probability
            novel_prob = poisson.pmf(num_novel, self.alpha / float(cur_z.shape[0]))
            # ignore the novel == 0 special case
            novel_prob[num_novel==0] = 1.

            # multiply it by prior prob
            prior_prob = prior_prob * novel_prob
            
            # END p(z|z_inferred) calculation

            # BEGIN p(x|z, y_inferred)
            n_by_d = np.dot(all_z, cur_y)
            not_on_p = np.power(1. - self.lam, n_by_d) * (1. - self.epislon)
            for j in xrange(len(self.obs)):
                prob = np.abs(self.obs[j] - not_on_p).prod(axis=1) 
                prob = prob #* prior_prob
                prob = prob.sum()
                logprob_result[i,j] = prob
            # END
                
        return logprob_result.max(axis=0), logprob_result.std(axis=0)
        
        
if __name__ == '__main__':
    
    p = GibbsPredictor(cl_mode=False)
    p.read_test_csv('/home/qian/Dropbox/Projects/NegCorr/mturk/training-images-bits/test.csv')
    p.read_samples('/home/qian/Dropbox/Projects/NegCorr/mturk/training-images-bits/add-1000-noisyor-chain-1-nocl.pickled')
    #p.read_samples_csv('Y', '../data/ibp-image-n4-1000-noisyor-chain-1-nocl-Y.csv.gz')
    #p.read_samples_csv('Z', '../data/ibp-image-n4-1000-noisyor-chain-1-nocl-Z.csv.gz')
    results = p.predict()
    print(results)
    #print(results[:,:2].mean(), results[:,:2].std())
    #print(results[:,2].mean(), results[:,2].std())
    #print(results[:,3].mean(), results[:,3].std())

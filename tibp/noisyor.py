#!/usr/bin/env python2
#-*- coding: utf-8 -*-

from __future__ import print_function, division
import sys, os, os.path, itertools
import sys
from pyopencl.clrandom import rand as clrand
import pyopencl.array as clarray

pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(pkg_dir)

from scipy.stats import poisson
from scipy.special import gammaln
from scipy.misc import comb
from MPBNP import *
from MPBNP import BaseSampler, BasePredictor
from transforms import *
import math
import numpy.matlib

np.set_printoptions(suppress=True)

def harmonic_n(n):
    """ Calculates the approximate value of n-th harmonic number (from stackoverflow.com/questions/404346/)
        is correct up to O(n^{-6})
    :param n:
    :return: n-th harmonic number
    """
    # Euler-Mascheroni constant (could expand more if needed)
    e_m_const = 0.577215665
    return e_m_const+np.log(n) + 0.5/n - 1./(12*n**2) + 1./(120*n**4)

class Gibbs(BaseSampler):
    #TODO: consistency with transformations
    V_TRANS = 0
    H_TRANS = 1
    #currently not working for V_scale and H_scale
    V_SCALE = -1
    H_SCALE = -2

    NUM_TRANS = 2
    
    def __init__(self, cl_mode = True, cl_device = None, record_best = True,
                 alpha = None, lam = 0.999, theta = 0.2, epislon = 0.001,
                 sim_annealP = True, init_k = 10, T_init=40., anneal_rate = .9999,
                 splitProb = 0.5, split_mergeP = True, split_steps=20,
                 splitAdj = 1):
        """Initialize the class.
        """
        BaseSampler.__init__(self, cl_mode = cl_mode, cl_device = cl_device, record_best = record_best)

        if cl_mode:
            program_str = open(pkg_dir + 'MPBNP/tibp/kernels/tibp_noisyor_cl.c', 'r').read()
            self.prg = cl.Program(self.ctx, program_str).build() 

        self.alpha = alpha # tendency to generate new features
        self.k = init_k    # initial number of features
        self.theta = theta # prior probability that a pixel is on in a feature image
        self.lam = lam # efficacy of a feature
        self.epislon = epislon # probability that a pixel is on by change in an actual image
        self.phi = 0.5 # prior probability that no transformation is applied
        self.samples = {'z': [], 'y': [], 'r': []} # sample storage, to be pickled
        self.k_max_new = 10 # nax number of new features in one iteration
        self.T_init = T_init
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
                #self.d_obs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.obs.astype(np.int32))
                self.d_obs = clarray.Array(self.queue, shape=self.obs.shape, dtype=np.int32)
                self.d_obs.set(self.obs)


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
                # self.d_obs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.obs.astype(np.int32))
                self.d_obs = clarray.Array(self.queue, shape=self.obs.shape, dtype=np.int32)
                self.d_obs.set(self.obs)

        self.d = self.obs.shape[1]
        print("D: ", self.d)
        self.img_h = int(self.d / self.img_w)
        if self.cl_mode:
            #calculate approx. max # of features possible to allocate at once.
            max_mem = self.device.max_mem_alloc_size
            mem_per_feat = self.N * self.d * np.dtype('float32').itemsize
            self.max_k = max_mem // (mem_per_feat*6)
            #self.max_k = 30
            print("max_k:", self.max_k)

        #self.alpha = float(self.N) * 5
        self.alpha = np.log(float(self.N))
        return


    # def read_csv(self, filepath, header=True):
    #     """Read the data from a csv file.
    #     """
    #     print("Begin data read in....", file=sys.stderr)
    #     BaseSampler.read_csv(self, filepath, header)
    #     # convert the data to the appropriate formats
    #     self.new_obs = []
    #     self.img_w, self.img_h = None, None
    #     for row in self.obs:
    #         if self.img_w is None:
    #             self.img_w = int(row[0])
    #             if self.img_w == 0 or (len(row)-1) % self.img_w != 0:
    #                 raise Exception('The sampler does not understand the format of the data. Did you forget to specify image width in the data file?')
    #         self.new_obs.append([int(_) for _ in row])
    #
    #     self.obs = np.array(self.new_obs)[:,1:]
    #     if self.cl_mode:
    #         #self.d_obs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.obs.astype(np.int32))
    #         self.d_obs = clarray.Array(self.queue, shape=self.obs.shape, dtype=np.int32)
    #         self.d_obs.set(self.obs)
    #     # self.d is the length of the flattened vectors
    #     self.d = self.obs.shape[1]
    #     self.img_h = int(self.d / self.img_w)
    #     self.alpha = self.N
    #     self.hn = harmonic_n(self.N)
    #     print("Data read in finished....", file=sys.stderr)
    #     return

    def direct_read_obs(self, obs):
        """Read the data from a numpy array.
        """
        BaseSampler.direct_read_obs(self, obs)
        self.d = self.obs.shape[1]
        
    def do_inference(self, init_y = None, init_z = None, init_r = None, output_file = None):
        """Perform inference on the given observations assuming data are generated by an IBP model
        with noisy-or as the likelihood function.
        @param init_y: An initial feature image matrix, where values are 0 or 1
        @param init_z: An initial feature ownership matrix, where values are 0 or 1
        """
        BaseSampler.do_inference(self, output_file=None)
        if init_y is None:
            init_y = np.random.randint(0, 2, (self.k, self.d))
        else:
            assert(type(init_y) is np.ndarray)
            assert(init_y.shape == (self.k, self.d))
        if init_z is None:
            init_z = np.random.randint(0, 2, (len(self.obs), self.k))
        else:
            assert(type(init_z) is np.ndarray)
            assert(init_z.shape == (len(self.obs), self.k))

        if init_r is None:
            init_r = np.require(np.zeros(shape = (self.N, self.k*self.NUM_TRANS), dtype=np.int32),
                                dtype=np.int32, requirements=['C','A'])
            # init_r[:,:,self.V_SCALE] = 0
            # init_r[:,:,self.H_SCALE] = 0
            # init_r[:,:,self.V_TRANS] = np.random.randint(0, 2, (self.N, self.k))
            # init_r[:,:,self.H_TRANS] = np.random.randint(0, 2, (self.N, self.k))

        else:
            assert(init_r is None)

        if self.cl_mode:
            #timing_stats = self._cl_infer_yzr(init_y, init_z, init_r)
            timing_stats = self._cl_infer_yzr(init_y, init_z, init_r,True)
        else:
            timing_stats = self._infer_yzr(init_y, init_z, init_r)

        # report the results
        if output_file is sys.stdout:
            if self.record_best:
                final_y, final_z, final_r = self.best_sample[0]
                num_of_feats = final_z.shape[1]
                print('parameter,value',
                      'alpha,%f' % self.alpha, 'lambda,%f' % self.lam, 'theta,%f' % self.theta,
                      'epislon,%f' % self.epislon, 'phi,%f' % self.phi, 'inferred_K,%d' % num_of_feats,
                      'gpu_time,%f' % timing_stats[0], 'total_time,%f' % timing_stats[1],
                      file = output_file, sep = '\n')
                
                np.savetxt(output_file, final_z, fmt="%d", comments='', delimiter=',',
                           header=','.join(['feature%d' % _ for _ in range(num_of_feats)]))

                for k in xrange(num_of_feats):
                    print('Feature %d\n---------' % k, file = output_file)
                    np.savetxt(output_file, final_y[k].reshape(self.img_w, self.img_h),
                               fmt="%d", delimiter=',')

                print('object', 'feature', 'v_scale', 'h_scale', 'v_translation', 'h_translation',
                      file=output_file, sep=',')
                trans_inds = np.arange(self.NUM_TRANS)
                for n in xrange(self.N):
                    for k in xrange(num_of_feats):
                        print(n, k, *final_r[n,k+trans_inds], file=output_file, sep=',')
                print("outputting reconstructed objects:",file=output_file)
                d_obj_recon = self.d_obj_recon.fill(0)
                if final_z.size > 0:
                    N = self.N
                    K = final_y.shape[0]
                    D = final_y.shape[1]
                    d_cur_z = clarray.Array(self.queue,shape=(N,K),dtype=np.int32, allocator=self.d_mp)
                    d_cur_z.set(ary=final_z)
                    d_cur_y = clarray.Array(self.queue,shape=(K,D),dtype=np.int32, allocator=self.d_mp)
                    d_cur_y.set(ary=final_y)
                    d_cur_r= clarray.Array(self.queue,shape=(N,K*self.NUM_TRANS),dtype=np.int32, allocator=self.d_mp)
                    d_cur_r.set(ary=final_r)

                    self.prg.compute_recon_objs_trans(self.queue, (N, K, D), None,
                                                      d_cur_y.data, d_cur_z.data, d_cur_r.data,
                                                      d_obj_recon.data, np.int32(N), np.int32(K), np.int32(D),
                                                      np.int32(self.img_w))
                obj_recon = d_obj_recon.get()
                np.savetxt(output_file, obj_recon, fmt="%d", comments="", delimiter=",")
        else:
            if self.record_best:
                final_y, final_z, final_r = self.best_sample[0]
                num_of_feats = final_z.shape[1]
                try: os.mkdir(output_file)
                except: pass
                print('parameter,value',
                      'alpha,%f' % self.alpha, 'lambda,%f' % self.lam, 'theta,%f' % self.theta,
                      'epislon,%f' % self.epislon, 'phi,%f' % self.phi, 'inferred_K,%d' % num_of_feats,
                      'gpu_time,%f' % timing_stats[0], 'total_time,%f' % timing_stats[1],
                      file = gzip.open(output_file + 'parameters.csv.gz', 'w'), sep = '\n')
                
                np.savetxt(gzip.open(output_file + 'feature_ownership.csv.gz', 'w'), final_z,
                           fmt="%d", comments='', delimiter=',',
                           header=','.join(['feature%d' % _ for _ in range(num_of_feats)]))

                for k in xrange(num_of_feats):
                    np.savetxt(gzip.open(output_file + 'feature_%d_image.csv.gz' % k, 'w'),
                               final_y[k].reshape(self.img_w, self.img_h), fmt="%d", delimiter=',')

                transform_fp = gzip.open(output_file + 'transformations.csv.gz', 'w')
                print('object', 'feature', 'v_scale', 'h_scale', 'v_translation', 'h_translation',
                      file = transform_fp, sep=',')
                trans_inds = np.arange(self.NUM_TRANS)
                for n in xrange(self.N):
                    for k in xrange(num_of_feats):
                        print(n, k, *final_r[n,k+trans_inds], file=transform_fp, sep=',')
                transform_fp.close()
            else:
                try: os.mkdir(output_file)
                except: pass
                print('parameter,value',
                      'alpha,%f' % self.alpha, 'lambda,%f' % self.lam, 'theta,%f' % self.theta,
                      'epislon,%f' % self.epislon, 'phi,%f' % self.phi,
                      'gpu_time,%f' % timing_stats[0], 'total_time,%f' % timing_stats[1],
                      file = gzip.open(output_file + 'parameters.csv.gz', 'w'), sep = '\n')
                np.savez_compressed(output_file + 'feature_ownership.npz', self.samples['z'])
                np.savez_compressed(output_file + 'feature_images.npz', self.samples['y'])
                np.savez_compressed(output_file + 'transformations.npz', self.samples['r'])

        return timing_stats

    def _infer_yzr(self, init_y, init_z, init_r):
        """Wrapper function to start the inference on y, z and r.
        This function is not supposed to directly invoked by an end user.
        @param init_y: Passed in from do_inference()
        @param init_z: Passed in from do_inference()
        @param init_r: Passed in from do_inference()
        """
        cur_y = init_y
        cur_z = init_z
        cur_r = init_r

        a_time = time()
        if self.record_best: self.auto_save_sample(sample = (cur_y, cur_z, cur_r))
        for i in xrange(self.niter):
            temp_cur_y = self._infer_y(cur_y, cur_z, cur_r)
            temp_cur_y, temp_cur_z, temp_cur_r = self._infer_z(temp_cur_y, cur_z, cur_r)
            temp_cur_r = self._infer_r(temp_cur_y, temp_cur_z, temp_cur_r)

            if self.record_best:
                if self.auto_save_sample(sample = (temp_cur_y, temp_cur_z, temp_cur_r)):
                    cur_y, cur_z, cur_r = temp_cur_y, temp_cur_z, temp_cur_r
                if self.no_improvement(1000):
                    break                    
                
            elif i >= self.burnin:
                cur_y, cur_z, cur_r = temp_cur_y, temp_cur_z, temp_cur_r
                self.samples['z'].append(cur_z)
                self.samples['y'].append(cur_y)
                self.samples['r'].append(cur_r)

        self.total_time += time() - a_time
        return self.gpu_time, self.total_time, None

    def _infer_y(self, cur_y, cur_z, cur_r):
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
                on_loglik[row, col] = self._loglik_nth(cur_y, cur_z, cur_r, n = affected_data_index)
                cur_y[row, col] = 0
                off_loglik[row, col] = self._loglik_nth(cur_y, cur_z, cur_r, n = affected_data_index)
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

    def _infer_z(self, cur_y, cur_z, cur_r):
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
                on_prob[row, col] = on_prob[row, col] * np.exp(self._loglik_nth(cur_y, cur_z, cur_r, n = row))
                cur_z[row, col] = 0
                off_prob[row, col] = off_prob[row, col] * np.exp(self._loglik_nth(cur_y, cur_z, cur_r, n = row))
                cur_z[row, col] = old_value

        # normalize the probability
        on_prob = on_prob / (on_prob + off_prob)

        # sample the values
        cur_z = np.random.binomial(1, on_prob)

        # sample new features use importance sampling
        k_new = self._sample_k_new(cur_y, cur_z, cur_r)
        if k_new:
            cur_y, cur_z, cur_r = k_new
        
        # delete empty feature images
        non_empty_feat_img = np.where(cur_y.sum(axis = 1) > 0)
        cur_y = cur_y[non_empty_feat_img[0],:]
        cur_z = cur_z[:,non_empty_feat_img[0]]
        cur_r = np.array([_[non_empty_feat_img[0],:] for _ in cur_r])
        
        # delete null features
        active_feat_col = np.where(cur_z.sum(axis = 0) > 0)
        cur_z = cur_z[:,active_feat_col[0]]
        cur_y = cur_y[active_feat_col[0],:]
        cur_r = np.array([_[active_feat_col[0],:] for _ in cur_r])

        # update self.k
        self.k = cur_z.shape[1]
        
        return cur_y, cur_z, cur_r

    def _infer_r(self, cur_y, cur_z, cur_r):
        """Infer transformations.
        """
        rand_v = np.random.randint(0, self.img_h, size=(cur_z.shape[0], cur_z.shape[1]))
        rand_h = np.random.randint(0, self.img_w, size=(cur_z.shape[0], cur_z.shape[1]))
        rand_v_scale = np.random.randint(-self.img_h+2, self.img_h, size=(cur_z.shape[0], cur_z.shape[1]))
        rand_h_scale = np.random.randint(-self.img_w+2, self.img_w, size=(cur_z.shape[0], cur_z.shape[1]))
        # iterate over each transformation and resample it 
        for nth_img in xrange(cur_r.shape[0]):
            for kth_feature in xrange(cur_r.shape[1]):
                old_loglik = self._loglik_nth(cur_y, cur_z, cur_r, n=nth_img)

                # resample vertical translation
                old_v_trans = cur_r[nth_img, kth_feature, self.V_TRANS]
                # set a new vertical transformation
                cur_r[nth_img, kth_feature, self.V_TRANS] = rand_v[nth_img, kth_feature] #np.random.randint(0, self.img_h)

                old_logprior = np.log(abs((old_v_trans > 0) - self.phi))
                new_logprior = np.log(abs((rand_v[nth_img, kth_feature] > 0) - self.phi))
                
                new_loglik = self._loglik_nth(cur_y, cur_z, cur_r, n = nth_img)
                move_prob = 1 / (1 + np.exp(old_loglik + old_logprior - new_loglik - new_logprior))
                if random.random() > move_prob: # revert changes if move_prob too small
                    cur_r[nth_img, kth_feature, self.V_TRANS] = old_v_trans
                else:
                    old_loglik = new_loglik

                # resample horizontal translation
                old_h_trans = cur_r[nth_img, kth_feature, self.H_TRANS]
                # set a new vertical transformation
                cur_r[nth_img, kth_feature, self.H_TRANS] = rand_h[nth_img, kth_feature]

                old_logprior = np.log(abs((old_h_trans > 0) - self.phi))
                new_logprior = np.log(abs((rand_h[nth_img, kth_feature] > 0) - self.phi))

                new_loglik = self._loglik_nth(cur_y, cur_z, cur_r, n = nth_img)
                move_prob = 1 / (1 + np.exp(old_loglik + old_logprior - new_loglik - new_logprior))
                if random.random() > move_prob: # revert changes if move_prob too small
                    cur_r[nth_img, kth_feature, self.H_TRANS] = old_h_trans
                else:
                    old_loglik = new_loglik

                # resample scale percentage
                old_v_scale = cur_r[nth_img, kth_feature, self.V_SCALE]
                # set a new vertical scale
                cur_r[nth_img, kth_feature, self.V_SCALE] = rand_v_scale[nth_img, kth_feature]

                old_logprior = np.log(abs((old_v_scale > 0) - self.phi))
                new_logprior = np.log(abs((rand_v_scale[nth_img, kth_feature] > 0) - self.phi))

                new_loglik = self._loglik_nth(cur_y, cur_z, cur_r, n = nth_img)
                move_prob = 1 / (1 + np.exp(old_loglik + old_logprior - new_loglik - new_logprior))
                if random.random() > move_prob: # revert changes if move_prob too small
                    cur_r[nth_img, kth_feature, self.V_SCALE] = old_v_scale
                else:
                    old_loglik = new_loglik

                # resample scale percentage
                old_h_scale = cur_r[nth_img, kth_feature, self.H_SCALE]
                # set a new horizontal scale
                cur_r[nth_img, kth_feature, self.H_SCALE] = rand_h_scale[nth_img, kth_feature]

                old_logprior = np.log(abs((old_h_scale > 0) - self.phi))
                new_logprior = np.log(abs((rand_h_scale[nth_img, kth_feature] > 0) - self.phi))

                new_loglik = self._loglik_nth(cur_y, cur_z, cur_r, n = nth_img)
                move_prob = 1 / (1 + np.exp(old_loglik + old_logprior - new_loglik - new_logprior))
                if random.random() > move_prob: # revert changes if move_prob too small
                    cur_r[nth_img, kth_feature, self.H_SCALE] = old_h_scale
                    
        return cur_r
    
    def _sample_k_new(self, cur_y, cur_z, cur_r):
        """TODO: Joe isn't so sure this is correct for non-small dimensions/#of objs...

        Sample new features for all rows using Metropolis hastings.
        (This is a heuristic strategy aiming for easy parallelization in an 
        equivalent GPU implementation. We here have effectively treated the
        current Z as a snapshot frozen in time, and each new k is based on
        this frozen snapshot of Z. In a more correct procedure, we should
        go through the rows and sample k new for each row given all previously
        sampled new ks.)
        """
        N = float(len(self.obs))
        #old_loglik = self._loglik(cur_y, cur_z, cur_r)

        k_new_count = np.random.poisson(self.alpha / N)
        if k_new_count == 0: return False
            
        # modify the feature ownership matrix
        cur_z_new = np.hstack((cur_z, np.random.randint(0, 2, size = (cur_z.shape[0], k_new_count))))
        #cur_z_new[:, [xrange(-k_new_count,0)]] = 1
        # propose feature images by sampling from the prior distribution
        cur_y_new = np.vstack((cur_y, np.random.binomial(1, self.theta, (k_new_count, self.d))))
        cur_r_new = np.array([np.vstack((_, np.zeros((k_new_count, self.NUM_TRANS)))) for _ in cur_r])
        return cur_y_new.astype(np.int32), cur_z_new.astype(np.int32), cur_r_new.astype(np.int32)

    def _sample_lam(self, cur_y, cur_z):
        """Resample the value of lambda.
        """
        old_loglik = self._loglik(cur_y, cur_z)
        old_lam = self.lam
    
        # modify the feature ownership matrix
        self.lam = np.random.beta(1,1)
        new_loglik = self._loglik(cur_y, cur_z)
        move_prob = 1 / (1 + np.exp(old_loglik - new_loglik))
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

    def _loglik_nth(self, cur_y, cur_z, cur_r, n):
        """Calculate the loglikelihood of the nth data point
        given Y, Z and R.
        """
        assert(cur_z.shape[1] == cur_y.shape[0] == cur_r.shape[1])

        if type(n) is int: n = [n]
        else: n = n[0]
        not_on_p = np.empty((len(n), cur_y.shape[1]))
        
        # transform the feature images to obtain the effective y
        # this needs to be done on a per object basis

        for i in xrange(len(n)):
            nth = n[i]
            nth_y = copy.deepcopy(cur_y) # the transformed cur_y with respect to nth
            kth_feat = 0
            for r_feat in cur_r[nth]: # r_feat refers to the transforms applied one feature
                nth_y[kth_feat] = scale_manual(nth_y[kth_feat], self.img_w, r_feat[self.H_SCALE], r_feat[self.V_SCALE])
                nth_y[kth_feat] = v_translate(nth_y[kth_feat], self.img_w, r_feat[self.V_TRANS])
                nth_y[kth_feat] = h_translate(nth_y[kth_feat], self.img_w, r_feat[self.H_TRANS])
                kth_feat += 1
                
            not_on_p[i] = np.power(1. - self.lam, np.dot(cur_z[nth], nth_y)) * (1. - self.epislon)
        loglik = np.log(np.abs(self.obs[n] - not_on_p)).sum()
        return loglik

    def _loglik(self, cur_y, cur_z, cur_r):
        """Calculate the loglikelihood of data given Y, Z and R.
        """
        assert(cur_z.shape[1] == cur_y.shape[0] == cur_r.shape[1])

        not_on_p = np.empty((self.N, self.d))

        # transform the feature images to obtain the effective y
        # this needs to be done on a per object basis
        
        for nth in xrange(self.N):
            nth_y = copy.deepcopy(cur_y) # the transformed cur_y with respect to nth
            kth_feat = 0
            for r_feat in cur_r[nth]: # r_feat refers to the transforms applied one feature
                nth_y[kth_feat] = scale_manual(nth_y[kth_feat], self.img_w, r_feat[self.H_SCALE], r_feat[self.V_SCALE])
                nth_y[kth_feat] = v_translate(nth_y[kth_feat], self.img_w, r_feat[self.V_TRANS])
                nth_y[kth_feat] = h_translate(nth_y[kth_feat], self.img_w, r_feat[self.H_TRANS])
                kth_feat += 1
                
            not_on_p[nth] = np.power(1. - self.lam, np.dot(cur_z[nth], nth_y)) * (1. - self.epislon)
        
        loglik_mat = np.log(np.abs(self.obs - not_on_p))
        return loglik_mat.sum()

    def _z_by_ry(self, cur_y, cur_z, cur_r):
        """
        """
        z_by_ry = np.empty(shape = (cur_z.shape[0], cur_y.shape[1]), dtype=np.int64)
        for nth in xrange(self.N):
            nth_y = copy.deepcopy(cur_y) # the transformed cur_y with respect to nth
            kth_feat = 0
            for r_feat in cur_r[nth]: # r_feat refers to the transforms applied one feature
                nth_y[kth_feat] = v_translate(nth_y[kth_feat], self.img_w, r_feat[self.V_TRANS])
                nth_y[kth_feat] = h_translate(nth_y[kth_feat], self.img_w, r_feat[self.H_TRANS])
                kth_feat += 1
            z_by_ry[nth,] = np.dot(cur_z[nth], nth_y)
        return z_by_ry

    def _update_allocs(self, for_split_merge= False, accepted_sm=False, reset_after_sm=False,first_call=False):
        N = self.N
        K = self.k
        D = self.d
        if first_call is True:
            self.d_mp =  cl.tools.MemoryPool(cl.tools.ImmediateAllocator(self.queue))
            self.d_obj_recon = clarray.zeros(self.queue,shape=(N,D),dtype=np.int32, allocator=self.d_mp)
            # print('pre')
            # print(clarray.sum(self.d_obj_recon))
            self.d_recon_lps = clarray.Array(self.queue,shape=(N,D),dtype=np.float32, allocator=self.d_mp)

        if accepted_sm is False or reset_after_sm is True:
            self.d_lp_y_off = clarray.zeros(self.queue,shape=(K,D),dtype=np.float32, allocator=self.d_mp)
            self.d_lp_y_on = clarray.zeros(self.queue,shape=(K,D),dtype=np.float32, allocator=self.d_mp)
            # self.d_lp_z_off = clarray.zeros(self.queue,shape=(N,K,D), dtype=np.float32, allocator=self.d_mp)
            # self.d_lp_z_on = clarray.zeros(self.queue,shape=(N,K,D), dtype=np.float32, allocator=self.d_mp)
            self.d_lp_nkr_on = clarray.Array(self.queue,shape=(N,K,D), dtype = np.float32, allocator=self.d_mp)
            self.d_lp_nk_off = clarray.Array(self.queue, shape=(N,K), dtype=np.float32, allocator=self.d_mp)
            self.d_rsums = clarray.Array(self.queue,shape=(N,K,D), dtype=np.float32, allocator=self.d_mp)

        if reset_after_sm is False and (accepted_sm is True or for_split_merge is False):
            self.tmp_z = np.require(np.zeros(shape=(self.N,self.k), dtype=np.int32),
                                                     dtype=np.int32, requirements=['C','A'])
            self.tmp_y = np.require(np.zeros(shape=(self.k,self.d), dtype=np.int32),
                                                     dtype=np.int32, requirements=['C','A'])

            self.d_knew_lp = clarray.Array(self.queue,shape=(N,D,self.k_max_new+1),dtype=np.float32,
                                           allocator=self.d_mp)

    def _cl_infer_yzr(self, init_y, init_z, init_r, no_scaling=True):
        """Wrapper function to start the inference on y and z.
        This function is not supposed to directly invoked by an end user.
        @param init_y: Passed in from do_inference()
        @param init_z: Passed in from do_inference()
        @param init_r: Passed in from do_inference()
        @param no_scaling: Passed in from do_inference() true limits transformations to translations.
        """
        total_time = time()
        print("no scaling: " + str(no_scaling), file=sys.stderr)
        #for debugging
        # init_z = np.require(np.array([[1],[1],[0],[0],[1],[1],[0],[0]]),
        #                     dtype=np.int32, requirements=['C','A'])
        # init_y = np.require(np.array([[1,1,0,0]]), dtype=np.int32,
        #                     requirements=['C','A'])
        # init_r = np.require(np.zeros(shape=(self.N,1*self.NUM_TRANS),dtype=np.int32),
        #                     dtype=np.int32, requirements=['C','A'])
        # self.k = 1

        cur_y = init_y.astype(np.int32)
        cur_z = init_z.astype(np.int32)
        cur_r = init_r.astype(np.int32) # this is fine with only translations

        self.tmp_y = cur_y
        self.tmp_z = cur_z

        debugP = False
        N = cur_z.shape[0]
        K = cur_z.shape[1]
        D = cur_y.shape[1]
        #d_cur_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = cur_y.astype(np.int32))
        #d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = cur_z.astype(np.int32))
        self._update_allocs(first_call=True)
        d_cur_y = clarray.Array(self.queue, cur_y.shape, dtype=np.int32, allocator=self.d_mp)
        d_cur_y.set(cur_y)
        d_cur_z = clarray.Array(self.queue, cur_z.shape, dtype=np.int32, allocator=self.d_mp)
        d_cur_z.set(cur_z)
        d_cur_r = clarray.Array(self.queue, cur_r.shape, dtype=np.int32, allocator=self.d_mp)
        d_cur_r.set(cur_r)


        self.auto_save_sample(sample = (d_cur_y, d_cur_z, d_cur_r))
        for i in xrange(self.niter):
            self.T = max(self.T*self.anneal_rate,1.)
            if (i % (self.niter/10)) == 0:
                print("%d sweeps of %d with T %f and K %d" % (i,self.niter,self.T, self.k))
            self.cur_iter = i
            a_time = time()

            if (no_scaling is False):
                d_cur_y = self._cl_infer_y(cur_y, cur_z, cur_r, d_cur_y, d_cur_z, d_cur_r)
                d_cur_z = self._cl_infer_z(cur_y, cur_z, cur_r, d_cur_y, d_cur_z, d_cur_r)
                temp_cur_r = self._cl_infer_r(cur_y, cur_z, cur_r, d_cur_y, d_cur_z, d_cur_r)
            else:
                d_cur_y = self._cl_infer_y_noscNew(d_cur_y, d_cur_z, d_cur_r)
                d_cur_z, d_cur_r = self._cl_infer_zr_nosc(d_cur_y, d_cur_z, d_cur_r)
                self.tmp_z = d_cur_z.map_to_host(is_blocking=True)
                self.tmp_y = d_cur_y.map_to_host(is_blocking=True)
                self.tmp_r = d_cur_r.map_to_host(is_blocking=True)

                d_cur_y, d_cur_z, d_cur_r = self._cl_infer_k_newJLA2(self.tmp_y, self.tmp_z, self.tmp_r)

                #d_cur_z, d_cur_y, d_cur_r = self._cl_infer_knew(cur_y,cur_z,cur_r)


            a_time = time()
            # temp_cur_y = np.empty_like(cur_y)
            # cl.enqueue_copy(self.queue, temp_cur_y, d_cur_y)
            # temp_cur_z = np.empty_like(cur_z)
            # cl.enqueue_copy(self.queue, temp_cur_z, d_cur_z)
            # temp_cur_r = np.empty_like(cur_r)
            # cl.enqueue_copy(self.queue,temp_cur_r, d_cur_r)
            self.gpu_time += time() - a_time
            if no_scaling is False:
                temp_cur_y, temp_cur_z, temp_cur_r = self._cl_infer_k_new(temp_cur_y, temp_cur_z, temp_cur_r)


            if self.record_best:
                if no_scaling is False:
                    if self.auto_save_sample(sample = (temp_cur_y, temp_cur_z, temp_cur_r)):
                        print('Number of features:', cur_z.shape[1], file=sys.stderr)
                    cur_y, cur_z, cur_r = temp_cur_y, temp_cur_z, temp_cur_r
                else:
                    self.auto_save_sample(sample=(d_cur_y, d_cur_z, d_cur_r))

                if self.no_improvement(1000):
                    break                    
            elif i >= self.burnin:
                if no_scaling is False:
                    cur_y, cur_z, cur_r = temp_cur_y, temp_cur_z, temp_cur_r
                    self.samples['z'].append(cur_z)
                    self.samples['y'].append(cur_y)
                    self.samples['r'].append(cur_r)
                else:
                    self.samples['z'].append(d_cur_z.get())
                    self.samples['y'].append(d_cur_y.get())
                    self.samples['r'].append(d_cur_r.get())
            
        self.total_time += time() - total_time

        return self.gpu_time, self.total_time, None

    def _cl_infer_y(self, cur_y, cur_z, cur_r, d_cur_y, d_cur_z, d_cur_r):
        """Infer feature images
        """
        a_time = time()
        d_z_by_ry = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                              hostbuf = np.empty(shape = self.obs.shape, dtype = np.int32))
        d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                           hostbuf = np.random.random(cur_y.shape).astype(np.float32))
        transformed_y = np.empty(shape = (self.obs.shape[0], cur_z.shape[1], self.obs.shape[1]), dtype = np.int32)
        d_transformed_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = transformed_y)
        d_temp_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = transformed_y)

        # first transform the feature images and calculate z_by_ry
        self.prg.compute_z_by_ry(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r, d_transformed_y, d_temp_y, d_z_by_ry,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # calculate the prior probability that a pixel is on
        self.prg.sample_y(self.queue, cur_y.shape, None,
                          d_cur_y, d_cur_z, d_z_by_ry, d_cur_r, self.d_obs, d_rand,
                          np.int32(self.N), np.int32(self.d), np.int32(cur_y.shape[0]), np.int32(self.img_w),
                          np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        self.gpu_time += time() - a_time
        return d_cur_y

    def _cl_infer_y_nosc(self, cur_y, cur_z, cur_r, d_cur_y, d_cur_z, d_cur_r):
        """Infer feature images (only translations)
        """
        a_time = time()
        d_z_by_ry = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                              hostbuf = np.empty(shape = self.obs.shape, dtype = np.int32))
        d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                           hostbuf = np.random.random(cur_y.shape).astype(np.float32))
        transformed_y = np.empty(shape = (self.obs.shape[0], cur_z.shape[1], self.obs.shape[1]), dtype = np.int32)
        d_transformed_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = transformed_y)
        d_temp_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = transformed_y)

        # first transform the feature images and calculate z_by_ry
        self.prg.compute_z_by_ry_nosc(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r, d_transformed_y, d_temp_y, d_z_by_ry,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # calculate the prior probability that a pixel is on
        self.prg.sample_y_nosc(self.queue, cur_y.shape, None,
                          d_cur_y, d_cur_z, d_z_by_ry, d_cur_r, self.d_obs, d_rand,
                          np.int32(self.N), np.int32(self.d), np.int32(cur_y.shape[0]), np.int32(self.img_w),
                          np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        self.gpu_time += time() - a_time
        return d_cur_y

    def _cl_infer_z(self, cur_y, cur_z, cur_r, d_cur_y, d_cur_z, d_cur_r):
        """Infer feature ownership
        """
        a_time = time()
        d_z_by_ry = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                              hostbuf = np.empty(shape = self.obs.shape, dtype = np.int32))
        d_z_col_sum = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                hostbuf = cur_z.sum(axis = 0).astype(np.int32))
        d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                           hostbuf = np.random.random(cur_z.shape).astype(np.float32))
        transformed_y = np.empty(shape = (self.obs.shape[0], cur_z.shape[1], self.obs.shape[1]), dtype = np.int32)
        d_transformed_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = transformed_y)
        d_temp_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = transformed_y)


        # first transform the feature images and calculate z_by_ry
        self.prg.compute_z_by_ry(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r, d_transformed_y, d_temp_y, d_z_by_ry,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))


        # calculate the prior probability that a pixel is on
        self.prg.sample_z(self.queue, cur_z.shape, None,
                          d_cur_y, d_cur_z, d_cur_r, d_z_by_ry, d_z_col_sum, self.d_obs, d_rand,
                          np.int32(self.N), np.int32(self.d), np.int32(cur_y.shape[0]), np.int32(self.img_w),
                          np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        self.gpu_time += time() - a_time
        return d_cur_z

    def _cl_infer_knew(self, cur_y, cur_z, cur_r, d_cur_y, d_cur_z, d_cur_r):
        """Written by Joe Austerweil on 07/16/15
            Infer new features (bounded by self.k_max_new). Cleans (removes unused features) first
        :param cur_y: feature images on host
        :param cur_z: feature ownership on host
        :param cur_r: feature transforms on host
        :param d_cur_y: feature images on device
        :param d_cur_z: feature ownership on device
        :param d_cur_r: feature transforms on device
        :return: d_cur_z, d_cur_y, d_cur_r
        """
        #TODO: do better gpu timing for this function!!
        a_time = time()
        cl.enqueue_copy(self.queue, cur_z, d_cur_z)
        cl.enqueue_copy(self.queue, cur_y, d_cur_y)
        cl.enqueue_copy(self.queue, cur_r, d_cur_r)

         # delete empty feature images
        #non_empty_feat_img = np.where(cur_y.sum(axis = 1) > 0)
        #cur_y = cur_y[non_empty_feat_img[0],:].astype(np.int32)
        #cur_z = cur_z[:,non_empty_feat_img[0]].astype(np.int32)
        #cur_r = np.array([_[non_empty_feat_img[0],:] for _ in cur_r]).astype(np.int32)

        # delete null features
        active_feat_col = np.where(cur_z.sum(axis = 0) > 0)
        cur_z = cur_z[:,active_feat_col[0]].astype(np.int32)
        cur_y = cur_y[active_feat_col[0],:].astype(np.int32)
        cur_r = np.array([_[active_feat_col[0],:] for _ in cur_r]).astype(np.int32)

        # update self.k
        self.k = cur_z.shape[1]

        z_s0, z_s1 = cur_z.shape
        cur_z = cur_z.reshape((z_s0 * z_s1, 1))
        cur_z = np.require(cur_z.reshape((z_s0, z_s1)), dtype=np.int32, requirements=['C','A'])

        y_s0, y_s1 = cur_y.shape
        cur_y = cur_y.reshape((y_s0 * y_s1, 1))
        cur_y = np.require(cur_y.reshape((y_s0, y_s1)), dtype=np.int32, requirements=['C','A'])

        r_s0, r_s1, r_s2 = cur_r.shape
        cur_r = cur_r.reshape((r_s0 * r_s1 * r_s2, 1))
        cur_r = np.require(cur_r.reshape((r_s0, r_s1, r_s2)), dtype=np.int32, requirements=['C','A'])

        N = self.obs.shape[0]
        K = cur_z.shape[1]
        D = self.obs.shape[1]

        #sample new features. 1st recon!

        d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf=cur_z.astype(np.int32))
        d_cur_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf = cur_y.astype(np.int32))
        d_cur_r = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf = cur_r.astype(np.int32))
        d_obj_recon = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                                hostbuf = np.require(np.zeros(shape = self.obs.shape, dtype=np.int32),
                                                     dtype = np.float32, requirements=['C','A']))
        self.prg.compute_recon_objs(self.queue, (N, K, D), None,
                                 d_cur_y, d_cur_z, d_cur_r, d_obj_recon,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))
        #TODO: gpuify lpx calc
        lps = np.require(np.zeros((N,D,self.k_max_new+1), dtype=np.float32), dtype=np.float32, requirements=['C','A'])
        d_lps = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                          hostbuf=lps.astype(np.float32))
        #TODO: localify k_max_new to share some values
        self.prg.calc_lp_fornew(self.queue, (N, K, self.k_max_new+1), None,
                         d_obj_recon, self.d_obs, d_lps,
                         np.int32(N), np.int32(D), np.int32(K), np.int32(self.k_max_new+1),
                         np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        cl.enqueue_copy(self.queue, lps, d_lps)
        KNewRng =  np.arange(K, self.k_max_new+1+K)
        kNewLPs = np.sum(lps,axis=1) - \
                  np.matlib.repmat(self.alpha/N + KNewRng.T * np.log(self.alpha/N) - gammaln(KNewRng.T-1),N,1)

        logmaxs = np.amax(kNewLPs,axis=1)
        pdfs = (np.exp(kNewLPs.T-logmaxs)).T
        pdfs = np.divide(pdfs.T, np.sum(pdfs,axis=1)).T
        knews = np.apply_along_axis(lambda x: np.random.choice(KNewRng-K,p=x), axis=1, arr=pdfs)
        #knews = np.require(knews,dtype=np.int32, requirements=['C', 'A'])
        partSum = 0
        for i in np.arange(knews.size):
            if (partSum +knews[i]) > self.k_max_new:
                knews[i] = max(self.k_max_new-partSum,0)
            partSum += knews[i]
        totNewK = np.sum(knews)

        new_z = np.zeros(shape=(N,totNewK),dtype=np.int32)
        for k_ind in np.arange(totNewK):
            new_z[:,k_ind] = knews > k_ind

        #update z,y,r to prep for creation
        cur_z = np.require(np.hstack((cur_z, new_z)), dtype=np.int32, requirements=['C','A'])

        cur_y = np.require(np.vstack((cur_y, np.zeros(shape=(totNewK,D), dtype=np.int32))),
                           dtype=np.int32, requirements=['C', 'A'])
        cur_r = np.require(np.array([np.vstack((_, np.zeros((totNewK, self.NUM_TRANS)))) for _ in cur_r]),
                           dtype=np.int32, requirements=['C', 'A'])
        oldK = K
        K += totNewK

        d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf = cur_z.astype(np.int32))
        d_cur_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf = cur_y.astype(np.int32))
        d_cur_r = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf = cur_r.astype(np.int32))
        #d_knews = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
        #                          hostbuf = knews.astype(np.int32))

        comb_vec = np.require(comb(totNewK, np.arange(totNewK+1)),
                              dtype=np.int32, requirements=['C','A'])
        d_comb_vec = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                               hostbuf = comb_vec.astype(np.int32))
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
        curNumToRun = max(D, workGroupSize)
        num_d_workers = workGroupSize

        num_loop = int(math.ceil(D/workGroupSize))
        for i in np.arange(num_loop):
            self.prg.new_y_val_probs(self.queue, (num_d_workers, N, totNewK), (workGroupSize, 1,1), d_locmemInt,
                                       d_cur_z, d_cur_y, d_cur_r, d_comb_vec, d_obj_recon, self.d_obs,
                                       d_new_y_val_probs, np.int32(N), np.int32(K), np.int32(D), np.int32(totNewK),
                                       np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        cl.enqueue_copy(self.queue, new_y_val_probs, d_new_y_val_probs)
        #normalize with appropriate k_new per x
        #TODO: JLA thinks there's an error here bc the xs that want new ys could be misaligned\
        # it's possible sampling iwll be robust to this mistake
            #used to be this:  np.swapaxes(np.swapaxes(new_y_val_probs,1,2) * (knews > k_ind),2,1) (changed while tired)
        for k_ind in np.arange(totNewK):
            new_y_val_probs[k_ind,:,:] = (new_y_val_probs[k_ind,:,:].T * (knews > k_ind)).T
        new_y_norms = np.sum(new_y_val_probs,axis=0)

        new_y_val_probs = new_y_val_probs / new_y_norms

        #now sample num to turn on in y per d
        new_y_on = np.apply_along_axis(lambda x: np.random.choice(KNewRng-K,p=x), axis=1, arr=new_y_val_probs)
        knews_cumsum = np.cumsum(knews)

        uniqs, allToUn, unToAll = np.unique(knews_cumsum, update_indices = True, update_inverse = True)
        uniqs = np.setdiff1d(uniqs, 0, assume_unique=True)
        lastKNew = 0

        #update cur_y appropriately
        for i in uniqs:
            cur_y[:,(lastKNew+oldK):(oldK+i)] = int(new_y_on > i)
            lastKNew = i
        d_cur_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                            hostbuf = cur_y.astype(np.int32))

        self.gpu_time += time() - a_time

        return d_cur_z, d_cur_y, d_cur_r

    def _cl_infer_y_noscNew(self, d_cur_y, d_cur_z, d_cur_r):
        """ Infer new feature values with transforms and ownership known 
            Written by Joe Austerweil
        :param d_cur_y: feature images on device
        :param d_cur_z: feature ownership on device
        :param d_cur_r: feature transforms on device
        :return d_cur_y
        """
        a_time = time()

        K = self.k

        if K > 0:
            N = self.N
            D = self.d

            d_obj_recon = self.d_obj_recon.fill(0)

            self.prg.compute_recon_objs_trans(self.queue, (N, K, D), None,
                                     d_cur_y.data, d_cur_z.data, d_cur_r.data, d_obj_recon.data,
                                     np.int32(N), np.int32(K), np.int32(D),
                                     np.int32(self.img_w))


            #afterwards.. same as ibp samp y
            rand_vals = np.require(np.random.random(size = (K, D)),
                                   dtype=np.float32, requirements=['C','A'])
            d_rand = clarray.Array(self.queue,shape=(K,D),dtype=np.float32)
            d_rand.set(rand_vals)

            #TODO: figure out why clrand sometimes raises issues in oclgrind, so that i can use it...
            #d_rand = clrand(self.queue,(K,D), dtype=np.float32)

            d_lp_off = self.d_lp_y_off.fill(value=0.)
            d_lp_on = self.d_lp_y_on.fill(value=0.)
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
                if N <= 200:
                    self.prg.calc_y_lps_old(self.queue, (curNumToRun,K,D), (curNumToRun, 1,1), d_locmemFlt, d_cur_y.data, d_cur_z.data,
                                        d_obj_recon.data, self.d_obs.data, d_lp_off.data, d_lp_on.data,
                                        np.int32(N), np.int32(D), np.int32(K), np.int32(numPrevRun),
                                        np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))
                else:
                    self.prg.calc_y_lp_off(self.queue, (curNumToRun,K,D), (curNumToRun, 1,1),
                                       d_locmemFlt, d_cur_y.data, d_cur_z.data,
                                       d_obj_recon.data, self.d_obs.data, d_lp_off.data,
                                       np.int32(N), np.int32(D), np.int32(K), np.int32(numPrevRun),
                                       np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))
                    self.prg.calc_y_lp_off(self.queue, (curNumToRun,K,D), (curNumToRun, 1,1),
                                       d_locmemFlt, d_cur_y.data, d_cur_z.data,
                                       d_obj_recon.data, self.d_obs.data, d_lp_on.data,
                                       np.int32(N), np.int32(D), np.int32(K), np.int32(numPrevRun),
                                       np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))
                numToRun -= curNumToRun
                numPrevRun += curNumToRun


            self.prg.sample_y_pre_calc(self.queue, (K,D, 1), None, d_cur_y.data, d_lp_off.data, d_lp_on.data,
                                       d_rand.data, np.int32(K), np.int32(D))
        return d_cur_y



    def _cl_infer_zr_nosc(self, d_cur_y, d_cur_z, d_cur_r):
        """ Infer feature ownership and transforms jointly
            Written by Joe Austerweil on 07/10/15
        :param d_cur_y: feature images on device
        :param d_cur_z: feature ownership on device
        :param d_cur_r: feature transforms on device
        :return: d_cur_z, d_cur_r
        """

        a_time = time()
        N = self.obs.shape[0]
        K = self.k
        D = self.obs.shape[1]

        if K > 0:
            d_obj_recon = self.d_obj_recon.fill(0)

            d_z_col_sum = clarray.zeros(self.queue,(K,1),dtype=np.int32,allocator=self.d_mp)

            self.prg.compute_recon_objs_transzsum(self.queue, (N,K,D), None,
                                     d_cur_y.data, d_cur_z.data, d_cur_r.data, d_obj_recon.data, d_z_col_sum.data,
                                     np.int32(N), np.int32(K), np.int32(D),
                                     np.int32(self.img_w))

            #for debugging only:
    #        tmpRecon = np.zeros(shape=self.obs.shape,dtype=np.int32)
    #        cl.enqueue_copy(self.queue, tmpRecon, d_obj_recon)

            # lp_nkr_on =  np.require(np.zeros(shape=(N, K, D), dtype=np.float32),
            #                                         dtype = np.float32, requirements=['C','A'])
            # d_lp_nkr_on = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                    hostbuf = lp_nkr_on.astype(np.float32))

            d_lp_nkr_on = self.d_lp_nkr_on
            d_lp_nk_off = self.d_lp_nk_off

            # d_lp_nk_off = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                    hostbuf = np.require(np.zeros(shape=cur_z.shape, dtype=np.float32),
            #                                         dtype = np.float32, requirements=['C','A']))


            #TODO: this could be made more efficient with local memory and reduction.
            self.prg.calc_zr_lps(self.queue, (N, K, D), None, d_cur_y.data, d_cur_z.data, d_cur_r.data,
                                    d_obj_recon.data, d_lp_nkr_on.data, d_lp_nk_off.data,
                                    self.d_obs.data,
                                    np.int32(N), np.int32(K), np.int32(D), np.int32(self.img_w),
                                    np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

           # nk_rmax = np.require(np.empty(shape = (self.obs.shape[0], cur_z.shape[1]), dtype=np.float32),
                                 #dtype = np.float32, requirements=['C','A'])
            # d_rsums = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                    hostbuf = np.require(
            #                         np.zeros(shape=(self.obs.shape[0], cur_z.shape[1], self.obs.shape[1]), dtype=np.float32),
            #                         dtype = np.float32, requirements=['C','A']))
            d_rsums = clarray.zeros(self.queue,shape=(N,K,D),dtype=np.float32,allocator=self.d_mp)

            #convert to using memory pool
            d_tmpNKRAns = d_lp_nkr_on.copy()
            d_tmpNKRWork = d_lp_nkr_on.copy()

            # d_tmpNKRWork = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                      hostbuf = lp_nkr_on.astype(np.float32))
            # d_tmpNKRAns = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
            #                      hostbuf = lp_nkr_on.astype(np.float32))
            #
            # cl.enqueue_copy(self.queue, d_tmpNKRWork, d_lp_nkr_on)
            # cl.enqueue_copy(self.queue, d_tmpNKRAns, d_lp_nkr_on)

            maxWorkGroupSize = np.int32(self.device.max_work_group_size/2)
            maxLocalMem = self.device.local_mem_size
            #TODO: rewrite for float4s

            workGroupSize = int(min(maxLocalMem//np.dtype(np.float32).itemsize, maxWorkGroupSize))
            tmpLocMemNP = np.empty(workGroupSize,dtype=np.float32)
            d_locmem = cl.LocalMemory(tmpLocMemNP.nbytes)

            curNumToReduce = max(D, workGroupSize)

            #this is due to the weird pyopencl/intel cpu bug where # of workgroups must be a multiple of workgroup size
            #no still doesn't work
            #num_d_workers = int(math.ceil(D/workGroupSize) * workGroupSize)

            num_d_workers = workGroupSize
            #
            #

            num_loop = int(math.ceil(D/workGroupSize))
            # this is a reduction step to get the rmaxes

            while (curNumToReduce > workGroupSize):
                for i in np.arange(num_loop):
                    self.prg.reduc_vec_max3d(self.queue, (num_d_workers,N,K), (workGroupSize,1,1),
                                             d_locmem, d_tmpNKRWork.data, d_tmpNKRAns.data,
                                             np.int32(N), np.int32(K), np.int32(D), np.int32(i*num_d_workers))
                d_OldTmpNKRWork = d_tmpNKRWork
                d_tmpNKRWork = d_tmpNKRAns
                d_tmpNKRAns = d_OldTmpNKRWork
                curNumToReduce = curNumToReduce // workGroupSize

            self.prg.finish_reduc_vec_max3d(self.queue, (workGroupSize, N, K), (workGroupSize, 1, 1),
                                            d_locmem, d_tmpNKRWork.data, d_tmpNKRAns.data,
                                            np.int32(N), np.int32(K), np.int32(D),
                                            np.int32(min(D,curNumToReduce)))


            d_lp_nk_rmax = d_tmpNKRAns
           # blah = d_lp_nk_rmax.get()
            d_rsums = self.d_rsums
            #this does the logtrick
            self.prg.rz_do_log_trick(self.queue, (N,K,D), None,
                                    d_lp_nkr_on.data, d_lp_nk_rmax.data, d_rsums.data,
                                    np.int32(N),np.int32(K), np.int32(D))

            d_rsums_work = d_rsums.copy()

            #now reduce via sum to get the norm. constant


            while curNumToReduce > workGroupSize:
                 for i in np.arange(num_loop):
                     self.prg.reduc_vec_sum3d(self.queue, (num_d_workers,N,K),
                                      (workGroupSize,1,1),
                                      d_locmem, d_rsums.data, d_rsums_work.data,
                                      np.int32(N), np.int32(K), np.int32(D), np.int32(i*num_d_workers))
          #
                 curNumToReduce = curNumToReduce // workGroupSize
                 if curNumToReduce > workGroupSize:
                     d_old_rsums = d_rsums
                     d_rsums = d_rsums_work
                     d_rsums_work = d_old_rsums
          # #  quit()
          #   print("starting complete kernel")
            self.prg.finish_reduc_vec_sum3d(self.queue, (workGroupSize,N,K),
                                             (workGroupSize,1,1),
                                             d_locmem, d_rsums_work.data, d_rsums.data,
                                             np.int32(N), np.int32(K), np.int32(D),
                                             np.int32(min(D,curNumToReduce)))

          #   cl.enqueue_copy(self.queue, tmpRet, d_tmpNKRAns)
          #   r_sums = np.require(tmpRet[:,:,0].copy(), dtype=np.float32, requirements=['C','A'])
          #   d_rsums = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
          #                           hostbuf = r_sums.astype(np.float32))
            rand_vals_z = np.require(np.random.random(size = (N,K)),
                                   dtype=np.float32, requirements=['C','A'])
            d_rand_z = clarray.Array(self.queue,shape=(N,K),dtype=np.float32, allocator=self.d_mp)
            d_rand_z.set(rand_vals_z)

            rand_vals_r = np.require(np.random.random(size = (N,K)),
                                   dtype=np.float32, requirements=['C','A'])
            d_rand_r = clarray.Array(self.queue,shape=(N,K),dtype=np.float32, allocator=self.d_mp)
            d_rand_r.set(rand_vals_r)

            self.prg.sample_zr_precalc(self.queue, (N,K), None,
                                       d_cur_z.data,d_cur_r.data, d_lp_nk_rmax.data, d_rsums.data,
                                       d_lp_nkr_on.data, d_lp_nk_off.data, d_z_col_sum.data,
                                       d_rand_z.data,d_rand_r.data,
                                       np.int32(N), np.int32(K), np.int32(D), np.float32(self.T),
                                       np.int32(self.img_w))


            # #TODO: include a local memory so the shared normalization only needs to be loaded once from global memory, should be an easy-sh speedup, but need to make sure work group sizes are right...
            # self.prg.sample_rz_noscOvRP4b(self.queue, (N,K,D), None,
            #                         d_cur_y, d_cur_z, d_cur_r, d_obj_recon, d_lp_nkr_on, d_lp_nk_off,
            #                         d_lp_nk_rmax, d_rsums, d_z_col_sum, self.d_obs,
            #                         d_rand_z, d_rand_r, np.int32(cur_z.shape[0]),
            #                         np.int32(cur_y.shape[1]), np.int32(cur_y.shape[0]), np.int32(self.img_w),
            #                          np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        #    tmpRSums = np.require(np.zeros(shape=(N,K,D), dtype=np.float32), dtype=np.float32, requirements=['C','A'])
        #    cl.enqueue_copy(self.queue, tmpRSums, d_rsums)
            #= np.zeros(shape= (cur_z.shape[0], cur_z.shape[1], cur_y.shape[1]), dtype=np.float32)
            #l.enqueue_copy(self.queue, tmpNKROn, d_lp_nkr_on)

           # self.prg.sample_rz_noscOvRP5(self.queue, (cur_z.shape[0))


            # self.prg.sample_rz_noscOvRP5(self.queue, (cur_z.shape[0], cur_z.shape[1]), None,
            #                         d_cur_y, d_cur_z, d_cur_r, d_obj_recon, d_lp_nkr_on, d_lp_nk_off,
            #                         d_lp_nk_rmax, d_rsums, d_z_col_sum, self.d_obs,
            #                         d_rand_z, d_rand_r, np.int32(cur_z.shape[0]),
            #                         np.int32(cur_y.shape[1]), np.int32(cur_y.shape[0]), np.int32(self.img_w),
            #                          np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

            self.gpu_time += time() - a_time
            #for debuggin gonly:\
            # _cl
            return d_cur_z, d_cur_r

    def _cl_infer_z_nosc(self, cur_y, cur_z, cur_r, d_cur_y, d_cur_z, d_cur_r):
        """Infer feature ownership
        """
        a_time = time()
        d_z_by_ry = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                              hostbuf = np.empty(shape = self.obs.shape, dtype = np.int32))
        d_z_col_sum = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                hostbuf = cur_z.sum(axis = 0).astype(np.int32))
        d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                           hostbuf = np.random.random(cur_z.shape).astype(np.float32))
        transformed_y = np.empty(shape = (self.obs.shape[0], cur_z.shape[1], self.obs.shape[1]), dtype = np.int32)
        d_transformed_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = transformed_y)
        d_temp_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = transformed_y)


        # first transform the feature images and calculate z_by_ry
        self.prg.compute_z_by_ry_nosc(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r, d_transformed_y, d_temp_y, d_z_by_ry,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # calculate the prior probability that a pixel is on
        self.prg.sample_z_nosc(self.queue, cur_z.shape, None,
                          d_cur_y, d_cur_z, d_cur_r, d_z_by_ry, d_z_col_sum, self.d_obs, d_rand,
                          np.int32(self.N), np.int32(self.d), np.int32(cur_y.shape[0]), np.int32(self.img_w),
                          np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        self.gpu_time += time() - a_time
        return d_cur_z

    def _cl_infer_k_new(self, cur_y, cur_z, cur_r):

        # sample new features use importance sampling
        k_new = self._sample_k_new(cur_y, cur_z, cur_r)
        if k_new:
            cur_y, cur_z, cur_r = k_new

        # delete empty feature images
        non_empty_feat_img = np.where(cur_y.sum(axis = 1) > 0)
        cur_y = cur_y[non_empty_feat_img[0],:].astype(np.int32)
        cur_z = cur_z[:,non_empty_feat_img[0]].astype(np.int32)
        cur_r = np.array([_[non_empty_feat_img[0],:] for _ in cur_r]).astype(np.int32)
        
        # delete null features
        active_feat_col = np.where(cur_z.sum(axis = 0) > 0)
        cur_z = cur_z[:,active_feat_col[0]].astype(np.int32)
        cur_y = cur_y[active_feat_col[0],:].astype(np.int32)
        cur_r = np.array([_[active_feat_col[0],:] for _ in cur_r]).astype(np.int32)

        # update self.k
        self.k = cur_z.shape[1]

        z_s0, z_s1 = cur_z.shape
        cur_z = cur_z.reshape((z_s0 * z_s1, 1))
        cur_z = cur_z.reshape((z_s0, z_s1))

        y_s0, y_s1 = cur_y.shape
        cur_y = cur_y.reshape((y_s0 * y_s1, 1))
        cur_y = cur_y.reshape((y_s0, y_s1))

        r_s0, r_s1, r_s2 = cur_r.shape
        cur_r = cur_r.reshape((r_s0 * r_s1 * r_s2, 1))
        cur_r = cur_r.reshape((r_s0, r_s1, r_s2))

        return cur_y, cur_z, cur_r
    def convert_sample_to_host(self, sample):
        BaseSampler.convert_sample_to_host(self,sample)
        if self.cl_mode:
            d_cur_y, d_cur_z, d_cur_r = sample
            cur_y = d_cur_y.get()
            cur_z = d_cur_z.get()
            cur_r = d_cur_r.get()
            return (cur_y,cur_z,cur_r)
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
            d_cur_y, d_cur_z, d_cur_r = sample

            #for debugging on aug 14 2015
            #cl.enqueue_copy(self.queue, self.tmp_y, d_cur_y)
            #cl.enqueue_copy(self.queue, self.tmp_z, d_cur_z)

            N = self.obs.shape[0]
            D = self.obs.shape[1]
            K = self.k

            d_z_col_sum = clarray.zeros(self.queue,(K,1),dtype=np.int32,allocator=self.d_mp)
            d_obj_recon = self.d_obj_recon.fill(0)

            self.prg.compute_recon_objs_transzsum(self.queue, (N,K,D), None,
                                        d_cur_y.data, d_cur_z.data, d_cur_r.data, d_obj_recon.data, d_z_col_sum.data,
                                        np.int32(N), np.int32(K), np.int32(D), np.int32(self.img_w))

            # d_lps = self.d_recon_lps
            #TODO: really need to fill?
            d_lps = self.d_recon_lps.fill(0.)
            self.prg.calc_lps(self.queue, (N,D), None,
                              d_obj_recon.data, self.d_obs.data, d_lps.data,
                              np.int32(N), np.int32(D), np.float32(self.lam), np.float32(self.epislon))

            lpX = clarray.sum(d_lps).get()

            mks = d_z_col_sum.get()
                # np.require(np.zeros(shape = (K,1), dtype=np.int32),
                #              dtype = np.int32, requirements=['C','A'])

            # cl.enqueue_copy(self.queue, mks, d_z_col_sum)
            # cl.enqueue_copy(self.queue, lpX, d_sum1d)
            cur_z = d_cur_z.map_to_host(is_blocking=True)

            #for debugging
            # cur_y = np.require(np.zeros(shape = (K,D), dtype=np.int32),
            #                    dtype = np.int32, requirements=['C','A'])
            # obj_recon = np.require(np.zeros(shape = (K,D), dtype=np.int32),
            #                    dtype = np.int32, requirements=['C','A'])
            # cl.enqueue_copy(self.queue, cur_y, d_cur_y)
            # cl.enqueue_copy(self.queue,obj_recon,d_obj_recon)

            zT = np.require(cur_z.T, requirements=['C','A'])

            _,cts = np.unique(zT.view(np.dtype((np.void, zT.dtype.itemsize *zT.shape[1]))), return_counts=True)
            mk_mask = np.where(mks > 0)
            lpZ = np.sum(gammaln(mks[mk_mask])+gammaln(N-mks[mk_mask]+1)-gammaln(N+1)) -np.sum(gammaln(cts))

            return lpZ + lpX
        else:
            print("non-cl not supported at the moment")
            return 0
        return 0

    def _cl_infer_k_newJLA2(self, cur_y, cur_z, cur_r):
        """Written by Joe Austerweil on 07/20/15
            Infer new features (bounded by self.k_max_new). Cleans (removes unused features) first.
            Only one object can get new features per run
            NOTE: same as IBP implementation, but needs to update cur_r too
        :param cur_y: feature images on host
        :param cur_z: feature ownership on host
        :param cur_r: feature transformations on host
        :return: d_cur_y, d_cur_z, d_cur_r"""
        a_time = time()

        startK = self.k
        trans_inds = np.arange(self.NUM_TRANS)

        # delete empty feature images
        non_empty_feat_img = np.where(cur_y.sum(axis = 1) > 0)
        cur_y = cur_y[non_empty_feat_img[0],:].astype(np.int32)
        cur_z = cur_z[:,non_empty_feat_img[0]].astype(np.int32)
        r_good_feat_imgs = np.repeat(non_empty_feat_img[0],self.NUM_TRANS) \
                         + np.tile(trans_inds, non_empty_feat_img[0].shape[0])
        cur_r = cur_r[:,r_good_feat_imgs].astype(np.int32)

        # delete null features
        active_feat_col = np.where(cur_z.sum(axis = 0) > 0)
        cur_z = cur_z[:,active_feat_col[0]].astype(np.int32)
        cur_y = cur_y[active_feat_col[0],:].astype(np.int32)


        r_active_inds = np.repeat(active_feat_col[0],self.NUM_TRANS) \
                        + np.tile(trans_inds, active_feat_col[0].shape[0])
        #cur_r = cur_r[:,active_feat_col[0]].astype(np.int32)
        cur_r = cur_r[:,r_active_inds].astype(np.int32)
        z_s0, z_s1 = cur_z.shape
        cur_z = cur_z.reshape((z_s0 * z_s1, 1))
        cur_z = np.require(cur_z.reshape((z_s0, z_s1)), dtype=np.int32, requirements=['C','A'])

        y_s0, y_s1 = cur_y.shape
        cur_y = cur_y.reshape((y_s0 * y_s1, 1))
        cur_y = np.require(cur_y.reshape((y_s0, y_s1)), dtype=np.int32, requirements=['C','A'])

        #TODO: is this part necessary?
        r_s0, r_s1 = cur_r.shape
        cur_r = cur_r.reshape((r_s0 * r_s1, 1))
        cur_r = np.require(cur_r.reshape((r_s0, r_s1)), dtype=np.int32, requirements=['C','A'])
        # update self.k
        self.k = cur_z.shape[1]
        updated_p = False
        d_cur_z = None
        d_cur_y = None
        d_cur_r = None

        if self.k <= (self.max_k - self.k_max_new):
            N = self.obs.shape[0]
            K = cur_z.shape[1]
            D = self.obs.shape[1]

            obj_recon = None

            #self.d_obj_recon=clarray.zeros(self.queue,shape=(N,D),dtype=np.int32,allocator=self.d_mp)
            d_obj_recon = self.d_obj_recon.fill(0)
            # if cur_z.size is 0:
                # obj_recon = np.zeros(self.obs.shape)
            if cur_z.size > 0:
                d_cur_z = clarray.Array(self.queue,shape=(N,K),dtype=np.int32, allocator=self.d_mp)
                d_cur_z.set(ary=cur_z)
                d_cur_y = clarray.Array(self.queue,shape=(K,D),dtype=np.int32, allocator=self.d_mp)
                d_cur_y.set(ary=cur_y)
                d_cur_r= clarray.Array(self.queue,shape=(N,K*self.NUM_TRANS),dtype=np.int32, allocator=self.d_mp)
                d_cur_r.set(ary=cur_r)

                self.prg.compute_recon_objs_trans(self.queue, (N, K, D), None,
                                                  d_cur_y.data, d_cur_z.data, d_cur_r.data,
                                                  d_obj_recon.data, np.int32(N), np.int32(K), np.int32(D),
                                                  np.int32(self.img_w))

                # self.prg.compute_recon_objs(self.queue, (N, K, D), None,
                #                         d_cur_y.data, d_cur_z.data, d_obj_recon.data,
                #                         np.int32(N), np.int32(D), np.int32(K))


            # lps = np.require(np.empty((N,D,self.k_max_new+1), dtype=np.float32),
            #                  dtype=np.float32, requirements=['C','A'])
            # d_lps = cl.Buffer(self.ctx, self.mf.READ_WRITE,size=(N*D*(self.k_max_new+1)*np.dtype('float32').itemsize))
            #d_lps = clarray.Array(self.queue,shape=(N,D,self.k_max_new+1),dtype=np.float32)
            d_lps = self.d_knew_lp
            # self.prg.init_bufferFlt(self.queue, (N*D*(self.k_max_new+1),1), None, d_lps,
            #                         np.int32(N*D*(self.k_max_new+1)), np.float32(0.))
            #                  hostbuf=lps.astype(np.float32))
            #TODO: localify k_max_new to share some values
            self.prg.calc_lp_fornew(self.queue, (N, D, self.k_max_new+1), None,
                                    d_obj_recon.data, self.d_obs.data, d_lps.data,
                                    np.int32(N), np.int32(K), np.int32(D), np.int32(self.k_max_new+1),
                                    np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

            # cl.enqueue_copy(self.queue, lps, d_lps)
            lps = d_lps.get()
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

                cur_r_new = np.zeros(shape=(N,new_k*self.NUM_TRANS),dtype=np.int32)
                cur_r = np.require(np.hstack((cur_r,cur_r_new)), dtype=np.int32,
                                   requirements=['C', 'A'])


                oldK = K
                K += new_k

                self.k = K

                d_cur_z= clarray.Array(self.queue, shape = cur_z.shape, dtype=np.int32, allocator=self.d_mp)
                d_cur_z.set(ary=cur_z)
                d_cur_y= clarray.Array(self.queue, shape = cur_y.shape, dtype=np.int32, allocator=self.d_mp)
                d_cur_y.set(ary=cur_y)
                d_cur_r = clarray.Array(self.queue, shape=cur_r.shape, dtype=np.int32, allocator=self.d_mp)
                d_cur_r.set(ary=cur_r)
                # d_new_z.set(ary=new_z)
                # d_cur_z = clarray.concatenate((d_cur_z, d_new_z), axis=1)

                # d_new_y = clarray.zeros(self.queue,shape=(new_k,D),dtype=np.int32)
                # d_cur_y = clarray.concatenate((d_cur_y, d_new_y),axis= 0)

                # d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                #                     hostbuf = cur_z.astype(np.int32))
                # d_cur_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                #                     hostbuf = cur_y.astype(np.int32))
                # d_knews = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR,
                #                     hostbuf = knews.astype(np.int32))
                #for ensuring z & y are proper size later, but only reallocating as necessary
                self.tmp_z = cur_z
                self.tmp_y = cur_y
                self.tmp_r = cur_r

                comb_vec = np.require(comb(new_k, np.arange(new_k+1)),
                                      dtype=np.int32, requirements=['C','A'])
                d_comb_vec = clarray.Array(self.queue,shape=(new_k+1,1),dtype=np.int32,strides=comb_vec.strides,
                                           allocator=self.d_mp)
                d_comb_vec.set(ary=comb_vec)
                # d_comb_vec = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                #                        hostbuf=comb_vec.astype(np.int32))
                # new_y_val_probs = np.require(np.empty(shape=(new_k+1,D), dtype=np.float32),
                #                              dtype=np.float32, requirements=['C','A'])
                # d_new_y_val_probs = cl.Buffer(self.ctx, self.mf.READ_WRITE,
                #                               size=((new_k+1)*D*np.dtype('float32').itemsize))

                d_new_y_val_probs = clarray.Array(self.queue,shape=(new_k+1,D),dtype=np.float32, allocator=self.d_mp)
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
                                          d_cur_z.data, d_cur_y.data,  d_comb_vec.data,
                                          d_obj_recon.data, self.d_obs.data,
                                          d_new_y_val_probs.data, np.int32(new_k_ind), np.int32(new_k),
                                          np.int32(N), np.int32(D), np.int32(K),
                                          np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))


                #cl.enqueue_copy(self.queue, new_y_val_probs, d_new_y_val_probs)

                y_rand_vals = np.require(np.random.random((D)), dtype=np.float32,
                                     requirements=['C', 'A'])
                # d_y_rand_vals = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                #                           hostbuf=y_rand_vals.astype(np.float32))

                d_y_rand_vals = clarray.Array(self.queue,shape=y_rand_vals.shape,dtype=np.float32,
                                              allocator=self.d_mp)
                d_y_rand_vals.set(y_rand_vals)
                #d_y_rand_vals = clrand(self.queue,(D,1), dtype=np.float32)
                tmpLocMemNPFlt = np.empty(workGroupSize,dtype=np.float32)
                d_locmemFlt = cl.LocalMemory(tmpLocMemNPFlt.nbytes)

                self.prg.y_new_samp2(self.queue, (new_k+1,1,D), (new_k+1,1,1), d_locmemFlt,
                                     d_new_y_val_probs.data,  d_cur_y.data,  d_y_rand_vals.data,
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
                if self.k is not startK:
                    self._update_allocs()
                    updated_p = True

                d_cur_z, d_cur_r = self._cl_infer_zr_nosc(d_cur_y, d_cur_z, d_cur_r)
        else:
            print("warning: reached upper limit %d on features for device. cannot create new" % self.max_k)
            d_cur_z = clarray.Array(self.queue,shape=(N,K),dtype=np.int32, allocator=self.d_mp)
            d_cur_z.set(ary=cur_z)
            d_cur_y = clarray.Array(self.queue,shape=(K,D),dtype=np.int32, allocator=self.d_mp)
            d_cur_y.set(ary=cur_y)
            d_cur_r = clarray.Array(self.queue,shape=(N,K*self.NUM_TRANS),dtype=np.int32,allocator=self.d_mp)
            d_cur_r.set(ary=cur_r)

        #TODO: make so all reallocation steps happen here!
        if self.k is not startK and updated_p is False:
            self._update_allocs()

        self.gpu_time += time() - a_time

        return d_cur_y, d_cur_z, d_cur_r

    def _cl_infer_r(self, cur_y, cur_z, cur_r, d_cur_y, d_cur_z, d_cur_r):
        """Infer transformations using opencl.
        Note: the algorithm works because resampling one value of cur_r at one time
        only affects the loglikelihood of the corresponding image. Therefore, it is
        possible to resample one aspect of transformation for all images at the same
        time, as long as the new values are accepted / rejected independently of
        each other.
        """
        a_time = time()
        d_z_by_ry_old = cl.array.empty(self.queue, self.obs.shape, np.int32, allocator=self.mem_pool)
        d_z_by_ry_new = cl.array.empty(self.queue, self.obs.shape, np.int32, allocator=self.mem_pool)
        d_replace_r = cl.array.empty(self.queue, (self.N,), np.int32, allocator=self.mem_pool)
        d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                           hostbuf=np.random.random(self.N).astype(np.float32))
        transformed_y = np.empty(shape = (self.obs.shape[0], cur_z.shape[1], self.obs.shape[1]), dtype = np.int32)
        d_transformed_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = transformed_y)
        d_temp_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = transformed_y)

        ########### Dealing with vertical translations first ##########
        d_cur_r = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_r.astype(np.int32))

        # calculate the z_by_ry_old under old transformations
        self.prg.compute_z_by_ry(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r, d_transformed_y, d_temp_y, d_z_by_ry_old.data,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # calculate the z_by_ry_new under new randomly generated transformations
        cur_r_new = np.copy(cur_r)
        cur_r_new[:,:,self.V_TRANS] = np.random.randint(0, self.img_h, size = (cur_r_new.shape[0], cur_r_new.shape[1]))
        d_cur_r_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_r_new.astype(np.int32))

        self.prg.compute_z_by_ry(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r_new, d_transformed_y, d_temp_y, d_z_by_ry_new.data,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # reject or accept newly proposed transformations on a per-object basis
        d_logprior_old = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.log(abs((cur_r[:,:,self.V_TRANS] > 0) - self.phi)).astype(np.float32))
        d_logprior_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.log(abs((cur_r_new[:,:,self.V_TRANS] > 0) - self.phi)).astype(np.float32))

        self.prg.sample_r(self.queue, (self.N, ), None,
                          d_replace_r.data, d_z_by_ry_old.data, d_z_by_ry_new.data,
                          d_logprior_old, d_logprior_new, self.d_obs, d_rand,
                          np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                          np.float32(self.lam), np.float32(self.epislon))

        replace_r = d_replace_r.get()
        cur_r[np.where(replace_r == 1)] = cur_r_new[np.where(replace_r == 1)]

        ########### Dealing with horizontal translations next ##########
        d_cur_r = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_r.astype(np.int32))

        # calculate the z_by_ry_old under old transformations
        self.prg.compute_z_by_ry(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r, d_transformed_y, d_temp_y, d_z_by_ry_old.data,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # calculate the z_by_ry_new under new randomly generated transformations
        cur_r_new = np.copy(cur_r)
        cur_r_new[:,:,self.H_TRANS] = np.random.randint(0, self.img_w, size = (cur_r_new.shape[0], cur_r_new.shape[1]))
        d_cur_r_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_r_new.astype(np.int32))

        self.prg.compute_z_by_ry(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r_new, d_transformed_y, d_temp_y, d_z_by_ry_new.data,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # reject or accept newly proposed transformations on a per-object basis
        d_logprior_old = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.log(abs((cur_r[:,:,self.H_TRANS] > 0) - self.phi)).astype(np.float32))
        d_logprior_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.log(abs((cur_r_new[:,:,self.H_TRANS] > 0) - self.phi)).astype(np.float32))

        self.prg.sample_r(self.queue, (self.N, ), None,
                          d_replace_r.data, d_z_by_ry_old.data, d_z_by_ry_new.data,
                          d_logprior_old, d_logprior_new, self.d_obs, d_rand,
                          np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                          np.float32(self.lam), np.float32(self.epislon))

        replace_r = d_replace_r.get()
        cur_r[np.where(replace_r == 1)] = cur_r_new[np.where(replace_r == 1)]

        ########### Dealing with vertical scaling next ##########
        d_cur_r = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_r.astype(np.int32))

        # calculate the z_by_ry_old under old transformations
        self.prg.compute_z_by_ry(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r_new, d_transformed_y, d_temp_y, d_z_by_ry_old.data,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # calculate the z_by_ry_new under new randomly generated transformations
        cur_r_new = np.copy(cur_r)
        cur_r_new[:,:,self.V_SCALE] = np.random.randint(-self.img_h+2, self.img_h, size = (cur_r_new.shape[0], cur_r_new.shape[1]))
        d_cur_r_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_r_new.astype(np.int32))

        self.prg.compute_z_by_ry(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r_new, d_transformed_y, d_temp_y, d_z_by_ry_new.data,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # reject or accept newly proposed transformations on a per-object basis
        d_logprior_old = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.log(abs((cur_r[:,:,self.V_SCALE] > 0) - self.phi)).astype(np.float32))
        d_logprior_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.log(abs((cur_r_new[:,:,self.V_SCALE] > 0) - self.phi)).astype(np.float32))

        self.prg.sample_r(self.queue, (self.N, ), None,
                          d_replace_r.data, d_z_by_ry_old.data, d_z_by_ry_new.data,
                          d_logprior_old, d_logprior_new, self.d_obs, d_rand,
                          np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                          np.float32(self.lam), np.float32(self.epislon))

        replace_r = d_replace_r.get()
        cur_r[np.where(replace_r == 1)] = cur_r_new[np.where(replace_r == 1)]


        ########### Dealing with horizontal scaling next ##########
        d_cur_r = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_r.astype(np.int32))

        # calculate the z_by_ry_old under old transformations
        self.prg.compute_z_by_ry(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r_new, d_transformed_y, d_temp_y, d_z_by_ry_old.data,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # calculate the z_by_ry_new under new randomly generated transformations
        cur_r_new = np.copy(cur_r)
        cur_r_new[:,:,self.H_SCALE] = np.random.randint(-self.img_w+2, self.img_w, size = (cur_r_new.shape[0], cur_r_new.shape[1]))
        d_cur_r_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_r_new.astype(np.int32))

        self.prg.compute_z_by_ry(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r_new, d_transformed_y, d_temp_y, d_z_by_ry_new.data,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # reject or accept newly proposed transformations on a per-object basis
        d_logprior_old = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.log(abs((cur_r[:,:,self.H_SCALE] > 0) - self.phi)).astype(np.float32))
        d_logprior_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.log(abs((cur_r_new[:,:,self.H_SCALE] > 0) - self.phi)).astype(np.float32))
        self.prg.sample_r(self.queue, (self.N, ), None,
                          d_replace_r.data, d_z_by_ry_old.data, d_z_by_ry_new.data,
                          d_logprior_old, d_logprior_new, self.d_obs, d_rand,
                          np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                          np.float32(self.lam), np.float32(self.epislon))

        replace_r = d_replace_r.get()
        cur_r[np.where(replace_r == 1)] = cur_r_new[np.where(replace_r == 1)]

        self.gpu_time += time() - a_time
        return cur_r


    def _cl_infer_r_nosc(self, cur_y, cur_z, cur_r, d_cur_y, d_cur_z, d_cur_r):
        """Infer transformations using opencl (only translations).
        Note: the algorithm works because resampling one value of cur_r at one time
        only affects the loglikelihood of the corresponding image. Therefore, it is
        possible to resample one aspect of transformation for all images at the same
        time, as long as the new values are accepted / rejected independently of
        each other.
        """
        a_time = time()
        d_z_by_ry_old = cl.array.empty(self.queue, self.obs.shape, np.int32, allocator=self.mem_pool)
        d_z_by_ry_new = cl.array.empty(self.queue, self.obs.shape, np.int32, allocator=self.mem_pool)
        d_replace_r = cl.array.empty(self.queue, (self.N,), np.int32, allocator=self.mem_pool)
        d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                           hostbuf=np.random.random(self.N).astype(np.float32))
        transformed_y = np.empty(shape = (self.obs.shape[0], cur_z.shape[1], self.obs.shape[1]), dtype = np.int32)
        d_transformed_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = transformed_y)
        d_temp_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = transformed_y)

        ########### Dealing with vertical translations first ##########
        d_cur_r = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_r.astype(np.int32))

        # calculate the z_by_ry_old under old transformations
        self.prg.compute_z_by_ry_nosc(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r, d_transformed_y, d_temp_y, d_z_by_ry_old.data,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # calculate the z_by_ry_new under new randomly generated transformations
        cur_r_new = np.copy(cur_r)
        cur_r_new[:,:,self.V_TRANS] = np.random.randint(0, self.img_h, size = (cur_r_new.shape[0], cur_r_new.shape[1]))
        d_cur_r_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_r_new.astype(np.int32))

        self.prg.compute_z_by_ry_nosc(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r_new, d_transformed_y, d_temp_y, d_z_by_ry_new.data,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # reject or accept newly proposed transformations on a per-object basis
        d_logprior_old = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.log(abs((cur_r[:,:,self.V_TRANS] > 0) - self.phi)).astype(np.float32))
        d_logprior_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.log(abs((cur_r_new[:,:,self.V_TRANS] > 0) - self.phi)).astype(np.float32))

        self.prg.sample_r(self.queue, (self.N, ), None,
                          d_replace_r.data, d_z_by_ry_old.data, d_z_by_ry_new.data,
                          d_logprior_old, d_logprior_new, self.d_obs, d_rand,
                          np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                          np.float32(self.lam), np.float32(self.epislon))

        replace_r = d_replace_r.get()
        cur_r[np.where(replace_r == 1)] = cur_r_new[np.where(replace_r == 1)]

        ########### Dealing with horizontal translations next ##########
        d_cur_r = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_r.astype(np.int32))

        # calculate the z_by_ry_old under old transformations
        self.prg.compute_z_by_ry_nosc(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r, d_transformed_y, d_temp_y, d_z_by_ry_old.data,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # calculate the z_by_ry_new under new randomly generated transformations
        cur_r_new = np.copy(cur_r)
        cur_r_new[:,:,self.H_TRANS] = np.random.randint(0, self.img_w, size = (cur_r_new.shape[0], cur_r_new.shape[1]))
        d_cur_r_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf = cur_r_new.astype(np.int32))

        self.prg.compute_z_by_ry_nosc(self.queue, cur_z.shape, (1, cur_z.shape[1]),
                                 d_cur_y, d_cur_z, d_cur_r_new, d_transformed_y, d_temp_y, d_z_by_ry_new.data,
                                 np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                                 np.int32(self.img_w))

        # reject or accept newly proposed transformations on a per-object basis
        d_logprior_old = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.log(abs((cur_r[:,:,self.H_TRANS] > 0) - self.phi)).astype(np.float32))
        d_logprior_new = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR,
                                   hostbuf = np.log(abs((cur_r_new[:,:,self.H_TRANS] > 0) - self.phi)).astype(np.float32))

        self.prg.sample_r(self.queue, (self.N, ), None,
                          d_replace_r.data, d_z_by_ry_old.data, d_z_by_ry_new.data,
                          d_logprior_old, d_logprior_new, self.d_obs, d_rand,
                          np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                          np.float32(self.lam), np.float32(self.epislon))

        replace_r = d_replace_r.get()
        cur_r[np.where(replace_r == 1)] = cur_r_new[np.where(replace_r == 1)]

        self.gpu_time += time() - a_time
        return cur_r

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

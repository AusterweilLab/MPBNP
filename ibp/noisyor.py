#!/usr/bin/env python2
#-*- coding: utf-8 -*-

from __future__ import print_function, division
import sys, os.path, itertools, cPickle
pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../../'
sys.path.append(pkg_dir)

import pyopencl.array
from scipy.stats import poisson
from MPBNP import *
from MPBNP import BaseSampler, BasePredictor

np.set_printoptions(suppress=True)

class Gibbs(BaseSampler):

    def __init__(self, cl_mode = True, cl_device = None, record_best = True,
                 alpha = None, lam = 0.98, theta = 0.1, epislon = 0.02, init_k = 10):
        """Initialize the class.
        """
        BaseSampler.__init__(self, cl_mode = cl_mode, cl_device = cl_device, record_best = record_best)

        if cl_mode:
            program_str = open(pkg_dir + 'MPBNP/ibp/kernels/ibp_noisyor_cl.c', 'r').read()
            self.prg = cl.Program(self.ctx, program_str).build()

        self.alpha = alpha # tendency to generate new features
        self.k = init_k    # initial number of features
        self.theta = theta # prior probability that a pixel is on in a feature image
        self.lam = lam # effecacy of a feature
        self.epislon = epislon # probability that a pixel is on by change in an actual image
        self.samples = {'z': [], 'y': []} # sample storage, to be pickled

    def read_csv(self, filepath, header=True):
        """Read the data from a csv file.
        """
        BaseSampler.read_csv(self, filepath, header)
        # convert the data to floats
        self.new_obs = []
        for row in self.obs:
            self.new_obs.append([int(_) for _ in row])
        self.obs = np.array(self.new_obs)
        self.d = len(self.obs[0])
        self.alpha = self.N
        return

    def direct_read_obs(self, obs):
        BaseSampler.read_csv(self, obs)
        self.d = len(self.obs[0])
        
    def do_inference(self, init_y = None, init_z = None, output_file = None):
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

        if self.cl_mode:
            return self._cl_infer_yz(init_y, init_z, output_file)
        else:
            return self._infer_yz(init_y, init_z, output_file)

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
            elif i >= self.burnin:
                cur_y, cur_z = temp_cur_y, temp_cur_z
                self.samples['z'].append(cur_z)
                self.samples['y'].append(cur_y)

        # delete empty feature images
        #non_empty_feat_img = np.where(cur_y.sum(axis = 1) > 0)
        #cur_y = cur_y[non_empty_feat_img[0],:]
        #cur_z = cur_z[:,non_empty_feat_img[0]]
                    
        if output_file is not None:
            if self.record_best:
                # print out the Y matrix
                final_y, final_z = self.best_sample[0]
                print(final_y, file = output_file)
                print(final_z, file = output_file)
            else:
                cPickle.dump(self.samples, open(output_file, 'w'))

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
        old_loglik = self._loglik(cur_y, cur_z)

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
        d_obs = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, hostbuf=self.obs.astype(np.int32))

        self.auto_save_sample(sample = (cur_y, cur_z))
        for i in xrange(self.niter):
            a_time = time()
            temp_cur_y = self._cl_infer_y(cur_y, cur_z, d_obs)
            temp_cur_z = self._cl_infer_z(temp_cur_y, cur_z, d_obs)
            self.gpu_time += time() - a_time
            temp_cur_y, temp_cur_z = self._cl_infer_k_new(temp_cur_y, temp_cur_z)

            if self.record_best:
                if self.auto_save_sample(sample = (temp_cur_y, temp_cur_z)):
                    cur_y, cur_z = temp_cur_y, temp_cur_z
            elif i >= self.burnin:
                cur_y, cur_z = temp_cur_y, temp_cur_z
                self.samples['z'].append(cur_z)
                self.samples['y'].append(cur_y)

            self.total_time += time() - a_time

        if output_file is not None:
            if self.record_best:
                # print out the Y matrix
                final_y, final_z = self.best_sample[0]
                print(final_y, file = output_file)
                print(final_z, file = output_file)
            else:
                cPickle.dump(self.samples, open(output_file, 'w'))

        return self.gpu_time, self.total_time, None

    def _cl_infer_y(self, cur_y, cur_z, d_obs):
        """Infer feature images
        """
        d_cur_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = cur_y.astype(np.int32))
        d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = cur_z.astype(np.int32))
        d_z_by_y = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, 
                             hostbuf = np.dot(cur_z, cur_y).astype(np.int32))
        d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, 
                           hostbuf=np.random.random(size = cur_y.shape).astype(np.float32))

        # calculate the prior probability that a pixel is on
        self.prg.sample_y(self.queue, cur_y.shape, None,
                          d_cur_y, d_cur_z, d_z_by_y, d_obs,
                          d_rand, #d_y_on_loglik.data, d_y_off_loglik.data,
                          np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_y.shape[0]),
                          np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        cl.enqueue_copy(self.queue, cur_y, d_cur_y)
        return cur_y

    def _cl_infer_z(self, cur_y, cur_z, d_obs):
        """Infer feature ownership
        """
        d_cur_y = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = cur_y.astype(np.int32))
        d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = cur_z.astype(np.int32))
        d_z_by_y = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, 
                             hostbuf = np.dot(cur_z, cur_y).astype(np.int32))
        d_z_col_sum = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, 
                                hostbuf = cur_z.sum(axis = 0).astype(np.int32))
        d_rand = cl.Buffer(self.ctx, self.mf.READ_ONLY | self.mf.COPY_HOST_PTR, 
                           hostbuf=np.random.random(size = cur_z.shape).astype(np.float32))

        # calculate the prior probability that a pixel is on
        self.prg.sample_z(self.queue, cur_z.shape, None,
                          d_cur_y, d_cur_z, d_z_by_y, d_z_col_sum, d_obs, d_rand, 
                          np.int32(self.obs.shape[0]), np.int32(self.obs.shape[1]), np.int32(cur_z.shape[1]),
                          np.float32(self.lam), np.float32(self.epislon), np.float32(self.theta))

        cl.enqueue_copy(self.queue, cur_z, d_cur_z)
        return cur_z
        
    def _cl_infer_k_new(self, cur_y, cur_z):

        # sample new features use importance sampling
        k_new = self._sample_k_new(cur_y, cur_z)
        if k_new:
            cur_y, cur_z = k_new

        # delete null features
        inactive_feat_col = np.where(cur_z.sum(axis = 0) == 0)
        cur_z_new = np.delete(cur_z, inactive_feat_col[0], axis=1).astype(np.int32)
        cur_y_new = np.delete(cur_y, inactive_feat_col[0], axis=0).astype(np.int32)

        z_new_s0, z_new_s1 = cur_z_new.shape
        cur_z_new = cur_z_new.reshape((z_new_s0 * z_new_s1, 1))
        cur_z_new = cur_z_new.reshape((z_new_s0, z_new_s1))

        y_new_s0, y_new_s1 = cur_y_new.shape
        cur_y_new = cur_y_new.reshape((y_new_s0 * y_new_s1, 1))
        cur_y_new = cur_y_new.reshape((y_new_s0, y_new_s1))

        # update self.k
        self.k = cur_z_new.shape[1]
        
        return cur_y_new, cur_z_new

    def _logprob(self, sample):
        """Calculate the joint log probability of data and model given a sample.
        """
        cur_y, cur_z = sample
        log_prior = 0
        log_lik = 0
        if cur_z.shape[1] == 0: return -99999999.9
    
        if self.cl_mode:
            a_time = time()
            d_cur_z = cl.Buffer(self.ctx, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf = cur_z.astype(np.int32))
            d_logprob_z = cl.array.empty(self.queue, cur_z.shape, np.float32)
            d_novel_f = cl.array.empty(self.queue, cur_z.shape, np.int32)

            self.prg.logprob_z(self.queue, cur_z.shape, None, 
                               d_cur_z, d_logprob_z.data, d_novel_f.data,
                               np.int32(self.N), np.int32(cur_z.shape[1]), np.float32(self.alpha))
            log_prior = d_logprob_z.get().sum()
            novel_counts = poisson.logpmf(d_novel_f.get().sum(axis = 1), self.alpha / np.arange(1, cur_z.shape[0]+1))
            log_prior += novel_counts[np.where(d_novel_f.get() > 0)[0]].sum()
            self.gpu_time += time() - a_time

            # calculate the prior probability of Y
            num_on = (cur_y == 1).sum()
            num_off = (cur_y == 0).sum()
            log_prior += num_on * np.log(self.theta) + num_off * np.log(1 - self.theta)
            # calculate the logliklihood
            log_lik = self._loglik(cur_y = cur_y, cur_z = cur_z)

        else:
            # calculate the prior probability of Z
            for n in xrange(cur_z.shape[0]):
                num_novel = 0
                for k in xrange(cur_z.shape[1]):
                    m = cur_z[:n,k].sum()
                    if m > 0:
                        if cur_z[n,k] == 1: log_prior += np.log(m / (n+1))
                        else: log_prior += np.log(1 - m / (n + 1))
                    else: 
                        if cur_z[n,k] == 1: num_novel += 1
                if num_novel > 0:
                    log_prior += poisson.logpmf(num_novel, self.alpha / (n+1))
            # calculate the prior probability of Y
            num_on = (cur_y == 1).sum()
            num_off = (cur_y == 0).sum()
            log_prior += num_on * np.log(self.theta) + num_off * np.log(1 - self.theta)
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

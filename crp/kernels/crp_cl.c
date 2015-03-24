float t_logpdf(float theta, float df, float loc, float scale) {
  float part1 = lgamma((df + 1.0f) / 2.0f) - lgamma(0.5f * df) -  log(pow(df * M_PI_F, 0.5f) * scale);
  float part2 = -0.5f * (df + 1.0f) * log(1.0f + (1.0f / df) * pow((float)(theta - loc) / scale, 2.0f));
  return part1 + part2;
}

float mvt_logpdf(global float *data_vec, int data_i, int dim, float df, 
		 global float *loc_vec, int cluster_i, float det, 
		 global float *inv_mat) {
  float part1 = lgamma((df + dim) / 2.0f) - lgamma(df / 2.0f);
  float part2 = -0.5f * log(det) - 0.5f * dim * log(df * M_PI_F);
  float mat_mul = 0.0f;
  float mat_inner;
  for (int i = 0; i < dim; i++) {
    mat_inner = 0.0f;
    for (int j = 0; j < dim; j++) {
      mat_inner += (data_vec[data_i * dim + j] - loc_vec[cluster_i * dim + j]) * 
	inv_mat[cluster_i * dim * dim + j * dim + i];
    }
    mat_mul += mat_inner * (data_vec[data_i * dim + i] - loc_vec[cluster_i * dim + i]);
  }
  float part3 = -0.5f * (df + dim) * log(1.0f + mat_mul / df);
  return part1 + part2 + part3;
}

float max_arr(global float *arr, int start, int length) {
  float result = arr[start];
  for (int i = start + 1; i < start + length; i++) {
    //if (arr[i] > result) result = arr[i];
    result = fmax(result, arr[i]);
  }
  return result;
}

float min_arr(global float *arr, int start, int length) {
  float result = arr[start];
  for (int i = start + 1; i < start + length; i++) {
    //if (arr[i] < result) result = arr[i];
    result = fmin(result, arr[i]);
  }
  return result;
}

float sum(global float *arr, int start, int length) {
  float result = 0;
  for (int i = start; i < start + length; i++) {
    result += arr[i];
  }
  return result;
}

void lognormalize(global float *logp, int start, int length) {
  float m = max_arr(logp, start, length);
  for (int i = start; i < start + length; i++) {
    logp[i] = powr(exp(1.0f), logp[i] - m);
    // this line is a hack to prevent a weird nVIDIA-only global memory bug
    // !logp is global
    // a simple exp(logp[i] - m) fails to compile on nvidia cards
  }
  float p_sum = sum(logp, start, length);
  for (int i = start; i < start + length; i++) {
    logp[i] = logp[i] / p_sum;
  }
}


uint sample(uint a_size, global uint *a, global float *p, int start, float rand) {
  float total = 0.0f;
  for (int i = start; i < start + a_size; i++) {
    total = total + p[i];
    if (total > rand) return a[i-start];
  }
  return a[a_size - 1];
}

__kernel void normal_1d_logpost(global uint *labels, global float *data, global uint *uniq_label, 
				global float *mu, global float *ss, global uint *n,
				int cluster_num, global float *hyper_param, global float *rand,
				global float *logpost) {
  uint i = get_global_id(0);
  uint c = get_global_id(1);
  uint data_size = get_global_size(0); // total number of data

  float gaussian_mu0 = hyper_param[0];
  float gaussian_k0 = hyper_param[1];
  float gamma_alpha0 = hyper_param[2];
  float gamma_beta0 = hyper_param[3];
  float alpha = hyper_param[4];
  float k_n, mu_n;
  float alpha_n, beta_n;
  float Lambda, sigma;
  float new_size = n[c];

  k_n = gaussian_k0 + new_size;
  mu_n = (gaussian_k0 * gaussian_mu0 + new_size * mu[c]) / k_n;
  alpha_n = gamma_alpha0 + new_size / 2.0f;
  beta_n = gamma_beta0 + 0.5f * ss[c] + gaussian_k0 * new_size * pow((mu[c] - gaussian_mu0), 2.0f) / (2.0f * k_n);

  Lambda = alpha_n * k_n / (beta_n * (k_n + 1.0f));
  sigma = pow(1.0f/Lambda, 0.5f);
  logpost[i * cluster_num + c] = t_logpdf(data[i], 2.0f * alpha_n, mu_n, sigma);
  logpost[i * cluster_num + c] += (new_size > 0) ? 
    log(new_size/(alpha + data_size)) : log(alpha/(alpha + data_size));
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
}

__kernel void normal_1d_logpost_loopy(global uint *labels, global float *data, global uint *uniq_label, 
				global float *mu, global float *ss, global uint *n,
				uint cluster_num, global float *hyper_param, global float *rand,
				global float *logpost) {

  uint i = get_global_id(0);
  uint data_size = get_global_size(0);
  
  float gaussian_mu0 = hyper_param[0];
  float gaussian_k0 = hyper_param[1];
  float gamma_alpha0 = hyper_param[2];
  float gamma_beta0 = hyper_param[3];
  float alpha = hyper_param[4];

  uint old_label = labels[i];
  uint new_label;
  uint new_size;
  uint original_cluster;
  float k_n, mu_n;
  float alpha_n, beta_n;
  float Lambda, sigma;
  uint empty_n = 1;

  for (int c = 0; c < cluster_num; c++) {
    new_label = uniq_label[c];
    new_size = n[c];
    original_cluster = old_label == new_label;
    empty_n += (original_cluster && new_size == 1);
    // compute other variables 
    k_n = gaussian_k0 + new_size;
    mu_n = (gaussian_k0 * gaussian_mu0 + new_size * mu[c]) / k_n;
    alpha_n = gamma_alpha0 + new_size / 2.0f;
    beta_n = gamma_beta0 + 0.5f * ss[c] + gaussian_k0 * new_size * pow((mu[c] - gaussian_mu0), 2.0f) / (2.0f * k_n);
    Lambda = alpha_n * k_n / (beta_n * (k_n + 1.0f));
    sigma = pow(1.0f/Lambda, 0.5f);
    logpost[i * cluster_num + c] = t_logpdf(data[i], 2.0f * alpha_n, mu_n, sigma);
    
    // calculate the log prior
    logpost[i * cluster_num + c] += (new_size > original_cluster) ? 
      log((new_size - original_cluster) / (alpha + data_size-1)):
      log(alpha/(float)empty_n/(alpha + data_size-1));
  }
  
  lognormalize(logpost, i * cluster_num, cluster_num);
  labels[i] = sample(cluster_num, uniq_label, logpost, i * cluster_num, rand[i]);
}

// kernel to compute the joint log probability of data and a given sample (i.e., labels)
__kernel void joint_logprob(global uint *labels, global float *data, 
			    global float *hyper_param, global float *logprob) {

  uint data_pos = get_global_id(0);
  uint label = labels[data_pos];
  
  float gaussian_mu0 = hyper_param[0];
  float gaussian_k0 = hyper_param[1];
  float gamma_alpha0 = hyper_param[2];
  float gamma_beta0 = hyper_param[3];
  float alpha = hyper_param[4];

  float k_n, mu_n;
  float alpha_n, beta_n;
  float Lambda, sigma;
  float sum = 0.0f, mu=0.0f, ss=0.0f;
  int n = 0;

  for (int i = 0; i < data_pos; i++) {
    if (labels[i] == label) {
      n += 1;
      sum += data[i];
    }
  }
  if (n == 0) {
    mu = 0.0f;
    ss = 0.0f;
  }
  else { 
    mu = sum / n;
    for (int i = 0; i < data_pos; i++) {
      if (labels[i] == label) {
	ss += pow(data[i] - mu, 2.0f);
      }
    }
  }

  // compute other variables 
  k_n = gaussian_k0 + n;
  mu_n = (gaussian_k0 * gaussian_mu0 + n * mu) / k_n;
  alpha_n = gamma_alpha0 + n / 2.0f;
  beta_n = gamma_beta0 + 0.5f * ss + gaussian_k0 * n * pow(mu - gaussian_mu0, 2.0f) / (2.0f * k_n);
  Lambda = alpha_n * k_n / (beta_n * (k_n + 1.0f));
  sigma = pow(1.0f/Lambda, 0.5f);
  // compute the joint log probability 
  logprob[data_pos] = t_logpdf(data[data_pos], 2.0f * alpha_n, mu_n, sigma);
  logprob[data_pos] += (n > 0) ? log( (float)n / ((float)data_pos + alpha)) : log(alpha / ((float)data_pos + alpha));
}

__kernel void normal_kd_sigma_matrix(global uint *n, global float *cov_obs, global float *cov_mu0, global float *T, float k0, float v0, global float *sigma) {
  
  uint cluster_i = get_global_id(0); 
  uint dim = get_global_size(1);
  uint d1 = get_global_id(1);
  uint d2 = get_global_id(2);
  uint cluster_size = n[cluster_i];
  uint _3d_increm = dim * dim;
  float k_n, v_n;

  k_n = k0 + cluster_size;
  v_n = v0 + cluster_size;

  sigma[d1 * dim + d2 + cluster_i * _3d_increm] = 
    (T[d1 * dim + d2] + cov_obs[d1 * dim + d2 + cluster_i * _3d_increm] +
     k0 * cluster_size / k_n * cov_mu0[d1 * dim + d2 + cluster_i * _3d_increm]) *
    (k_n + 1) / (k_n * (v_n - dim + 1));
}

__kernel void normal_kd_logpost(global uint *labels, global float *data, global uint *uniq_label, global float *mu, global uint *n,  global float *determinants, global float *inverses, uint cluster_num, float alpha, uint dim, float v0, global float *logpost, global float *rand) {
  
  uint data_size = get_global_size(0);
  uint i = get_global_id(0);
  uint c = get_global_id(1);
  uint new_size = n[c];
  float t_df = v0 + new_size - dim + 1.0f;
 
  float loglik = mvt_logpdf(data, i, dim, //data array, start_index, length
			    t_df,  //degrees of freedom
			    mu, c, //mu array, which cluster
			    determinants[c], inverses);

  loglik += (new_size > 0) ? 
    log(new_size/(alpha + data_size)) : log(alpha/(alpha + data_size));
  logpost[i * cluster_num + c] = loglik;
  /*
  if (c == 0) {
    lognormalize(logpost, i * cluster_num, cluster_num);
    labels[i] = sample(cluster_num, uniq_label, logpost, i * cluster_num, rand[i]);
  }
  */
}


__kernel void resample_labels(global uint *labels, global uint *uniq_label, uint cluster_num, global float *rand, global float *logpost) {

  uint i = get_global_id(0);
  lognormalize(logpost, i * cluster_num, cluster_num);
  //printf("Data %d Before: %d\n", i, labels[i]);
  labels[i] = sample(cluster_num, uniq_label, logpost, i * cluster_num, rand[i]);
  //printf("Data %d After: %d\n", i, labels[i]);
}

__kernel void normal_kd_logpost_loopy(global uint *labels, global float *data, global uint *uniq_label, global float *mu, global uint *n,  global float *determinants, global float *inverses, uint cluster_num, float alpha, uint dim, float v0, global float *logpost, global float *rand) {
  
  uint i = get_global_id(0);
  uint data_size = get_global_size(0);
  uint old_label = labels[i];
  uint new_label;
  uint new_size;
  uint original_cluster;
  float t_df;
  uint empty_n = 1;

  for (int c = 0; c < cluster_num; c++) {
    new_label = uniq_label[c];
    new_size = n[c];
    empty_n += (old_label == new_label && new_size == 1);

    t_df = v0  + new_size - dim + 1.0f;
    logpost[i * cluster_num + c] = mvt_logpdf(data, i, dim, //data array, start_index, length
					      t_df,  //degrees of freedom
					      mu, c, //mu array, which cluster
					      determinants[c], inverses);

    original_cluster = old_label == new_label;
    logpost[i * cluster_num + c] += (new_size > original_cluster) ? 
      log((new_size - original_cluster) / (alpha + data_size-1)) : log(alpha / empty_n / (alpha + data_size-1));
  }
  lognormalize(logpost, i * cluster_num, cluster_num);
  //printf("Data %d Before: %d\n", i, labels[i]);
  labels[i] = sample(cluster_num, uniq_label, logpost, i * cluster_num, rand[i]);
  //printf("Data %d After: %d\n", i, labels[i]);
}

/*
__kernel void get_mu(global uint *labels, global float *data, gloal uint *uniq_label, 
		     volatile global float *sum,// volatile global uint *n, volatile global float *mu,
		     uint cluster_num //global float *hyper_param, global float *rand,
		     ) {
  int gid = get_global_id(0);
  int wid = get_group_id(0);
  
  printf("Group %d, Cluster %d\n", wid, labels[gid]);
  barrier(CLK_GLOBAL_MEM_FENCE);

  sum[labels[gid]] += data[gid];
  barrier(CLK_GLOBAL_MEM_FENCE);
  if (gid == 0) {
    for (int c = 0; c < cluster_num; c++) {
      printf("Cluster %d, sum: %f\n", uniq_label[c], sum[uniq_label[c]]);
    }
  }
}
  
  
*/
/*
__kernel void normal_kd_suf_stats(global uint *uniq_label, global int *labels, global float *data, global float *mu0, global uint *indices, global uint *n, uint data_size, uint num_of_clusters, uint dim, global float *mu, local float *mu0_mu_dev, global float *cov_mu0, global float *obs_mu_dev, global float *cov_obs) {
  
  uint cluster_i = get_global_id(0);
  
  //for (int i = 0; i < data_size; i++) {
  //  n[cluster_i] += (uniq_label[cluster_i] == labels[i]);
  //}
  //barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  uint n_so_far = 0;
  for (int c = 0; c < cluster_i; c++) {
    n_so_far += n[c];
  }
  
  for (int i = 0; i < n[cluster_i]; i++) {
    for (int d = 0; d < dim; d++) {
      mu[cluster_i * dim + d] += data[indices[n_so_far + i] * dim + d] / n[cluster_i];
    }
  }
  
  // compute the deviance from mu0
  for (int d1 = 0; d1 < dim; d1++) {
    mu0_mu_dev[cluster_i * dim + d1] = mu0[d1] - mu[cluster_i * dim + d1];
  }
  // compute the deviances from each data point
  for (int i = 0; i < n[cluster_i]; i++) {
    for (int d1 = 0; d1 < dim; d1++) {
      obs_mu_dev[indices[n_so_far + i] * dim + d1] = 
	data[indices[n_so_far + i] * dim + d1] - mu[cluster_i * dim + d1];
    }
  }
  
  //barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  // compute cov_mu0
  //float target_cell, target_cell2;
  for (int d1 = 0; d1 < dim; d1++) {
    for (int d2 = 0; d2 < dim; d2++) {
      cov_mu0[cluster_i * dim * dim + d1 * dim + d2] = 
	mu0_mu_dev[cluster_i * dim + d1] * mu0_mu_dev[cluster_i * dim + d2];
      // tricky: calculate cov_obs
      cov_obs[cluster_i * dim * dim + d1 * dim + d2] = 0;
      for (int j = 0; j < n[cluster_i]; j++) {
      	cov_obs[cluster_i * dim * dim + d1 * dim + d2] += obs_mu_dev[indices[n_so_far + j] * dim + d2] * obs_mu_dev[indices[n_so_far + j] * dim + d1];
      }
    }
  }
}
*/

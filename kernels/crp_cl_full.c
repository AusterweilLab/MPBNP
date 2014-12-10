float t_logpdf(float theta, float df, float loc, float scale) {
  float part1 = lgamma((df + 1) / 2.0f) - lgamma(0.5f * df) -  log(pow(df * M_PI_F, 0.5f) * scale);
  float part2 = -0.5f * (df + 1) * log(1 + (1 / df) * pow((theta - loc) / scale, 2.0f));
  return part1 + part2;
}

float max_arr(global float *arr, int start, int length) {
  float result = arr[start];
  for (uint i = start + 1; i < start + length; i++) {
    if (arr[i] > result) result = arr[i];
  }
  return result;
}

float sum(global float *arr, int start, int length) {
  float result = 0;
  for (uint i = start; i < start + length; i++) {
    result += arr[i];
  }
  return result;
}

void lognormalize(global float *logp, global float *post, int data_i, int cluster_size) {
  int start = data_i * cluster_size;
  int length = cluster_size;
  //float m = max_arr(logp, start, length);
  for (int j = start; j < start + length; j++) {
    //printf("%f\n", m);
    post[j] = exp(logp[j]); // - m);
  }
  float p_sum = sum(post, start, length);
  for (int i = start; i < start + length; i++) {
    post[i] = post[i] / p_sum;
  }
}

int sample(uint a_size, global uint *a, global float *p, int start, float rand) {
  float total = 0;
  for (uint i = start; i < start + a_size; i++) {
    total += p[i];
    if (total > rand) return a[i-start];
  }
  return a[a_size - 1];
}


__kernel void normal_1d_logpost(global uint *labels, global float *data, global uint *uniq_label, 
				global float *mu, global float *ss, global uint *n,
				int cluster_num, global float *hyper_param, global float *rand,
				global uint *empty_n, global float *logpost, global float *post) {

  int i = get_global_id(0);
  uint data_i = i / cluster_num;
  uint cluster_i = i % cluster_num;
  size_t data_num = get_global_size(0) / cluster_num; // total number of data

  float gaussian_mu0 = hyper_param[0];
  float gaussian_k0 = hyper_param[1];
  float gamma_alpha0 = hyper_param[2];
  float gamma_beta0 = hyper_param[3];
  float alpha = hyper_param[4];
  float k_n, mu_n;
  float alpha_n, beta_n;
  float Lambda, sigma;

  if (n[cluster_i] == 0) empty_n[data_i]++;

  if (labels[data_i] == uniq_label[cluster_i] && n[cluster_i] == 1) {
    empty_n[data_i]++;
    k_n = gaussian_k0;
    mu_n = gaussian_mu0;
    alpha_n = gamma_alpha0;
    beta_n = gamma_beta0;
  } else {
    k_n = gaussian_k0 + n[cluster_i];
    mu_n = (gaussian_k0 * gaussian_mu0 + n[cluster_i] * mu[cluster_i]) / k_n;
    alpha_n = gamma_alpha0 + n[cluster_i] / 2.0f;
    beta_n = gamma_beta0 + 0.5f * ss[cluster_i] + gaussian_k0 * n[cluster_i] * pow((mu[cluster_i] - gaussian_mu0), 2.0f) / (2.0f * k_n);
  }

  Lambda = alpha_n * k_n / (beta_n * (k_n + 1.0f));
  sigma = pow(1.0f/Lambda, 0.5f);
  logpost[i] = t_logpdf(data[data_i], 2.0f * alpha_n, mu_n, sigma);
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

  if (labels[data_i] == uniq_label[cluster_i]) {
    logpost[i] += (n[cluster_i] == 1) ? log(alpha/(float)empty_n[data_i]) : log(n[cluster_i]-1.0f);
  } else {
    logpost[i] += (n[cluster_i] > 0) ? log((float)n[cluster_i]) : log(alpha/(float)empty_n[data_i]);
  }
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);

  if (cluster_i == 0) {
    lognormalize(logpost, post, data_i, cluster_num);
    labels[data_i] = sample(cluster_num, uniq_label, post, data_i * cluster_num, rand[data_i]);
  }
}


float t_logpdf(float theta, float df, float loc, float scale) {
  float part1 = lgamma((df + 1) / 2.0) - lgamma(0.5 * df) -  log(pow(df * M_PI_F, (float)0.5) * scale);
  float part2 = -0.5 * (df + 1) * log(1 + (1 / df) * pow((theta - loc) / scale, (float)2.0));
  return part1 + part2;
}

__kernel void loglikelihood(int obs_index, global int *cluster_labels, constant int *cluster_sizes, constant float *cluster_contents, constant float *hyper_param, __global float *loglik) {
  int i = get_global_id(0);
  if (cluster_labels[i] > -1) {
    int n = cluster_sizes[i];
    float gaussian_mu0 = hyper_param[0];
    float gaussian_k0 = hyper_param[1];
    float gamma_alpha0 = hyper_param[2];
    float gamma_beta0 = hyper_param[3];
    float sum = 0;
    float ss = 0;
    float y_bar = 0;
    /* compute sample mean y_bar */
    for (int j = i; j < i + cluster_sizes[i]; j++) {
      if (j == obs_index) {
	n = n - 1;
	if (n == 0) {
	  cluster_labels[i] = -1;
	  return;
	}
      } else {
	sum += cluster_contents[j];
      }
    }
    if (n > 0) { // this makes sure the novel cluster won't be run here
      y_bar = sum / n;
    }
    /* compute the sum of squares */
    for (int j = i; j < i + cluster_sizes[i]; j++) {
      if (j != obs_index) ss += pow(cluster_contents[j] - y_bar, (float)2.0);
    }

    /* compute other variables */
    float k_n = gaussian_k0 + n;
    float mu_n = (gaussian_k0 * gaussian_mu0 + n * y_bar) / k_n;
    float alpha_n = gamma_alpha0 + n / 2.0;
    float beta_n = gamma_beta0 + 0.5 * ss + gaussian_k0 * n * pow((y_bar - gaussian_mu0), (float)2.0) / (2.0 * k_n);
    float Lambda = alpha_n * k_n / (beta_n * (k_n + 1));
    float sigma = pow(1/Lambda, (float)0.5);
    loglik[i] = t_logpdf(cluster_contents[obs_index], 2.0 * alpha_n, mu_n, sigma);
    //printf("Thread %d, Cluster %d, n: %d, Mean: %f, df: %f, beta_n: %f, mu_n: %f, sigma: %f, loglik: %f\n", i, cluster_labels[i], n, y_bar, 2 * alpha_n, beta_n, mu_n, sigma, loglik[i]);
  }
}

__kernel void logprior(int obs_index, float total_n, global int *cluster_labels, constant int *cluster_sizes, float alpha,  __global float *logprior) {
  int i = get_global_id(0);
  int n = cluster_sizes[i];
  int old_group = 0;
  if (cluster_labels[i] > -1) {
    if (obs_index >= i && obs_index < i + cluster_sizes[i]) {
      old_group = 1;
      n = n - 1;
    }
    if (old_group == 1 & n == 0) {
      cluster_labels[i] = -1;
      return;
    }
    if (old_group == 0 & n == 0) {
      logprior[i] = log(alpha / (total_n + alpha));
    } else {
      logprior[i] = log(n / (total_n + alpha));
    }
    //printf("Thread %d, Cluster %d, logprior: %f\n", i, cluster_labels[i], logprior[i]);
  }
}

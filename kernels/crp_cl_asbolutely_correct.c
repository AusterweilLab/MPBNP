float t_logpdf(float theta, float df, float loc, float scale) {
  float part1 = lgamma((df + 1) / 2.0f) - lgamma(0.5f * df) -  log(pow(df * M_PI_F, 0.5f) * scale);
  float part2 = -0.5f * (df + 1) * log(1 + (1 / df) * pow((theta - loc) / scale, 2.0f));
  return part1 + part2;
}

float max_arr(global float *arr, int start, int length) {
  float result = arr[start];
  for (int i = start + 1; i < start + length; i++) {
    if (arr[i] > result) result = arr[i];
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
    logp[i] = exp(logp[i] - m); //logp no longer holds log values any more
  }
  float p_sum = sum(logp, start, length);
  for (int i = start; i < start + length; i++) {
    logp[i] = logp[i] / p_sum;
  }
  //return logp;
}

int sample(uint a_size, global uint *a, global float *p, int start, float rand) {
  float total = 0;
  for (int i = start; i < start + a_size; i++) {
    total += p[i];
    if (total > rand) return a[i-start];
  }
  return a[a_size - 1];
}

__kernel void logpost(global uint *labels, global float *data, global uint *uniq_label, 
		      uint cluster_num, global float *hyper_param, global float *rand,
		      global float *logpost) {
  
  int i = get_global_id(0);
  size_t data_num = get_global_size(0); // total number of data
 

  float gaussian_mu0 = hyper_param[0];
  float gaussian_k0 = hyper_param[1];
  float gamma_alpha0 = hyper_param[2];
  float gamma_beta0 = hyper_param[3];
  float alpha = hyper_param[4];
  float k_n, mu_n;
  float alpha_n, beta_n;
  float Lambda, sigma;
  float sum, mu, ss;
  uint n;
  uint empty_n = 0; // a trick for making sure correctness: count of tables with no customers

  for (int c = 0; c < cluster_num; c++) {
    n = 0; mu = 0; ss = 0; sum = 0;
    /* compute the mu of this group */
    for (int d = 0; d < data_num; d++) {
      if (labels[d] == uniq_label[c] && d!=i) {
	n++;
	sum += data[d];
      }
    }
    if (n > 0) {
      mu = sum / n;
    } else empty_n++;

    /* compute the ss of this group */
    for (int d = 0; d < data_num; d++) {
      if (labels[d] == uniq_label[c] && d!=i) {
	ss += pow(data[d] - mu, 2.0f);
      }
    }
    
    /* compute other variables */
    k_n = gaussian_k0 + n;
    mu_n = (gaussian_k0 * gaussian_mu0 + n * mu) / k_n;
    alpha_n = gamma_alpha0 + n / 2.0f;
    beta_n = gamma_beta0 + 0.5f * ss + gaussian_k0 * n * pow((mu - gaussian_mu0), 2.0f) / (2.0f * k_n);
    Lambda = alpha_n * k_n / (beta_n * (k_n + 1));
    sigma = pow(1.0f/Lambda, 0.5f);
    logpost[i * cluster_num + c] = t_logpdf(data[i], 2.0f * alpha_n, mu_n, sigma);
    /*if (i == 20) {
      printf("%f -- target cluster %d, mu: %f, ss: %f, n: %d, loglik: %f\n", 
	     data[i], uniq_label[c], sum, ss, n, logpost[i * cluster_num + c]);
	     }*/
  }

  for (int c = 0; c < cluster_num; c++) {
    n = 0; 
    /* compute the mu of this group */
    for (int d = 0; d < data_num; d++) {
      if (labels[d] == uniq_label[c] && d!=i) {
	n++;
      }
    }
    logpost[i * cluster_num + c] += (n > 0) ? log((float)n) : log(alpha/(float)empty_n);
    /*if (i == 20) {
      printf("%f -- target cluster %d, empty_n %d, logpost %f, n: %d\n", 
	     data[i], uniq_label[c], empty_n, logpost[i * cluster_num + c], n);
	     }*/
  }

  /*
  for (int c = 0; c < cluster_num; c++) {
    if (i == 20) {printf("unnormalized: %f\n", logpost[i * cluster_num + c]);}
  }
  */
  lognormalize(logpost, i * cluster_num, cluster_num);
  /*
  for (int c = 0; c < cluster_num; c++) {
    if (i == 20) {
      printf("cluster %d normalized: %f\n", uniq_label[c], logpost[i * cluster_num + c]);   
    }
  }
  */

  //uint old_sample = labels[i];
  labels[i] = sample(cluster_num, uniq_label, logpost, i * cluster_num, rand[i]);
  //if (i == 20) printf("rand %f, new sample %d\n", rand[i], labels[i]);
}

/*
__kernel void normal_kd_sigma_matrix(global uint *labels, constant uint *uniq_label, constant uint *n, constant float *ss, 
				     constant float *T, constant float *cov_mu, 
				     global float *sigma_2, uint dim, uint cluster_num, float k, float v) {
  //float alpha, constant float *rand, global uint *empty_n, global float *logpost) {
  
  uint i = get_global_id(0);
  uint data_i = i / cluster_num;
  uint cluster_i = i % cluster_num;
  size_t data_num = get_global_size(0) / cluster_num; // total number of data
  
  float k_n, v_n;

  if (labels[data_i] == uniq_label[cluster_i] && n[cluster_i] == 1) {
    k_n = k;
    v_n = v;
    for (int d1 = 0; d1 < dim; d1++) {
      for (int d2 = 0; d2 < dim; d2++) {
	sigma_2[d1 * d2 + d2] = (T[d1 * d2 + d2] + ss[d1 * d2 + d2]) * (k_n + 1) / (k_n * (v_n - dim + 1));
      }
    }
  } else {
    k_n = k + n[cluster_i];
    v_n = v + n[cluster_i];
    for (int d1 = 0; d1 < dim; d1++) {
      for (int d2 = 0; d2 < dim; d2++) {
	sigma_2[d1 * d2 + d2] = (T[d1 * d2 + d2] + ss[d1 * d2 + d2] + k * n[cluster_i] / (k + n[cluster_i]) * cov_mu[d1 * d2 + d2]) *
	  (k_n + 1) / (k_n * (v_n - dim + 1));
      }
    }
  }
}
*/
/*
__kernel void normal_kd_logpost(global uint *labels, constant float *data, constant uint *uniq_label, 
				constant float *mu, constant float *ss, constant float *T, 
				constant uint *n, constant float *mu_0, constant float *cov_mu, 
				global float *df; global float *mu_n; global float *sigma_2;
				uint dim, uint cluster_num, float k, float v) {
  //float alpha, constant float *rand, global uint *empty_n, global float *logpost) {
  
  uint i = get_global_id(0);
  uint data_i = i / cluster_num;
  uint cluster_i = i % cluster_num;
  size_t data_num = get_global_size(0) / cluster_num; // total number of data
  
  float k_n, v_n;
  //float alpha_n, beta_n;
  //float sigma_2;

  if (n[cluster_i] == 0) empty_n[data_i]++;

  // compute other variables 
  if (labels[data_i] == uniq_label[cluster_i] && n[cluster_i] == 1) {
    empty_n[data_i]++;
    k_n = k;
    v_n = v;
    for (int d = 0; d < dim; d++) {
      mu_n[cluster_i * d + d] = mu_0[cluster_i * d + d];
    }
    for (int d1 = 0; d1 < dim; d1++) {
      for (int d2 = 0; d2 < dim; d2++) {
	//T_n[d1 * d2 + d2] = T[d1 * d2 + d2] + ss[d1 * d2 + d2];
	sigma_2[d1 * d2 + d2] = (T[d1 * d2 + d2] + ss[d1 * d2 + d2]) * (k_n + 1) / (k_n * (v_n - dim + 1));
      }
    }
  } else {
    k_n = k + n[cluster_i];
    v_n = v + n[cluster_i];
    for (int d = 0; d < dim; d++) {
      mu_n[d] = (k * mu_0[d] + n[cluster_i] * mu[cluster_i]) / k_n;
    }
    for (int d1 = 0; d1 < dim; d1++) {
      for (int d2 = 0; d2 < dim; d2++) {
	//T_n[d1 * d2 + d2] = T[d1 * d2 + d2] + ss[d1 * d2 + d2] + k * n[cluster_i] / (k + n[cluster_i]) * cov_mu[d1 * d2 + d2];
	sigma_2[d1 * d2 + d2] = (T[d1 * d2 + d2] + ss[d1 * d2 + d2] + k * n[cluster_i] / (k + n[cluster_i]) * cov_mu[d1 * d2 + d2]) *
	  (k_n + 1) / (k_n * (v_n - dim + 1))
      }
    }
  }
  df = v_n - dim + 1;
}

  logpost[i] = multi_t_logpdf(*data, data_i, dim, df, *mu_n, *sigma_2);
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  
  /*
    if (data_i == 12) {
    printf("%f -- target cluster %d, mu: %f, ss: %f, n: %d, loglik: %f\n", 
    data[data_i], uniq_label[cluster_i], mu[cluster_i], ss[cluster_i], n[cluster_i], logpost[i]);
    }
  if (data_i == 12) {
  //if (labels[data_i] == uniq_label[cluster_i]) {
    printf("%f -- target cluster %d, label %d,  mu %f, ss %f, empty_n %d, loglik %f, n: %d\n", 
	   data[data_i], mu[cluster_i], labels[data_i], ss[cluster_i], uniq_label[cluster_i], empty_n[data_i], logpost[i], n[cluster_i]);
    //printf("cluster %d normalized: %f\n", uniq_label[c], logpost[data_i * cluster_num + c]);   
  }
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  */
/*
  if (labels[data_i] == uniq_label[cluster_i]) {
    logpost[i] += (n[cluster_i] == 1) ? log(alpha/(float)empty_n[data_i]) : log(n[cluster_i]-1.0f);
  } else {
    logpost[i] += (n[cluster_i] > 0) ? log((float)n[cluster_i]) : log(alpha/(float)empty_n[data_i]);
  }

  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  /*
  if (data_i == 12) {
    printf("%f -- target cluster %d, empty_n %d, logpost %f, n: %d\n", 
	   data[data_i], uniq_label[cluster_i], empty_n, logpost[i], n[cluster_i]);
  }
  
  barrier(CLK_GLOBAL_MEM_FENCE);
  
  if (cluster_i == 0) {
    
    for (int c = 0; c < cluster_num; c++) {
      if (data_i == 12) {printf("unnormalized: %f\n", logpost[data_i * cluster_num + c]);}
    }
  }
  
  if (data_i == 12) {
  //if (labels[data_i] == uniq_label[cluster_i]) {
    printf("%f -- target cluster %d, label %d,  mu %f, ss %f, empty_n %d, logpost %f, n: %d\n", 
	   data[data_i], mu[cluster_i], labels[data_i], ss[cluster_i], uniq_label[cluster_i], empty_n[data_i], logpost[i], n[cluster_i]);
    //printf("cluster %d normalized: %f\n", uniq_label[c], logpost[data_i * cluster_num + c]);   
  }
  barrier(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE);
  */
/*
  if (cluster_i == 0) {
    lognormalize(logpost, data_i * cluster_num, cluster_num);
    //uint old_sample = labels[data_i];
    labels[data_i] = sample(cluster_num, uniq_label, logpost, data_i * cluster_num, rand[data_i]);
    //if (data_i == 12) printf("rand %f, new sample %d\n", rand[data_i], labels[data_i]);
  }
}
  
*/

float lfactorial (int x) {
  float lfac = 0;
  for (int i = 2; i <= x; i++) {
    lfac += log((float)i);
  }
  return lfac;
}

float pois_logpmf(int k, float lambda) {
  return k * log(lambda) - lambda - lfactorial(k);
}

float max_arr(float *arr, int start, int length) {
  float result = arr[start];
  for (int i = start + 1; i < start + length; i++) {
    //if (arr[i] > result) result = arr[i];
    result = fmax(result, arr[i]);
  }
  return result;
}

float min_arr(float *arr, int start, int length) {
  float result = arr[start];
  for (int i = start + 1; i < start + length; i++) {
    //if (arr[i] < result) result = arr[i];
    result = fmin(result, arr[i]);
  }
  return result;
}

float sum(float *arr, int start, int length) {
  float result = 0;
  for (int i = start; i < start + length; i++) {
    result += arr[i];
  }
  return result;
}

void lognormalize(float *logp, int start, int length) {
  float m = max_arr(logp, start, length);
  for (int i = start; i < start + length; i++) {
    logp[i] = native_powr(exp(1.0f), logp[i] - m);
  }
  float p_sum = sum(logp, start, length);
  for (int i = start; i < start + length; i++) {
    logp[i] = logp[i] / p_sum;
  }
}

void pnormalize(float *p, int start, int length) {
  float p_sum = sum(p, start, length);
  for (int i = start; i < start + length; i++) {
    p[i] = p[i] / p_sum;
  }
}

int sample(uint a_size,  uint *a, float *p, int start, float rand) {
  float total = 0;
  for (int i = start; i < start + a_size; i++) {
    total += p[i];
    if (total > rand) return a[i-start];
  }
  return a[a_size - 1];
}

void v_translate(local int *orig_y, local int *new_y, uint kth,
		 uint f_img_height, uint f_img_width, uint distance,
		 uint D) {
  for (int h = 0; h < f_img_height; h++) {
    for (int w = 0; w < f_img_width; w++) {
      new_y[kth * D + ((h + distance) % f_img_height) * f_img_width + w] = 
	orig_y[kth * D + h * f_img_width + w];
    }
  }
}

void h_translate(local int *orig_y, local int *new_y, uint kth,
		 uint f_img_height, uint f_img_width, uint distance,
		 uint D) {
  for (int h = 0; h < f_img_height; h++) {
    for (int w = 0; w < f_img_width; w++) {
      new_y[kth * D + h * f_img_width + (w + distance) % f_img_width] = 
	orig_y[kth * D + h * f_img_width + w];
    }
  }
}

kernel void compute_z_by_ry(global int *cur_y, global int *cur_z, global int *cur_r,
			    global int *z_by_ry, local int *orig_y, local int *new_y,
			    uint N, uint D, uint K, uint f_img_width) {

  const uint V_TRANS = 0, H_TRANS = 1, NUM_TRANS = 2;
  uint nth = get_global_id(0); // nth is the index of images
  uint kth = get_global_id(1); // kth is the index of features
  uint f_img_height = D / f_img_width;

  // copy the original feature image to local memory
  for (int dth = 0; dth < D; dth++) {
    orig_y[kth * D + dth] = cur_y[kth * D + dth];
  }
  // wait until copying is done
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 

  // vertically translate the feature image
  uint v_dist = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + V_TRANS];
  v_translate(orig_y, new_y, kth, f_img_height, f_img_width, v_dist, D);
  
  // copy new_y back to orig_y so that new_y can be used again
  for (int dth = 0; dth < D; dth++) {
    orig_y[kth * D + dth] = new_y[kth * D + dth];
  }

  // wait until copying is done
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 

  // horizontally translate the feature image
  uint h_dist = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + H_TRANS];
  h_translate(orig_y, new_y, kth, f_img_height, f_img_width, h_dist, D);

  // wait until all transformation is done
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 

  /* at this point, for each image, a transformed y (new_y) has been generated */
  if (kth == 0) {
    for (int k = 0; k < K; k++) {
      for (int dth = 0; dth < D; dth++) {
	z_by_ry[nth * D + dth] += new_y[k * D + dth] * cur_z[nth * K + k];
      }
    }
  }
}

kernel void sample_y(global int *cur_y,
		     global int *cur_z,
		     global int *z_by_ry,
		     global int *cur_r,
		     global int *obs,
		     global float *rand, 
		     uint N, uint D, uint K, uint f_img_width,
		     float lambda, float epislon, float theta) {
  
  const uint V_TRANS = 0, H_TRANS = 1, NUM_TRANS = 2;

  uint kth = get_global_id(0); // k is the index of features
  uint dth = get_global_id(1); // d is the index of pixels

  uint f_img_height = D / f_img_width;

  // unpack dth into h and w
  uint h = dth / f_img_width;
  uint w = dth % f_img_width;

  // calculate the prior probability of each cell is 1
  float on_loglik_temp = log(theta); 
  float off_loglik_temp = log(1 - theta);
  
  // extremely hackish way to calculate the loglikelihood
  for (int n = 0; n < N; n++) {
    // if the nth object has the kth feature
    if (cur_z[n * K + kth] == 1) {
      // retrieve the transformation applied to this feature by this object
      uint v_dist = cur_r[n * (K * NUM_TRANS) + kth * NUM_TRANS + V_TRANS];
      uint h_dist = cur_r[n * (K * NUM_TRANS) + kth * NUM_TRANS + H_TRANS];
      uint new_index = ((v_dist + h) % f_img_height) * f_img_width + (h_dist + w) % f_img_width;

      // if the observed pixel at dth is on
      if (obs[n * D + new_index] == 1) { // transformed feature affects the pixel at new_index not dth, cf., ibp
	// if the feature image previously has this pixel on
	if (cur_y[kth * D + dth] == 1) { // this is dth instead of new_index because we are referring to the original y
	  on_loglik_temp += log(1 - pow(1 - lambda, z_by_ry[n * D + new_index]) * (1 - epislon));
	  off_loglik_temp += log(1 - pow(1 - lambda, z_by_ry[n * D + new_index] - 1) * (1 - epislon));
	} else {
	  on_loglik_temp += log(1 - pow(1 - lambda, z_by_ry[n * D + new_index] + 1) * (1 - epislon));
	  off_loglik_temp += log(1 - pow(1 - lambda, z_by_ry[n * D + new_index]) * (1 - epislon));
	}
      } else {
	on_loglik_temp += log(1 - lambda);
	off_loglik_temp += log(1.0f);
      }
    } 
  }
  float logpost[2] = {on_loglik_temp, off_loglik_temp};
  //printf("%f %f %d \n", logpost[0], logpost[1], cur_y[kth * D + dth]);
  uint labels[2] = {1, 0};
  lognormalize(logpost, 0, 2);
  cur_y[kth * D + dth] = sample(2, labels, logpost, 0, rand[kth * D + dth]);
  //printf("%f %f %d \n", logpost[0], logpost[1], cur_y[kth * D + dth]);
}

kernel void sample_z(global int *cur_y,
		     global int *cur_z,
		     global int *cur_r,
		     global int *z_by_ry,
		     global int *z_col_sum,
		     global int *obs,
		     global float *rand, 
		     uint N, uint D, uint K, uint f_img_width,
		     float lambda, float epislon, float theta) {
  
  const uint V_TRANS = 0, H_TRANS = 1, NUM_TRANS = 2;
  uint h, w, new_index; // variables used in the for loop

  uint nth = get_global_id(0); // n is the index of data
  uint kth = get_global_id(1); // k is the index of features

  uint f_img_height = D / f_img_width;

  // calculate the prior probability of each cell is 1
  float on_prob_temp = (z_col_sum[kth] - cur_z[nth * K + kth]) / (float)N; 
  float off_prob_temp = 1 - (z_col_sum[kth] - cur_z[nth * K + kth]) / (float)N;

  // retrieve the transformation applied to this feature by this object
  uint v_dist = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + V_TRANS];
  uint h_dist = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + H_TRANS];
  
  // extremely hackish way to calculate the probelihood
  for (int d = 0; d < D; d++) {
    // if the kth feature can turn on a pixel at d
    if (cur_y[kth * D + d] == 1) {
      // unpack d into h and w and get new index
      h = d / f_img_width;
      w = d % f_img_width;
      new_index = ((v_dist + h) % f_img_height) * f_img_width + (h_dist + w) % f_img_width;
      
      // then the corresponding observed pixel is at new_index
      // so, if the observed pixel at new_index is on
      if (obs[nth * D + new_index] == 1) {
	// if the nth object previously has the kth feature
	if (cur_z[nth * K + kth] == 1) {
	  on_prob_temp *= 1 - pow(1 - lambda, z_by_ry[nth * D + new_index]) * (1 - epislon);
	  off_prob_temp *= 1 - pow(1 - lambda, z_by_ry[nth * D + new_index] - 1) * (1 - epislon);
	} else {
	  on_prob_temp *= 1 - pow(1 - lambda, z_by_ry[nth * D + new_index] + 1) * (1 - epislon);
	  off_prob_temp *= 1 - pow(1 - lambda, z_by_ry[nth * D + new_index]) * (1 - epislon);
	}
      } else {
	on_prob_temp *= 1 - lambda;
	off_prob_temp *= 1.0f;
      }
    } 
  }
  
  //printf("index: %d post_on: %f post_off: %f\n", nth * K + kth, on_prob_temp, off_prob_temp);
  float post[2] = {on_prob_temp, off_prob_temp};
  uint labels[2] = {1, 0};
  pnormalize(post, 0, 2);
  //printf("before index: %d %f %f %d \n", nth * K + kth, post[0], post[1], cur_z[nth * K + kth]);
  cur_z[nth * K + kth] = sample(2, labels, post, 0, rand[nth * K + kth]);
  //printf("after index: %d %f %f %d \n", nth * K + kth, post[0], post[1], cur_z[nth * K + kth]);
}

     
kernel void logprob_z_data(global int *cur_z,
			   global int *cur_y,
			   global int *obs,
			   global float *logprob,
			   uint N, uint D, uint K,
			   float alpha, float lambda, float epislon) {

  uint nth = get_global_id(0); // n is the index of data
  uint m;
  uint novel_count = 0;
  float logprob_temp = 0;

  /* calculate the log probability of the nth row of Z 
     i.e., the prior probability of having the features
     of the nth object.
   */
  for (int k = 0; k < K; k++) {
    m = 0;
    for (int n = 0; n < nth; n++) {
      m += cur_z[n * K + k];
    }
    if (m > 0) { // if other objects have had this feature
      if (cur_z[nth * K + k] == 1) {
	logprob_temp += log(m / (nth + 1.0f));
      } else {
	logprob_temp += log(1 - m / (nth + 1.0f));
      }
    } else { // if this is a novel feature
      if (cur_z[nth * K + k] == 1) novel_count += 1;
    }
  }
  if (novel_count > 0) {
    logprob_temp += pois_logpmf(novel_count, alpha / (nth+1.0f));
  }

  /* calculate the log-likelihood of the nth row of data
     given the corresponding row in Z and Y
  */
  uint weight;
  for (int d = 0; d < D; d++) {
    weight = 0;
    for (int k = 0; k < K; k++) {
      weight += cur_y[k * D + d] * cur_z[nth * K + k];
    }
    if (obs[nth * D + d] == 1) {
      logprob_temp += log(1 - pow(1 - lambda, weight) * (1 - epislon));
    } else {
      logprob_temp += weight * log(1 - lambda) + log(1 - epislon);
    }
  }
  logprob[nth] = logprob_temp;
}

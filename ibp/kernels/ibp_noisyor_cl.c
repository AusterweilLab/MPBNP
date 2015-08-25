// written by Joe Austerweil and Ting Qian

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
    logp[i] = pow(exp(1.0f), logp[i] - m);
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

kernel void init_bufferFlt(global float* data, uint len, float val) {
  int idx = get_global_id(0);

  if (idx < convert_int(len)) {
    data[idx] = val;
  }
}

kernel void init_bufferInt(global int* data, uint len, int val) {
  int idx = get_global_id(0);

  if (idx < convert_int(len)) {
    data[idx] = val;
  }

}


kernel void init_2bufferFlt(global float* data1, global float* data2, uint len, float val) {
  int idx = get_global_id(0);

  if (idx < convert_int(len)) {
    data1[idx] = val;
    data2[idx] = val;
  }
}

kernel void init_2bufferInt(global int* data1, global int* data2, uint len, int val) {
  int idx = get_global_id(0);

  if (idx < convert_int(len)) {
    data1[idx] = val;
    data2[idx] = val;
  }
}

kernel void sample_yJLA(local float* locmem, global int* cur_y, global int *cur_z, global int* cur_recon,
			global int* obs, global float*rand, uint N, uint D, uint K,
			float lambda, float epsilon, float theta) {
  int nth = get_global_id(0);
  int kth = get_global_id(1);
  int dth = get_global_id(2);
  int lid = get_local_id(0);
  
  int obs_ind = nth*D + dth;
  int my_y_ind = kth*D+dth;
  
  int my_obs_val = obs[obs_ind];
  int my_recon = cur_recon[obs_ind];
  int my_z_val = cur_z[nth*K+kth];
  
  if (lid == 0) {
    locmem[0] = (float) cur_y[my_y_ind];
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  int my_y_val = (int) locmem[0];
  
  float pow_val = native_powr(1-lambda, my_recon) * (1-epsilon);
  float pow_val_less1 = native_powr(1-lambda, max(my_recon-1,0)) * (1-epsilon);
  float pow_val_plus1 = native_powr(1-lambda, my_recon+1) * (1-epsilon);

  // lpOff
  locmem[1+lid] = my_z_val * my_y_val * my_obs_val * log(1-pow_val_less1)
    + my_z_val * my_y_val * (1-my_obs_val) * log(pow_val_less1)
    + my_z_val * (1-my_y_val)*my_obs_val *log(1-pow_val)
    + my_z_val * (1-my_y_val) * (1-my_obs_val) * log(pow_val)
    + (1-my_z_val) * my_obs_val * log(1-pow_val)
    + (1-my_z_val) * (1-my_obs_val) * log(pow_val);
  // + N for lpOn
  locmem[1+lid+N] = my_z_val * my_y_val * my_obs_val * log(1-pow_val)
    + my_z_val * my_y_val * (1-my_obs_val) * log(pow_val)
    + my_z_val * (1-my_y_val) * my_obs_val * log(1-pow_val_plus1)
    + my_z_val * (1-my_y_val) * (1-my_obs_val) * log(pow_val_plus1)
    + (1-my_z_val) * my_obs_val * log(1-pow_val)
    + (1-my_z_val) * (1-my_obs_val) * log(pow_val);
  barrier(CLK_LOCAL_MEM_FENCE);
  float lpOffSum = 0;
  float lpOnSum = 0;
  // TODO: reduce rather than for loop
  if (lid==0) {
    for (int i = 0; i < N; i++) {
      lpOffSum +=locmem[i+1];
      lpOnSum += locmem[i+N+1];
    }
    float logpost[2] = {lpOnSum, lpOffSum};
    uint labels[2] = {1, 0};
    lognormalize(logpost, 0, 2);
    cur_y[my_y_ind] = sample(2, labels, logpost, 0, rand[my_y_ind]);
  }

}

kernel void calc_y_lps_old(local float* locmem, global int* cur_y, global int *cur_z, global int* cur_recon,
		       global int* obs, global float* lp_off, global float* lp_on,
		       uint N, uint D, uint K, uint numPrevRun,
		       float lambda, float epsilon, float theta) {
  int nth = get_global_id(0) + numPrevRun;
  int numRun = get_global_size(0);
  int kth = get_global_id(1);
  int dth = get_global_id(2);
  int lid = get_local_id(0);

  int obs_ind = nth*D + dth;
  int my_y_ind = kth*D+dth;

  int my_obs_val = obs[obs_ind];
  int my_recon = cur_recon[obs_ind];
  int my_z_val = cur_z[nth*K+kth];
  //int my_obs_val = 1;
  //int my_recon = 0;
  //int my_z_val = 1;

  //TODO: see if it's faster to share global or do lpOn in parallel
  //TODO: Also, look whether it's best to make group_size one smaller
  //if (lid == 0) {
  //  locmem[0] = (float) cur_y[my_y_ind];
  //}
  //barrier(CLK_LOCAL_MEM_FENCE);

  int my_y_val =cur_y[my_y_ind];
  //int my_y_val = cur_y[my_y_ind];
  float pow_val = native_powr(1-lambda, my_recon) * (1-epsilon);
  float pow_val_less1 = native_powr(1-lambda, max(my_recon-1,0)) * (1-epsilon);
  float pow_val_plus1 = native_powr(1-lambda, my_recon+1) * (1-epsilon);

  // lpOff
  locmem[lid] = my_z_val * my_y_val * my_obs_val * log(1-pow_val_less1)
    + my_z_val * my_y_val * (1-my_obs_val) * log(pow_val_less1)
    + my_z_val * (1-my_y_val) * my_obs_val * log(1-pow_val)
    + my_z_val * (1-my_y_val) * (1-my_obs_val) * log(pow_val)
    + (1-my_z_val) * my_obs_val * log(1-pow_val)
    + (1-my_z_val) * (1-my_obs_val) * log(pow_val);

  // + numRun for lpOn
  locmem[lid+numRun] = my_z_val * my_y_val * my_obs_val * log(1-pow_val)
    + my_z_val * my_y_val * (1-my_obs_val) * log(pow_val)
    + my_z_val * (1-my_y_val) * my_obs_val * log(1-pow_val_plus1)
    + my_z_val * (1-my_y_val) * (1-my_obs_val) * log(pow_val_plus1)
    + (1-my_z_val) * my_obs_val * log(1-pow_val)
    + (1-my_z_val) * (1-my_obs_val) * log(pow_val);
  //printf("n: %d; k: %d; d: %d; z: %d, y: %d, x: %d; recon: %d; lpOff: %f; lpOn: %f\n",
  //  nth,kth,dth,my_z_val, my_y_val, my_obs_val, my_recon, locmem[1+lid], locmem[1+lid+numRun]);

  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid==0) {
    // TODO: should reduce instead of for loop
    float lpOffSum = log(1-theta);
    float lpOnSum = log(theta);

    for (int i = 0; i < numRun; i++) {
      lpOffSum +=locmem[i];
      lpOnSum += locmem[i+numRun];
    }

    //lpOffSum = locmem[3];


    //printf("my_y_ind %d; nth: %d, kth: %d, dth: %d; numRun %d:\n",
    //       my_y_ind, nth, kth, dth, numRun);
    //printf("nth: %d, kth: %d, dth: %d, cur_lp_off %f; lpOffSum: %f; cur_lp_on %f; lpOnSum: %f\n",
    //       nth, kth, dth,lp_off[my_y_ind], lpOffSum, lp_on[my_y_ind], lpOnSum);
    lp_off[my_y_ind] = lp_off[my_y_ind] + lpOffSum;
    lp_on[my_y_ind] = lp_on[my_y_ind] + lpOnSum;
  }
}

//TODO: understand why barriers break with printfs in this function... checks ok with a non-local version though...
//TODO: check if this is fixed after oclgrind checks...
kernel void calc_y_lps(local float* locmem, global int* cur_y, global int *cur_z, global int* cur_recon,
		       global int* obs, global float* lp_off, global float* lp_on,
		       uint N, uint D, uint K, uint numPrevRun,
		       float lambda, float epsilon, float theta) {
  int nth = get_global_id(0) + numPrevRun;
  int numRun = get_global_size(0);
  int kth = get_global_id(1);
  int dth = get_global_id(2);
  int lid = get_local_id(0);


  int obs_ind = nth*D + dth;
  int my_y_ind = kth*D+dth;
  if (obs_ind < N) {
    int my_obs_val = obs[obs_ind];
    int my_recon = cur_recon[obs_ind];
    int my_z_val = cur_z[nth*K+kth];
    //int my_obs_val = 1;
    //int my_recon = 0;
    //int my_z_val = 1;

    //TODO: see if it's faster to share global or do lpOn in parallel
    //TODO: Also, look whether it's best to make group_size one smaller
    //if (lid == 0) {
    //  locmem[0] = (float) cur_y[my_y_ind];
    //}
    //barrier(CLK_LOCAL_MEM_FENCE);

    int my_y_val =cur_y[my_y_ind];
    //int my_y_val = cur_y[my_y_ind];
    float pow_val = native_powr(1-lambda, my_recon) * (1-epsilon);
    float pow_val_less1 = native_powr(1-lambda, max(my_recon-1,0)) * (1-epsilon);
    float pow_val_plus1 = native_powr(1-lambda, my_recon+1) * (1-epsilon);

    // lpOff
    locmem[lid] = my_z_val * my_y_val * my_obs_val * log(1-pow_val_less1)
      + my_z_val * my_y_val * (1-my_obs_val) * log(pow_val_less1)
      + my_z_val * (1-my_y_val) * my_obs_val * log(1-pow_val)
      + my_z_val * (1-my_y_val) * (1-my_obs_val) * log(pow_val)
      + (1-my_z_val) * my_obs_val * log(1-pow_val)
      + (1-my_z_val) * (1-my_obs_val) * log(pow_val);

    // + numRun for lpOn
    locmem[lid+numRun] = my_z_val * my_y_val * my_obs_val * log(1-pow_val)
      + my_z_val * my_y_val * (1-my_obs_val) * log(pow_val)
      + my_z_val * (1-my_y_val) * my_obs_val * log(1-pow_val_plus1)
      + my_z_val * (1-my_y_val) * (1-my_obs_val) * log(pow_val_plus1)
      + (1-my_z_val) * my_obs_val * log(1-pow_val)
      + (1-my_z_val) * (1-my_obs_val) * log(pow_val);
    //printf("setting local memory - lid+numRun: %d for %d,%d,%d\n", lid+numRun, nth,kth,dth);
    //printf("n: %d; k: %d; d: %d; z: %d, y: %d, x: %d; recon: %d; lpOff: %f; lpOn: %f\n",
    //  nth,kth,dth,my_z_val, my_y_val, my_obs_val, my_recon, locmem[1+lid], locmem[1+lid+numRun]);
  }
  barrier(CLK_LOCAL_MEM_FENCE );
  if ((lid==0) && (obs_ind < N)) {
    // TODO: should reduce instead of for loop
    float lpOffSum = log(1-theta);
    float lpOnSum = log(theta);
    int numLoop = min(convert_int(N)-convert_int(numPrevRun),numRun);
    //printf("numLoop: %d; N: %d; numPrevRun: %d; numRun: %d\n",
      //     numLoop, N, numPrevRun, numRun);
    for (int i = 0; i < numLoop; i++) {
      //printf("in loop: i+numRun: %d for %d, %d, %d\n", i+numRun, nth,kth,dth);
      lpOffSum +=locmem[i];
      lpOnSum += locmem[i+numRun];
    }

    //lpOffSum = locmem[3];


    //printf("my_y_ind %d; nth: %d, kth: %d, dth: %d; numRun %d:\n",
    //       my_y_ind, nth, kth, dth, numRun);
    //printf("nth: %d, kth: %d, dth: %d, cur_lp_off %f; lpOffSum: %f; cur_lp_on %f; lpOnSum: %f\n",
    //       nth, kth, dth,lp_off[my_y_ind], lpOffSum, lp_on[my_y_ind], lpOnSum);
    lp_off[my_y_ind] = lp_off[my_y_ind] + lpOffSum;
    lp_on[my_y_ind] = lp_on[my_y_ind] + lpOnSum;
  }
}


//TODO: understand why barriers break with printfs in this function... checks ok with a non-local version though...
//TODO: check if this is fixed after oclgrind checks...
kernel void calc_y_lp_off(local float* locmem, global int* cur_y, global int *cur_z, global int* cur_recon,
		       global int* obs, global float* lp_off, 
		       uint N, uint D, uint K, uint numPrevRun,
		       float lambda, float epsilon, float theta) {
  int nth = get_global_id(0) + numPrevRun;
  int numRun = get_global_size(0);
  int kth = get_global_id(1);
  int dth = get_global_id(2);
  int lid = get_local_id(0);


  int obs_ind = nth*D + dth;

  int my_y_ind = kth*D+dth;
  if (nth < N) {
  
  int my_obs_val = obs[obs_ind];
  int my_recon = cur_recon[obs_ind];
  int my_z_val = cur_z[nth*K+kth];


  int my_y_val =cur_y[my_y_ind];

  float pow_val = native_powr(1-lambda, my_recon) * (1-epsilon);
  float pow_val_less1 = native_powr(1-lambda, max(my_recon-1,0)) * (1-epsilon);
  float pow_val_plus1 = native_powr(1-lambda, my_recon+1) * (1-epsilon);

  // lpOff
  locmem[lid] = my_z_val * my_y_val * my_obs_val * log(1-pow_val_less1)
    + my_z_val * my_y_val * (1-my_obs_val) * log(pow_val_less1)
    + my_z_val * (1-my_y_val) * my_obs_val * log(1-pow_val)
    + my_z_val * (1-my_y_val) * (1-my_obs_val) * log(pow_val)
    + (1-my_z_val) * my_obs_val * log(1-pow_val)
    + (1-my_z_val) * (1-my_obs_val) * log(pow_val);

 }
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid==0) {
    // TODO: should reduce instead of for loop
    float lpOffSum = log(1-theta);
    int numLoop = min(convert_int(N)-convert_int(numPrevRun),numRun);
    //printf("numLoop: %d; num_run %d; N: %d; numPrevRun %d\n", numLoop, numRun, N, numPrevRun);
    for (int i = 0; i < numLoop; i++) {
      lpOffSum +=locmem[i];
    }

    //lpOffSum = locmem[3];


    //printf("my_y_ind %d; nth: %d, kth: %d, dth: %d; numRun %d:\n",
    //       my_y_ind, nth, kth, dth, numRun);
    //printf("nth: %d, kth: %d, dth: %d, cur_lp_off %f; lpOffSum: %f; cur_lp_on %f; lpOnSum: %f\n",
    //       nth, kth, dth,lp_off[my_y_ind], lpOffSum, lp_on[my_y_ind], lpOnSum);
    lp_off[my_y_ind] = lp_off[my_y_ind] + lpOffSum;
  }
}

//TODO: understand why barriers break with printfs in this function... checks ok with a non-local version though...
//TODO: check if this is fixed after oclgrind checks...
kernel void calc_y_lp_on(local float* locmem, global int* cur_y, global int *cur_z, global int* cur_recon,
		       global int* obs, global float* lp_on,
		       uint N, uint D, uint K, uint numPrevRun,
		       float lambda, float epsilon, float theta) {
  int nth = get_global_id(0) + numPrevRun;

  int numRun = get_global_size(0);
  int kth = get_global_id(1);
  int dth = get_global_id(2);
  int lid = get_local_id(0);

  int obs_ind = nth*D + dth;
  int my_y_ind = kth*D+dth;
  if (nth < N) {
  int my_obs_val = obs[obs_ind];
  int my_recon = cur_recon[obs_ind];
  int my_z_val = cur_z[nth*K+kth];
 
  int my_y_val =cur_y[my_y_ind];

  float pow_val = native_powr(1-lambda, my_recon) * (1-epsilon);
  float pow_val_less1 = native_powr(1-lambda, max(my_recon-1,0)) * (1-epsilon);
  float pow_val_plus1 = native_powr(1-lambda, my_recon+1) * (1-epsilon);

  locmem[lid] = my_z_val * my_y_val * my_obs_val * log(1-pow_val)
    + my_z_val * my_y_val * (1-my_obs_val) * log(pow_val)
    + my_z_val * (1-my_y_val) * my_obs_val * log(1-pow_val_plus1)
    + my_z_val * (1-my_y_val) * (1-my_obs_val) * log(pow_val_plus1)
    + (1-my_z_val) * my_obs_val * log(1-pow_val)
    + (1-my_z_val) * (1-my_obs_val) * log(pow_val);
  //printf("n: %d; k: %d; d: %d; z: %d, y: %d, x: %d; recon: %d; lpOff: %f; lpOn: %f\n",
  //  nth,kth,dth,my_z_val, my_y_val, my_obs_val, my_recon, locmem[1+lid], locmem[1+lid+numRun]);
}
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid==0) {
    // TODO: should reduce instead of for loop
    float lpOnSum = log(theta);
    int numLoop = min(convert_int(N)-convert_int(numPrevRun),numRun);

    for (int i = 0; i < numLoop; i++) {
      lpOnSum += locmem[i];
    }

    //lpOffSum = locmem[3];


    //printf("my_y_ind %d; nth: %d, kth: %d, dth: %d; numRun %d:\n",
    //       my_y_ind, nth, kth, dth, numRun);
    //printf("nth: %d, kth: %d, dth: %d, cur_lp_off %f; lpOffSum: %f; cur_lp_on %f; lpOnSum: %f\n",
    //       nth, kth, dth,lp_off[my_y_ind], lpOffSum, lp_on[my_y_ind], lpOnSum);
    lp_on[my_y_ind] = lp_on[my_y_ind] + lpOnSum;
  }
}



kernel void calc_y_lps_noLcl(local float* locmem, global int* cur_y, global int *cur_z, global int* cur_recon,
		       global int* obs, global float* lp_off, global float* lp_on,
		       uint N, uint D, uint K, uint numPrevRun,
		       float lambda, float epsilon, float theta) {
  int nth = get_global_id(0) + numPrevRun;
  int numRun = get_global_size(0);
  int kth = get_global_id(1);
  int dth = get_global_id(2);
  int lid = get_local_id(0);

  int my_y_ind = kth*D+dth;

  //int my_obs_val = 1;
  //int my_recon = 0;
  //int my_z_val = 1;
  //int my_y_val = (int) locmem[0];
  int my_y_val = cur_y[my_y_ind];

  if (lid==0) {
    // should be log(theta) vs. log(one-theta), but not doing that yet for diagnostics
    float lpOffSum = log(1-theta);
    float lpOnSum = log(theta);

    for (int n = 0; n < numRun; n++) {
      int obs_ind = n*D + dth;

      int my_obs_val = obs[obs_ind];
      int my_recon = cur_recon[obs_ind];
      int my_z_val = cur_z[n*K+kth];
      float pow_val = native_powr(1-lambda, my_recon) * (1-epsilon);
      float pow_val_less1 = native_powr(1-lambda, max(my_recon-1,0)) * (1-epsilon);
      float pow_val_plus1 = native_powr(1-lambda, my_recon+1) * (1-epsilon);


      lpOffSum += my_z_val * my_y_val * my_obs_val * log(1-pow_val_less1)
                  + my_z_val * my_y_val * (1-my_obs_val) * log(pow_val_less1)
                  + my_z_val * (1-my_y_val) * my_obs_val * log(1-pow_val)
                  + my_z_val * (1-my_y_val) * (1-my_obs_val) * log(pow_val)
                  + (1-my_z_val) * my_obs_val * log(1-pow_val)
                  + (1-my_z_val) * (1-my_obs_val) * log(pow_val);
      lpOnSum += my_z_val * my_y_val * my_obs_val * log(1-pow_val)
              + my_z_val * my_y_val * (1-my_obs_val) * log(pow_val)
              + my_z_val * (1-my_y_val) * my_obs_val * log(1-pow_val_plus1)
              + my_z_val * (1-my_y_val) * (1-my_obs_val) * log(pow_val_plus1)
              + (1-my_z_val) * my_obs_val * log(1-pow_val)
              + (1-my_z_val) * (1-my_obs_val) * log(pow_val);
    }

    //lpOffSum = locmem[3];


    //printf("my_y_ind %d; nth: %d, kth: %d, dth: %d; numRun %d:\n",
    //       my_y_ind, nth, kth, dth, numRun);
   // printf("kth: %d, dth: %d, cur_lp_off %f; lpOffSum: %f; cur_lp_on %f; lpOnSum: %f\n",
  //          kth, dth,lp_off[my_y_ind], lpOffSum, lp_on[my_y_ind], lpOnSum);
    lp_off[my_y_ind] = lp_off[my_y_ind] + lpOffSum;
    lp_on[my_y_ind] = lp_on[my_y_ind] + lpOnSum;
  }
}

kernel void sample_y_pre_calc(global int* cur_y, global float* lp_off, global float* lp_on,
			      global float* rand, uint K, uint D) {
  int kth = get_global_id(0);
  int dth = get_global_id(1);
  int my_y_ind = kth*D+dth;

  float lpOnSum = lp_on[my_y_ind];
  float lpOffSum = lp_off[my_y_ind];

  float logpost[2] = {lpOnSum, lpOffSum};
  uint labels[2] = {1, 0};
  lognormalize(logpost, 0, 2);

  cur_y[my_y_ind] = sample(2, labels, logpost, 0, rand[my_y_ind]);

}

kernel void sample_y(global int *cur_y,
		     global int *cur_z,
		     global int *z_by_y,
		     global int *obs,
		     global float *rand, //global float *on_loglik, global float *off_loglik,
		     uint N, uint D, uint K,
		     float lambda, float epsilon, float theta) {
  
  uint kth = get_global_id(0); // k is the index of features
  uint dth = get_global_id(1); // d is the index of pixels
  // calculate the prior probability of each cell is 1
  //printf("kth: %d, D: %d, dth: %d\n", kth, D, dth);
  float on_loglik_temp = log(theta); 
  float off_loglik_temp = log(1 - theta);
  
  int z_by_y_nth;
  int cur_y_kd;
  
  // extremely hackish way to calculate the loglikelihood
  for (int n = 0; n < N; n++) {
    z_by_y_nth = z_by_y[n * D + dth];
    // if the nth object has the kth feature
    if (cur_z[n * K + kth] == 1) {
      // if the observed pixel at dth is on
      if (obs[n * D + dth] == 1) {
	cur_y_kd = cur_y[kth * D + dth];
	// if the feature image previously has this pixel on
	  on_loglik_temp += cur_y_kd*log(1 - pow(1 - lambda, z_by_y_nth) * (1 - epsilon)) + (1-cur_y_kd)*log(1 - pow(1 - lambda, z_by_y_nth + 1) * (1 - epsilon));
	  off_loglik_temp +=  cur_y_kd*log(1 - pow(1 - lambda, z_by_y_nth - 1) * (1 - epsilon))+(1- cur_y_kd)* log(1 - pow(1 - lambda, z_by_y_nth) * (1 - epsilon));
	
      } else {
	on_loglik_temp += log(1 - lambda);
	off_loglik_temp += log(1.0f);
      }
    } 
  }
  float logpost[2] = {on_loglik_temp, off_loglik_temp};
  uint labels[2] = {1, 0};
  lognormalize(logpost, 0, 2);
  cur_y[kth * D + dth] = sample(2, labels, logpost, 0, rand[kth * D + dth]);
  //printf("%f %f %d \n", logpost[0], logpost[1], cur_y[kth * D + dth]);
}

kernel void sample_z(global int *cur_y,
		     global int *cur_z,
		     global int *z_by_y,
		     global int *z_col_sum,
		     global int *obs,
		     global float *rand, 
		     uint N, uint D, uint K,
		     float lambda, float epsilon, float theta, float T) {
  
  uint nth = get_global_id(0); // n is the index of data
  uint kth = get_global_id(1); // k is the index of features
  
  // calculate the prior probability of each cell is 1
  float on_prob_temp = (z_col_sum[kth] - cur_z[nth * K + kth]) / (float)N; 
  float off_prob_temp = 1 - (z_col_sum[kth] - cur_z[nth * K + kth]) / (float)N;

  int z_by_y_dth;
  // extremely hackish way to calculate the probelihood
  for (int d = 0; d < D; d++) {
    z_by_y_dth = z_by_y[nth * D + d];
    // if the kth feature can turn on a pixel at d
    if (cur_y[kth * D + d] == 1) {
      // if the observed pixel at dth is on
      if (obs[nth * D + d] == 1) {
	// if the nth object previously has the kth feature
	if (cur_z[nth * K + kth] == 1) {
	  on_prob_temp *= 1 - pow(1 - lambda, z_by_y_dth) * (1 - epsilon);
	  off_prob_temp *= 1 - pow(1 - lambda, z_by_y_dth - 1) * (1 - epsilon);
	} else {
	  on_prob_temp *= 1 - pow(1 - lambda, z_by_y_dth + 1) * (1 - epsilon);
	  off_prob_temp *= 1 - pow(1 - lambda, z_by_y_dth) * (1 - epsilon);
	}
      } else {
	on_prob_temp *= 1 - lambda;
	off_prob_temp *= 1.0f;
      }
    } 
  }
  
  //printf("index: %d post_on: %f post_off: %f\n", nth * K + kth, on_prob_temp, off_prob_temp);
  float post[2] = {on_prob_temp *T, off_prob_temp*T};
  uint labels[2] = {1, 0};
  pnormalize(post, 0, 2);
  //printf("before index: %d %f %f %d \n", nth * K + kth, post[0], post[1], cur_z[nth * K + kth]);
  cur_z[nth * K + kth] = sample(2, labels, post, 0, rand[nth * K + kth]);
  //printf("after index: %d %f %f %d \n", nth * K + kth, post[0], post[1], cur_z[nth * K + kth]);
}


kernel void calc_z_lps(local float* locmem, global int *cur_y,  global int *cur_z,
		       global int* cur_recon, global int *z_col_sum, global int *obs,
		       global float* lp_off, global float* lp_on,
		       uint N, uint D, uint K, uint numPrevRun,
		       float lambda, float epsilon, float theta) {
  int nth = get_global_id(0)+numPrevRun;
  int numRun = get_global_size(0);
  int kth = get_global_id(1);
  int dth = get_global_id(2);
  int lid = get_local_id(0);

  int obs_ind = nth*D + dth;
  int my_y_ind = kth*D+dth;
  int my_z_ind = nth*K+kth;
  int myNKInd = nth*K*D + kth*D + dth;
  if (nth < N) {
    int my_obs_val = obs[obs_ind];
    int my_recon = cur_recon[obs_ind];
    int my_z_val = cur_z[my_z_ind];

  /*if (lid == 0) {
    locmem[0] = (float) cur_y[my_y_ind];
  }*/
  //barrier(CLK_LOCAL_MEM_FENCE);
  //int my_y_val = (int) locmem[0];
    int my_y_val = cur_y[my_y_ind];
    float pow_val = native_powr(1-lambda, my_recon) * (1-epsilon);
    float pow_val_less1 = native_powr(1-lambda, max(my_recon-1,0))*(1-epsilon);
    float pow_val_plus1 = native_powr(1-lambda, my_recon+1)*(1-epsilon);

  // lpOff
    lp_off[myNKInd] = my_z_val * my_y_val * my_obs_val * log(1-pow_val_less1)
      + my_z_val * my_y_val * (1-my_obs_val) * log(pow_val_less1)
      + my_z_val * (1-my_y_val) * my_obs_val * log(1-pow_val)
      + my_z_val * (1-my_y_val) * (1- my_obs_val) * log(pow_val)
      + (1-my_z_val) * my_obs_val * log(1-pow_val)
      + (1-my_z_val) * (1-my_obs_val) * log(pow_val);
  // lpOn (it may be possible to write this shorter -- yah, if you focus on my_y_val instead it's easy duh.)
    lp_on[myNKInd] = my_z_val * my_y_val * my_obs_val * log(1-pow_val)
      + my_z_val * my_y_val * (1-my_obs_val) * log(pow_val)
      + my_z_val * (1-my_y_val) * my_obs_val* log(1-pow_val)
      + my_z_val * (1-my_y_val) * (1-my_obs_val) * log(pow_val)
      + (1-my_z_val) * my_y_val * my_obs_val * log(1-pow_val_plus1)
      + (1-my_z_val) * my_y_val * (1-my_obs_val) *log(1-pow_val_plus1)
      + (1-my_z_val) * (1-my_y_val) * my_obs_val * log(pow_val)
      + (1-my_z_val) * (1-my_y_val) * (1-my_obs_val) * log(pow_val);
    if (dth==0) {
      int mk = z_col_sum[kth];
      float lpPr = (1.0* (mk-my_z_val))/N;
      lp_off[myNKInd] += log(1-lpPr);
      lp_on[myNKInd] += log(lpPr);
    }
//    printf("n: %d; k: %d: d: %d; myNKInd: %d; my_z_val: %d; my_y_val: %d; lpOn: %f; lpOff: %f\n",
//            nth, kth, dth, myNKInd, my_z_val, my_y_val, lp_on[myNKInd], lp_off[myNKInd]);
  }

  /*this is incorrect bc it's ok for lp_off to be Inf when everyone has taken the feature, you should too.
    if ((lp_off[myNKInd] < -99999.) || (lp_off[myNKInd] > 0.)) {
    printf("error, lp off should not be %f. nth %d; kth %d; dth %d; my_z_val %d, my_y_val %d; my_obs %d; my_recon %d; myNKInd %d\n",
           lp_off[myNKInd],nth,kth,dth,my_z_val,my_y_val, my_obs_val, my_recon, myNKInd);
  }
  if ((lp_on[myNKInd] < -99999.) || (lp_on[myNKInd] > 0.)) {
    printf("error, lp on should not be %f. nth %d; kth %d; dth %d; my_z_val %d, my_y_val %d; my_obs %d; my_recon %d; myNKInd %d\n",
           lp_on[myNKInd],nth,kth,dth,my_z_val,my_y_val, my_obs_val, my_recon, myNKInd);
  }*/

//  lp_off[myNKInd] = myNKInd;
//  lp_on[myNKInd] = myNKInd;
}
//TODO: combine with sample y pre calc to unify these functions...
kernel void sample_z_pre_calc(global int* cur_z, global float* lp_off, global float* lp_on,
			      global float* rand, uint N, uint K, uint D, float T) {

  // remember all are d=0 for all n, k, in lp_on/lp_off, so adjust indicies accordingly
  int nth = get_global_id(0);
  int kth = get_global_id(1);
  int my_nk_ind = nth*K*D+kth * D;
  int my_z_ind = nth*K + kth;
  if ((nth < N) && (kth < K)) {
    float lpOnSum = lp_on[my_nk_ind];
    float lpOffSum = lp_off[my_nk_ind];
    if (isinf(lpOffSum)) {
      cur_z[my_z_ind] = 1;
    }
    else if (isinf(lpOnSum)) {
     // printf("error: don't think i should be getting any empty features...\n");
      cur_z[my_z_ind] = 0;
    }
    else {
      float logpost[2] = {lpOnSum * T, lpOffSum*T};
      uint labels[2] = {1, 0};

      lognormalize(logpost, 0, 2);
//      if ((nth == 0) && (kth == 0)) {
//
//        printf("T: %f; pre: on = %e; off = %e; postAnneal: p(z_00 = 1) = %f; p(z_00 = 0) = %f; cur K = %d\n ",
//                T, lpOnSum, lpOffSum, logpost[0], logpost[1],K);
//      }
      cur_z[my_z_ind] = sample(2, labels, logpost, 0, rand[my_z_ind]);
    }
  }

}

//TODO: float4-ify
//TODO: make sure this still works without it being restricted to those gid+i
//TODO: and still works with other changes...

kernel void sum_reduc1d(local float* part_sum, global float* data,
                        global float* out, uint maxInd) {
  int group_size = get_local_size(0);
  int lid = get_local_id(0);
  int offId = get_global_id(1);

  int gid = offId * group_size+lid;
  if (gid < maxInd) {
    part_sum[lid] = data[gid];
  }
  else {
    part_sum[lid] = 0.; // for safety in case it tries to sum it...
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = group_size/2; i > 0; i >>= 1) {
    //if ((lid < i) && ((gid+i) < maxInd)) {
    if ((lid < i) && (gid < maxInd)) {
      part_sum[lid] += part_sum[lid+i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if ((lid == 0) && (gid < maxInd)) {
    out[offId] = part_sum[0];
  }
}

kernel void finish_sum_reduc1d(local float* part_sum, global float* data, global float* out,
                              uint numLeft) {
  int lid = get_local_id(0);
  int group_size = get_local_size(0);
  // = because of  off by one error
  if (lid < numLeft) {
    part_sum[lid] = data[lid];
  }
  else {
    part_sum[lid] = 0.;
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = group_size/2; i > 0; i>>=1) {
    if ((lid < i) && ((lid+i) <= numLeft)) {
      part_sum[lid] += part_sum[lid+i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0) {
    *out = part_sum[0];
  }

}

kernel void reduc_vec_sum3d(local float* part_sums, global float* data, global float* out,
                          uint D1, uint D2, uint D3, uint startInd){

  int group_size = get_local_size(0);
  int lid = get_local_id(0);
  int dth = get_global_id(0) + startInd;
  int nth = get_global_id(1);
  int kth = get_global_id(2);
  uint myNKInd = nth*D2*D3+kth*D3;

  if (dth < D3) {
    part_sums[lid] = data[myNKInd+dth];
    //if ((nth==199) && (kth == 9)) {
    //  printf("nth: %d; kth: %d; lid: %d; dth %d; part_sums[lid]: %f; data[myNKInd+dth]: %f\n",
    //         nth, kth, lid, dth, part_sums[lid], data[myNKInd+dth]);
     //}
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = group_size/2; i > 0; i >>= 1) {
    if  ((lid < i) && ((dth+i) < D3)) {
    //  if ((nth == 1) && (kth == 1) && (startInd > 1)) {
     //    printf("dth %d, lid %d, and i %d: \t summing indices %d and %d with values %f and %f\n",
     //          dth, lid, i, lid+startInd, startInd+lid+i, part_sums[lid+startInd],part_sums[lid+i+startInd]);
     //  }
      part_sums[lid] +=  part_sums[lid+i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }


  if (lid == 0) {
//     out[myNKInd + dth/group_size] = part_sums[0];
    out[myNKInd+dth/startInd] = part_sums[0];
     if (startInd > 0) {
     // get rid of the empty spot
        out[myNKInd+ dth] = 0.;
     }

  }
  else if (dth < D3) {
    out[myNKInd+dth] = 0.;
  }
}

kernel void finish_reduc_vec_sum3d(local float* part_sums, global float* data, global float* sum,
                                  int D1, int D2, int D3, int numLeft) {
  int lid = get_local_id(0);
  int group_size = get_local_size(0);
  int dth = get_global_id(0);
  int nth = get_global_id(1);
  int kth = get_global_id(2);

  int myNKInd = nth*D2*D3+kth*D3;


  if ((lid < numLeft) && (dth < numLeft)) {
    if (numLeft != D3) {
      part_sums[lid] = data[myNKInd + dth];
    }
    else {
      part_sums[lid] = sum[myNKInd+dth];
    }

   //if ((nth == 0) && (kth==0) && (dth == 0))
   //     printf("parts_sums[lid]: %f, data: %f, nkInd: %d\n", part_sums[lid], data[myNKInd+dth], myNKInd);
    //sum[myNKInd+dth] = myNKInd+dth;
    //printf("myNKInd %d; sum: %f \n", myNKInd+dth, sum[myNKInd+dth]);
//    if ((nth == 1) && (kth == 1))   {
//      printf("in finish: lid %d, dth %d, kth %d nth %d myNKInd %d part_sums val %f; numLeft: %d\n",
//              lid, dth, kth, nth, myNKInd, part_sums[lid], numLeft);
//    }
  }
  else {
    part_sums[lid] = 0.; // for safety
  }
  // fix off by 1 error when we're finishing prev. reduction...
  if ((D3 != numLeft) && (lid == numLeft)) {
    part_sums[lid] = data[myNKInd + lid];
//     if ((nth == 1) && (kth == 1))   {
//      printf("in finish: lid %d, dth %d, kth %d nth %d myNKInd %d part_sums val %f; numLeft: %d\n",
//              lid, dth, kth, nth, myNKInd, part_sums[lid], numLeft);
//    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = group_size/2; i > 0; i >>=1) {
        if ((lid < i) && ((lid+i) <= numLeft) && (dth < D3) && ((dth+i) < D3))  {
//          if ((nth == 1) && (kth == 1)) {
//                     printf("dth %d, lid %d, and i %d currently summing indices %d and %d",
//                          dth, lid, i, lid, lid+i);
//                     //printf("with values %f and %f\n", part_sums[lid],part_sums[lid+i]);
//          }
                    // printf("POST: dth %d, lid %d, and i %d summing indices %d and %d",
                      //    dth, lid, i, lid, lid+i);
                    // printf("with values %f and %f\n", part_sums[lid],part_sums[lid+i]);

          part_sums[lid] += part_sums[lid+i];

        }
        barrier(CLK_LOCAL_MEM_FENCE);
  }

  if ((lid == 0) && (dth < D3)) {
  //  for (int i = 1; i < numLeft; i++) {
   //     part_sums[0] += part_sums[i];
    //}
      //printf("nth %d, kth %d, dth %d, myNKInd %d; cur val %f; total sum = %f\n", nth, kth, dth, myNKInd, sum[myNKInd], part_sums[0]);
      //sum[nth*D2 +kth] = part_sums[0];
      sum[myNKInd] = part_sums[0];
  }

}

kernel void logprob_z_data(global int *cur_z,
			   global int *cur_y,
			   global int *obs,
			   global float *logprob,
			   uint N, uint D, uint K,
			   float alpha, float lambda, float epsilon) {

  uint nth = get_global_id(0); // n is the index of data
  uint m;
  uint novel_count = 0;
  float logprob_temp = 0;

  uint cur_z_nth;
  /* calculate the log probability of the nth row of Z 
     i.e., the prior probability of having the features
     of the nth object.
   */
  for (int k = 0; k < K; k++) {
    m = 0;
    cur_z_nth = cur_z[nth * K + k];
    for (int n = 0; n < nth; n++) {
      m += cur_z[n * K + k];
    }
    if (m > 0) { // if other objects have had this feature
      if (cur_z_nth == 1) {
	logprob_temp += log(m / (nth + 1.0f));
      } else {
	logprob_temp += log(1 - m / (nth + 1.0f));
      }
    } else { // if this is a novel feature
      novel_count += cur_z_nth == 1;
    }
  }
  logprob_temp += (novel_count > 0) * pois_logpmf(novel_count, alpha / (nth+1.0f));

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
      logprob_temp += log(1 - pow(1 - lambda, weight) * (1 - epsilon));
    } else {
      logprob_temp += weight * log(1 - lambda) + log(1 - epsilon);
    }
  }
  logprob[nth] = logprob_temp;
}

//TODO: localify
kernel void compute_recon_objs(global int*cur_y, global int* cur_z, global int* obj_recon,
                                uint N, uint K, uint D) {

  uint nth = get_global_id(0);
  uint kth = get_global_id(1);
  uint dth = get_global_id(2);

  int recon_idx = nth*D + dth;

  
  int my_val = cur_z[nth*K+kth] * cur_y[kth*D+dth];
  //printf("recon no zsum: recon id: %d\n", recon_idx);
  atomic_add((obj_recon+recon_idx), my_val);
  //  printf("my nth: %d kth %d dth %d, curz: %d, cury %d, my_val %d my recon %d \n", nth, kth, dth, cur_z[nth*K+kth], cur_y[kth*D+dth], my_val, obj_recon[recon_idx]);
 

  /*if (my_val < 0) {
    printf("bad my_val (%d). nth %d, kth %d, dth %d, my_trans_d %d, recon_idx %d, zInd %d, yInd %d, cur_z %d, cur_y %d\n
",
            my_val, nth, kth, dth, my_trans_d, recon_idx, (nth*K+kth), (kth*D+my_trans_d), cur_z[kth*K+dth], cur_y[kth*D+my_trans_d]);
  }*/
 }


//TODO: localify
kernel void compute_recon_objs_andzsums(global int*cur_y, global int* cur_z, global int* obj_recon,
					global int* zsums, uint N, uint K, uint D) {

  uint nth = get_global_id(0);
  uint kth = get_global_id(1);
  uint dth = get_global_id(2);

  if ((nth < N) && (kth < K) && (dth < D)) {
    int recon_idx = nth*D + dth;

    int z_val = cur_z[nth*K+kth];
    int my_val = z_val * cur_y[kth*D+dth];
    //printf("nth: %d; kth: %d; dth: %d; my_val: %d; recon id: %d\n", nth,kth,dth,my_val,recon_idx);
    atomic_add((obj_recon+recon_idx), my_val);

    if (dth == 0) {
      atomic_add((zsums+kth), z_val);
    }
  }
  //  printf("my nth: %d kth %d dth %d, curz: %d, cury %d, my_val %d my recon %d \n", nth, kth, dth, cur_z[nth*K+kth], cur_y[kth*D+dth], my_val, obj_recon[recon_idx]);
 

  /*if (my_val < 0) {
    printf("bad my_val (%d). nth %d, kth %d, dth %d, my_trans_d %d, recon_idx %d, zInd %d, yInd %d, cur_z %d, cur_y %d\n
",
            my_val, nth, kth, dth, my_trans_d, recon_idx, (nth*K+kth), (kth*D+my_trans_d), cur_z[kth*K+dth], cur_y[kth*D+my_trans_d]);
  }*/
 }

kernel void calc_lps(global int * obj_recon, global int* obs, global float* lps,
                    uint N, uint D, float lambda, float epsilon) {

  uint nth = get_global_id(0);
  uint dth = get_global_id(1);

  int my_obs_ind = nth*D+dth;

  int my_recon = obj_recon[my_obs_ind];
  int my_obs = obs[my_obs_ind];

  float pow_val = native_powr(1-lambda,my_recon) * (1-epsilon);
  lps[my_obs_ind] = my_obs * log(1-pow_val) + (1-my_obs) * log(pow_val);
}


kernel void calc_lp_fornew(global int* obj_recon, global int* obs, global float* lps,
                    uint N, uint K, uint D, uint KNewMax,
                    float lambda, float epsilon, float theta) {

  int nth = get_global_id(0);
  int dth = get_global_id(1);
  int k_newth = get_global_id(2);

  //  printf("nth %d; dth: %d; N: %d; D: %d; N*D: %d; nth*D+dth: %d\n",
  //       nth, dth, N, D, N*D, nth*D+dth);
  int my_obs_idx = nth*D+dth;
  int my_obs_val = obs[my_obs_idx];
  int my_pred = obj_recon[my_obs_idx];

  uint mylp_ind = nth*D*KNewMax+dth*KNewMax + k_newth;
  //uint mylp_ind = dth*D*KNewMax + nth*D+k_newth;
  float pow_val = (1-epsilon) * native_powr(1-lambda, my_pred) * native_powr(1-lambda*theta, k_newth);
  lps[mylp_ind] = my_obs_val * log(1-pow_val) + (1-my_obs_val) * log(pow_val);
  //lps[mylp_ind] = (1-my_obs_val) * log(pow_val);
  //printf("nth: %d dth: %d k_newth: %d myobs_idx: %d  myobs: %d mypred: %d: my lpInd:%d  mypow_val %f, mylp: %f\n", nth, dth, k_newth, my_obs_idx, my_obs_val,  my_pred, mylp_ind, pow_val, lps[mylp_ind]);
}

kernel void new_y_val_probs(local int* locmem, global float* cur_z, global float* cur_y,
                              global int* comb_vec, global int* obj_recon,
                              global int* obs, global float * new_probs,
                              uint N, uint K, uint D, uint newK,
                              float lambda, float epsilon, float theta) {
  const int  NEW_K_START = 0;
  int dth = get_global_id(0);
  int nth = get_global_id(1);
  int newKth = get_global_id(2);
  int lid = get_local_id(0);

  int my_obs_idx = nth*D+dth;
  //TODO: fix me
  /*  if (lid == 0) {
    locmem[MY_OBS] = obs[my_obs_idx];
    locmem[MY_RECON] = obj_recon[my_obs_idx];
    }*/
  if (lid < newK) {
    locmem[NEW_K_START+lid] = comb_vec[lid];
    // if ((newKth == 0) && (nth == 0)) {
    //  printf("lid: %d comb_val: %d locmem %d \n", lid, comb_vec[lid], locmem[NEW_K_START+lid]);
    // }
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if (dth < D) {
    //    int my_obs_val = locmem[MY_OBS];
    int my_obs_val = obs[my_obs_idx];
    // int my_recon_val = locmem[MY_RECON];
    int my_recon_val = obj_recon[my_obs_idx];
    int my_knew_choose_val = locmem[NEW_K_START + newKth];
    float lik_pow_val = native_powr(1-lambda,my_recon_val+newKth) * (1-epsilon);
    float pr_pow_val = my_knew_choose_val * native_powr(theta, newKth) * native_powr(1-theta, newK-newKth);

    int my_probs_ind = newKth*N*D + nth*D+dth;
    
    new_probs[my_probs_ind] = my_obs_val * (1-lik_pow_val) * pr_pow_val +
      (1-my_obs_val) * lik_pow_val * pr_pow_val;
    //new_probs[my_probs_ind] = (float) obs[my_obs_idx];
    //if (nth == 0) {
    // printf("my knew %d, knew val 1 %d knewval 2 %d, my knew val %d\n", newKth, locmem[NEW_K_START+0],locmem[NEW_K_START+1], locmem[NEW_K_START+newKth]); 
    // }
    //      printf("dth: %d, nth: %d, newKth: %d, my_obs: %d, my_recon %d, my_k_new_val %d, new-probs-ind %d, comb_val: %d, lik: %f, pr %f,  new_probs: %f\n", dth, nth, newKth, my_obs_val, my_recon_val, my_knew_choose_val, my_probs_ind, my_knew_choose_val, lik_pow_val, pr_pow_val,  new_probs[my_probs_ind]);
  }
}

kernel void y_new_samp(local float* locmem, global float* data, 
		       global int* cur_y, global int* knews, global float* y_rands,
		       int D1, int D2, int D3, int oldK) {
  int lid = get_local_id(0);
  int group_size = get_local_size(0);
  int knewth = get_global_id(0);
  int nth = get_global_id(1);
  int dth = get_global_id(2);

  int myInd = knewth*D2*D3+nth*D3+dth;
  float my_val = 0.f;
  if (knewth < D1) {
     my_val = data[myInd];
    locmem[lid] = my_val;
 //   if ((nth == 1) && (kth == 1))
 //  printf("in finish: lid %d, dth %d, kth %d nth %d myNKInd %d locmem val %f\n", lid, dth, kth, nth, myNKInd, locmem[lid]);
  }
  else {
    locmem[lid] = 0.; // for safety
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int i = group_size/2; i > 0; i >>=1) {
         if (lid < i)  {
        //if ((nth == 1) && (kth == 1)) {
        //           printf("dth %d, lid %d, and i %d currently summing indices %d and %d with values %f and %f\n",
      //                  dth, lid, i, lid, lid+i, locmem[lid],locmem[lid+i]);
          // }
          locmem[lid] += locmem[lid+i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
       }
  // locmem[0] has norm constant.
  // put normed value in 1+k_new to then sample from lid == 0

  float normVal = locmem[0];
  locmem[1+knewth] = my_val/normVal;
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid==0) {
    int my_rand_ind = nth*D3+dth;
    float my_rand_val = y_rands[my_rand_ind];

    int curInd = 1;
    float curP = locmem[curInd];
    while ((curP < my_rand_val) && (curInd < (D1+1))) {
      curInd++;
      curP+=locmem[curInd];
    }
    
    int numToTurnOn = curInd-1;
     
    int myKNews = knews[nth];
    int preKVals = 0;
    for (int i = 0; i < nth; i++) {
      preKVals += knews[i];
    }

    
    int kSt = oldK + preKVals;
    if (nth < kSt) {
    //if (nth <= numToTurnOn) {
      //      printf("knewth %d, nth %d, dth %d, numToTurnOn %d, preKVals %d, oldK %d, kSt %d \n", knewth, nth, dth, numToTurnOn, preKVals, oldK, kSt);
      for (int i = 0; i < numToTurnOn; i++) {
	cur_y[(kSt+i)*D3+dth] = 1;
      }
    }
  }
}

//samples new features for only one object at a time (use extra dimension to gain more dimension parallelization
kernel void new_y_val_probs2(local int* locmem, global float* cur_z, global float* cur_y,
			     global int* comb_vec, global int* obj_recon,
			     global int* obs, global float *new_probs,
			     uint nth, uint new_k, uint N, uint D, uint K,
			     float lambda, float epsilon, float theta) {
  const int NEW_K_START = 0;
  //int dBase = get_global_id(0);
  int lid0 = get_local_id(0); // where in cur dim window
  int lid1 = get_global_id(1); // which dim window
  int dth = lid0 + lid1*get_local_size(0); // window size
  int newKth = get_global_id(2);
  
  if (dth < D) {

    // remember 0... new_k are possible values...
    if (lid0 <= new_k) {
      locmem[NEW_K_START + lid0] = comb_vec[lid0];
    }
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  if (dth < D) {
    int my_obs_idx = nth*D+dth;
    int my_obs_val = obs[my_obs_idx];
    int my_recon_val = obj_recon[my_obs_idx];
    int my_knew_choose_val = locmem[NEW_K_START + newKth];

    float lik_pow_val = native_powr(1-lambda,my_recon_val+newKth) * (1-epsilon);
    float pr_pow_val = my_knew_choose_val * native_powr(theta, newKth) * native_powr(1-theta, new_k-newKth);
    
    int my_probs_ind = newKth*D + dth;
    //printf("myprobsind: %d; newkth: %d; dth: %d; lid0: %d; lid1: %d\n")
    new_probs[my_probs_ind] = my_obs_val * (1-lik_pow_val) * pr_pow_val +
      (1-my_obs_val) * lik_pow_val * pr_pow_val;
  }
}

kernel void y_new_samp2(local float* locmem, global float* data,
			global int* cur_y, global float * y_rands,
			uint nth, uint new_k, uint N, uint D, uint oldK) {
  int lid = get_local_id(0);
  int group_size = get_local_size(0);
  int knewth = get_global_id(0);
  int dth = get_global_id(2);
  int myInd = knewth * new_k + dth;

  float my_val = 0.f;

  //should be equal because values are inclusive 0...new_k (so new_k+1 values)
  if (knewth <= new_k) {
    my_val = data[myInd];
    locmem[lid] = my_val;
  }
  else {
    locmem[lid] = 0.f; // for safety
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  //TODO: localify norm over number of pix to turn on (these aren't so bad actually bc knew is at max 4)
  // (ALSO It's possible this previous code worked, bc i had a bug above
//  for (int i = group_size/2; i > 0; i >>=1) {
//         if (lid < i)  {
//        //if ((nth == 1) && (kth == 1)) {
//        //           printf("dth %d, lid %d, and i %d currently summing indices %d and %d with values %f and %f\n",
//      //                  dth, lid, i, lid, lid+i, locmem[lid],locmem[lid+i]);
//          // }
//          locmem[lid] += locmem[lid+i];
//        }
//        barrier(CLK_LOCAL_MEM_FENCE);
//       }
//  // locmem[0] has norm constant.
//  // put normed value in 1+k_new to then sample from lid == 0
//  float normVal = locmem[0];
//  locmem[1+knewth] = my_val/normVal;
//  barrier(CLK_LOCAL_MEM_FENCE);

  if (lid == 0) {

    float my_rand_val = y_rands[dth];
    float curNorm = 0.;
//    printf("pre-dth %d; p0On %f; p1On: %f; rand_val: %f\n",
//            dth, locmem[0], locmem[1], my_rand_val);
    for (int i = 0; i < (new_k+1); i++) curNorm+=locmem[i];


    int curInd = 0;
    float curP = (locmem[curInd]/curNorm);
    while ((curP < my_rand_val) && (curInd < (new_k+1))) {
//    while ((my_rand_val < curP) && (curInd < (new_k+1))) {
      curInd++;
      curP += (locmem[curInd]/curNorm);
    }

    int numToTurnOn = curInd;
//    printf("dth %d; norm: %f; p0On %f; p1On: %f; rand_val: %f; toTurnOn: %d\n",
//            dth, curNorm, locmem[0]/curNorm, locmem[1]/curNorm, my_rand_val, numToTurnOn);
    //TODO: localify this for loop (these aren't so bad actually bc knew is at max 4)
    for (int i = 0; i < numToTurnOn; i++) {
      int myInd = (oldK + i) * D + dth;
      cur_y[myInd] = 1;
    }
  }
}

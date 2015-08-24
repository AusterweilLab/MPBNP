// written by Joe Austerweil and Ting Qian

inline uint isPowerOfTwo (unsigned int x)
{
 return (
   x == 1 || x == 2 || x == 4 || x == 8 || x == 16 || x == 32 ||
   x == 64 || x == 128 || x == 256 || x == 512 || x == 1024 ||
   x == 2048 || x == 4096 || x == 8192 || x == 16384 ||
   x == 32768 || x == 65536 || x == 131072 || x == 262144 ||
   x == 524288 || x == 1048576 || x == 2097152 ||
   x == 4194304 || x == 8388608 || x == 16777216 ||
   x == 33554432 || x == 67108864 || x == 134217728 ||
   x == 268435456 || x == 536870912 || x == 1073741824 ||
   x == 2147483648);
}

// from http://graphics.stanford.edu/~seander/bithacks.html
inline uint powerOfTwoUpper(uint x) {
  uint v=x;
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

//inline void fatomic_add(volatile __global float*src, const float operand) {
//  union {
//    uint intVal;
//    float floatVal;
//  } newVal;
//  union {
//  uint intVal;
//  float floatVal;
//  } prevVal;
//  do {
//    prevVal.floatVal = *src;
//    newVal.floatVal = prevVal.floatVal + operand;
//  } while (atomic_cmpxchg((volatile __global uint*) src, prevVal.intVal, newVal.intVal) != prevVal.intVal);
//}
//
//inline void fatomic_max(volatile __global float*src, const float operand) {
//  union {
//    uint intVal;
//    float floatVal;
//  } newVal;
//  union {
//  uint intVal;
//  float floatVal;
//  } prevVal;
//  do {
//    prevVal.floatVal = *src;
//    newVal.floatVal = max(prevVal.floatVal, operand);
//  } while (atomic_cmpxchg((volatile __global uint*) src, prevVal.intVal, newVal.intVal) != prevVal.intVal);
//}
//
//inline void fatomic_min(volatile __global float*src, const float operand) {
//  union {
//    uint intVal;
//    float floatVal;
//  } newVal;
//  union {
//  uint intVal;
//  float floatVal;
//  } prevVal;
//  do {
//    prevVal.floatVal = *src;
//    newVal.floatVal = min(prevVal.floatVal, operand);
//  } while (atomic_cmpxchg((volatile __global uint*) src, prevVal.intVal, newVal.intVal) != prevVal.intVal);
//}

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

inline bool fEqualsP(float a, float b) {
  return fabs(a-b) < 1e-7;
}

//TODO: refactor to share shared functions
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

void v_translate_global(global int *temp_y, global int *transformed_y, uint nth, uint kth,
			uint f_img_height, uint f_img_width, uint distance,
			uint K, uint D) {
  for (int h = 0; h < f_img_height; h++) {
    for (int w = 0; w < f_img_width; w++) {
      transformed_y[nth * K * D + kth * D + ((h + distance) % f_img_height) * f_img_width + w] = 
	temp_y[nth * K * D + kth * D + h * f_img_width + w];
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

void h_translate_global(global int *temp_y, global int *transformed_y, uint nth, uint kth,
			uint f_img_height, uint f_img_width, uint distance,
			uint K, uint D) {
  for (int h = 0; h < f_img_height; h++) {
    for (int w = 0; w < f_img_width; w++) {
      transformed_y[nth * K * D + kth * D + h * f_img_width + (w + distance) % f_img_width] =
	temp_y[nth * K * D + kth * D + h * f_img_width + w];
    }
  }
}

void scale(local int *orig_y, local int *new_y, uint kth, 
	   uint f_img_height, uint f_img_width, uint x_pixel, uint y_pixel,
	   uint D) {

  // compute the new height and width of scaled matrix
  int new_height = f_img_height + y_pixel;
  int new_width = f_img_width + x_pixel;

  uint hh, ww, h, w;
  for (h = 0; h < f_img_height; h++) {
    for (w = 0; w < f_img_width; w++) {
      hh = (int)round((float)h / new_height * f_img_height);
      ww = (int)round((float)w / new_width * f_img_width);
      if (hh < f_img_height & ww < f_img_width) {
	new_y[kth * D + h * f_img_width + w] = orig_y[kth * D + hh * f_img_width + ww];
      } else {
	new_y[kth * D + h * f_img_width + w] = 0;
      }
    }
  }
}

void scale_global(global int *temp_y, global int *transformed_y, uint nth, uint kth, 
		  uint f_img_height, uint f_img_width, uint x_pixel, uint y_pixel,
		  uint K, uint D) {

  // compute the new height and width of scaled matrix
  int new_height = f_img_height + y_pixel;
  int new_width = f_img_width + x_pixel;

  uint hh, ww, h, w; // hh and ww are indices to original y
  for (h = 0; h < f_img_height; h++) {
    for (w = 0; w < f_img_width; w++) {
      hh = (int)round((float)h / new_height * f_img_height);
      ww = (int)round((float)w / new_width * f_img_width);
      if (hh < f_img_height & ww < f_img_width) {
	transformed_y[nth * K * D + kth * D + h * f_img_width + w] =
	  temp_y[nth * K * D + kth * D + hh * f_img_width + ww];
      } else {
	transformed_y[nth * K * D + kth * D + h * f_img_width + w] = 0;
      }
    }
  }
}

kernel void compute_recon_objs_posttrans(global int*cur_y, global int* cur_z, global int* obj_recon,
                                  uint N, uint D, uint K) {

  uint nth = get_global_id(0);
  uint kth = get_global_id(1);
  uint dth = get_global_id(2);

  int recon_idx = nth*D + dth;


  int my_val = cur_z[nth*K+kth] * cur_y[kth*D+dth];

  atomic_add((obj_recon+recon_idx), my_val);

 }

// TODO: make a lookup table of mods for covnerting transformed dimensions (avoid overuse of mods, which are slow on gpu)
//TODO: localify
kernel void compute_recon_objs_trans(global int*cur_y, global int* cur_z, global int *cur_r, global int* obj_recon,
                                uint N, uint K, uint D, uint f_img_width) {

  const uint V_TRANS = 0, H_TRANS = 1, NUM_TRANS = 2;
  uint nth = get_global_id(0);
  uint kth = get_global_id(1);
  uint dth = get_global_id(2);
  uint f_img_height = D/f_img_width;


  int my_h_off = cur_r[nth*(K*NUM_TRANS) + kth*NUM_TRANS + V_TRANS];
  int my_w_off = cur_r[nth*(K*NUM_TRANS) + kth*NUM_TRANS + H_TRANS];


  int my_trans_d = ((my_w_off % f_img_height) * f_img_width + my_h_off % f_img_width+dth)%D;
//  int my_trans_d = (my_w_off % f_img_height) * f_img_width + my_h_off % f_img_width;

  int recon_idx = nth*D + dth;
  int my_val = cur_z[nth*K+kth] * cur_y[kth*D+my_trans_d];

  atomic_add((obj_recon+recon_idx), my_val);

 }

 kernel void compute_recon_objs_transzsum(global int*cur_y, global int* cur_z, global int *cur_r,
                                          global int* obj_recon, global int* zsums,
                                           uint N, uint K, uint D, uint f_img_width) {

  const uint V_TRANS = 0, H_TRANS = 1, NUM_TRANS = 2;
  int nth = get_global_id(0);
  int kth = get_global_id(1);
  int dth = get_global_id(2);
  int f_img_height = D/f_img_width;

  int my_w_off = cur_r[nth*(K*NUM_TRANS) + kth*NUM_TRANS + H_TRANS];
  int my_h_off = cur_r[nth*(K*NUM_TRANS) + kth*NUM_TRANS + V_TRANS];

  int my_trans_d = ((my_w_off % f_img_height) * f_img_width + my_h_off % f_img_width+dth)%D;
//  printf("nth: %d, kth: %d, dth: %d, w_off: %d; h_off: %d; img_h: %d; img_w: %d; trans_d:%d\n",
//    nth, kth, dth, my_w_off, my_h_off, f_img_height, f_img_width, my_trans_d);
  int recon_idx = nth*D + dth;
  int z_val = cur_z[nth*K+kth];
//    if ((nth == 0) && (kth==0) && (dth==0)) {
//    printf("N: %d; K: %d; D: %d: img_w: %d\n", N, K, D, f_img_width);
//  }

  int my_val = z_val * cur_y[kth*D+my_trans_d];
//  printf("n: %d; k: %d; d: %d; w_off: %d; h_off: %d; d_t: %d; recon_id: %d; z_val: %d; y_val:%d; my_val: %d\n",
//          nth,kth,dth,my_w_off,my_h_off,my_trans_d,recon_idx,
//          cur_z[nth*K+kth],cur_y[kth*D+my_trans_d], my_val);

  atomic_add((obj_recon+recon_idx), my_val);

  if (dth == 0) {
    atomic_add((zsums+kth), z_val);
  }
 }

// split to separate f'ns for synch issues...
//TODO: use local memory to lower data overhead
kernel void calc_zr_lps(global int* cur_y, global int * cur_z,global int *cur_r,
                          global int* obj_recon,
                          global float* lp_nkr_on, global float* lp_nk_off,
                          global int * obs,
                          uint N, uint K, uint D, uint f_img_width,
                          float lambda, float epislon, float theta) {

  const uint V_TRANS = 0, H_TRANS = 1, NUM_TRANS = 2;
  uint nth = get_global_id(0);
  uint kth = get_global_id(1);
  uint rth = get_global_id(2);

  uint f_img_height = D/f_img_width;

  int my_z_ind = nth*K+kth;
  int my_z_val = cur_z[my_z_ind];

  int v_tr = rth % f_img_width;
  int h_tr = rth / f_img_width;
  int t_index = (v_tr) %f_img_height*f_img_width+(h_tr)%f_img_width;

  int old_v_tr = cur_r[nth*(K*NUM_TRANS) + kth*NUM_TRANS +V_TRANS];
  int old_h_tr = cur_r[nth*(K*NUM_TRANS) + kth*NUM_TRANS +H_TRANS];
  int t_old_index = (old_v_tr) % f_img_height*f_img_width+(old_h_tr)%f_img_width;


  //int my_y_val = cur_y[kth*D+t_index];
  //int my_old_y_val = cur_y[kth*D+t_old_index];


  float lP_z_off = 0.;
  float lP_z_on = 0.;
//  float lP_z_on = 10;

//  uint firstP = 0;
//  uint firstP2 = 0;

  for (uint d = 0; d < D; d++ ) {
    int my_obs_val = obs[nth*D+d];
    int cur_pred = obj_recon[nth*D+d];
    int old_y_idx = kth*D+ ((t_old_index +d)% D);
    int my_old_y_val = cur_y[old_y_idx];
    int y_idx = kth*D+((t_index+d)%D);
    int my_y_val = cur_y[y_idx];

    float pow_val = (1-epislon)*native_powr(1-lambda, cur_pred);
    float pow_val_less1 = (1-epislon)*native_powr(1-lambda, max(cur_pred-1,0));
    float pow_val_plus1 = (1-epislon)*native_powr(1-lambda, cur_pred+1);

    // calculate prob w/o feat k and w/ it (at the current transformation given by rth over all pixels d)
    lP_z_off += (1-my_z_val) * my_obs_val* log(1-pow_val)
              + (1-my_z_val) * (1-my_obs_val) * log(pow_val)
              + my_z_val * my_old_y_val *my_obs_val * log(1-pow_val_less1)
              + my_z_val * my_old_y_val * (1-my_obs_val) * log(pow_val_less1)
              + my_z_val * (1-my_old_y_val) * my_obs_val * log(1-pow_val)
              + my_z_val * (1-my_old_y_val) * (1-my_obs_val) * log(pow_val);

    lP_z_on += (1-my_z_val) * my_y_val * my_obs_val * log(1-pow_val_plus1)
                 +(1-my_z_val) * my_y_val * (1-my_obs_val) *log(pow_val_plus1)
                 +(1-my_z_val) * (1-my_y_val) *my_obs_val * log(1-pow_val)
                 +(1-my_z_val) * (1-my_y_val) *(1-my_obs_val) * log(pow_val)
                 +my_z_val*my_y_val*my_obs_val *  log(1-pow_val)
                 +my_z_val*my_y_val*(1-my_obs_val)*log(pow_val)
                 +my_z_val*my_y_val*(1-my_old_y_val)*my_obs_val *log(1-pow_val_plus1)
                 +my_z_val*my_y_val*(1-my_old_y_val)*(1-my_obs_val) *log(pow_val_plus1)
                 +my_z_val*(1-my_y_val) * my_old_y_val *my_obs_val * log(1-pow_val_less1)
                 +my_z_val*(1-my_y_val) * my_old_y_val *(1-my_obs_val) * log(pow_val_less1);

  }

  // w/o only needs to happen for one r
  if (rth == 0) {
    lp_nk_off[nth*K+kth] = lP_z_off;

  }

  lp_nkr_on[nth*K*D+kth*D+rth] = lP_z_on;
  //lp_nkr_on[nth*K*D+kth*D+rth] = nth*K*D+kth*D+rth;
  //printf("cur index of lp_nkr: %d\n", nth*K*D+kth*D+rth);
}
/*
// NOTE this is unused anymore. now uses the max reduce f'n
// split to separate f'ns for synch issues...
kernel void sample_rz_noscOvRP2(global int* cur_y, global int* cur_z, global int *cur_r, global int* obj_recon,
                          global float* lp_nkr_on, global float* lp_nk_off, global float * lp_nk_rmax,
                          global float* lp_rsums, global int* z_col_sum, global int * obs,
                          global float* randForZ, global float* randForR,
                          uint N, uint D, uint K, uint f_img_width,
                          float lambda, float epislon, float theta) {

  const uint V_SCALE = 0, H_SCALE = 1, V_TRANS = 2, H_TRANS = 3, NUM_TRANS = 4;
  uint nth = get_global_id(0);
  uint kth = get_global_id(1);
  uint rth = get_global_id(2);

  if ((nth==0) && (kth == 0) && (rth==0)) {
   //printf("one started part 2\n");
   }

  uint f_img_height = D/f_img_width;

  uint my_z_val = cur_z[nth*K+kth];
  uint v_tr = rth % f_img_width;
  uint h_tr = rth / f_img_width;
  uint t_index = (v_tr) %f_img_height*f_img_width+(h_tr)%f_img_width;

  uint old_v_tr = cur_r[nth*(K*NUM_TRANS) + kth*NUM_TRANS +V_TRANS];
  uint old_h_tr = cur_r[nth*(K*NUM_TRANS) + kth*NUM_TRANS +H_TRANS];
  uint t_old_index = (old_v_tr) % f_img_height*f_img_width+(old_h_tr)%f_img_width;
  uint my_z_sum = z_col_sum[kth] - my_z_val;
  uint my_y_val = cur_y[kth*D+t_index];
  uint my_old_y_val = cur_y[kth*D+t_old_index];

  uint myNKInd =nth*K*D+kth*D;

    // now we have prob of on vs. off for every translation over N (objects) and D (pixels)
    // get max for log trick
  float mylPValPreLT = lp_nkr_on[nth*K*D + kth*D+rth];

  fatomic_max((lp_nk_rmax+nth*K + kth), mylPValPreLT);
  if ((nth==0) && (kth == 0) && (rth==0)) {
   printf("one reached end of part 2\n");
   }
}*/

// split to separate f'ns for synch issues...
kernel void rz_do_log_trick(global float* lp_nkr_on, global float * lp_nk_rmax,
                            global float* lp_nk_sum, uint N, uint K, uint D) {

  const uint  V_TRANS = 0, H_TRANS = 1, NUM_TRANS = 2;
  uint nth = get_global_id(0);
  uint kth = get_global_id(1);
  uint rth = get_global_id(2);
  uint myNKInd =nth*K*D+kth*D;

  float mylPValPreLT = lp_nkr_on[myNKInd+rth];
  //float myLogMax = lp_nk_rmax[nth*K+kth];

  float myLogMax = lp_nk_rmax[myNKInd];
  // do log trick and exponentiate

  float myPostLTVal = native_powr(exp(1.0f), mylPValPreLT-myLogMax);
  lp_nkr_on[myNKInd+rth] = myPostLTVal;
  lp_nk_sum[myNKInd+rth] = myPostLTVal;
}

kernel void calc_lps(global int * obj_recon, global int* obs, global float* lps,
                    uint N, uint D, float lambda, float epislon) {

  uint nth = get_global_id(0);
  uint dth = get_global_id(1);

  int my_obs_ind = nth*D+dth;

  int my_recon = obj_recon[my_obs_ind];
  int my_obs = obs[my_obs_ind];

  float pow_val = native_powr(1-lambda,my_recon) * (1-epislon);
  lps[my_obs_ind] = my_obs * log(1-pow_val) + (1-my_obs) * log(pow_val);
}


kernel void calc_lp_fornew(global int* obj_recon, global int* obs, global float* lps,
                    uint N, uint K, uint D, uint KNewMax,
                    float lambda, float epislon, float theta) {

  int nth = get_global_id(0);
  int dth = get_global_id(1);
  int k_newth = get_global_id(2);

  int my_obs_idx = nth*D+dth;
  int my_obs_val = obs[my_obs_idx];
  int my_pred = obj_recon[my_obs_idx];

  int mylp_ind = nth*D*KNewMax+dth*KNewMax + k_newth;

  float pow_val = (1-epislon) * native_powr(1-lambda, my_pred) * native_powr(1-lambda*theta, k_newth);
  lps[mylp_ind] = my_obs_val * log(1-pow_val) + (1-my_obs_val) * log(pow_val);
}


kernel void calc_y_lps_old(local float* locmem, global int* cur_y, global int *cur_z, global int* cur_recon,
		       global int* obs, global float* lp_off, global float* lp_on,
		       uint N, uint D, uint K, uint numPrevRun,
		       float lambda, float epislon, float theta) {
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
  float pow_val = native_powr(1-lambda, my_recon) * (1-epislon);
  float pow_val_less1 = native_powr(1-lambda, max(my_recon-1,0)) * (1-epislon);
  float pow_val_plus1 = native_powr(1-lambda, my_recon+1) * (1-epislon);

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

kernel void calc_y_lp_off(local float* locmem, global int* cur_y, global int *cur_z, global int* cur_recon,
		       global int* obs, global float* lp_off,
		       uint N, uint D, uint K, uint numPrevRun,
		       float lambda, float epislon, float theta) {
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

  float pow_val = native_powr(1-lambda, my_recon) * (1-epislon);
  float pow_val_less1 = native_powr(1-lambda, max(my_recon-1,0)) * (1-epislon);
  float pow_val_plus1 = native_powr(1-lambda, my_recon+1) * (1-epislon);

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
		       float lambda, float epislon, float theta) {
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

  float pow_val = native_powr(1-lambda, my_recon) * (1-epislon);
  float pow_val_less1 = native_powr(1-lambda, max(my_recon-1,0)) * (1-epislon);
  float pow_val_plus1 = native_powr(1-lambda, my_recon+1) * (1-epislon);

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


// this might be wrong and need to take into account the transformations itself
// i think it does now

kernel void calc_y_probs(local int* ourPixs, global float* nkd_yprob_on, global float* nkd_yprob_off,
			 global int* obj_recon, global int* objs, global int* cur_z, global int* cur_y, global int* cur_r,
			 uint N, uint K, uint D, uint startInd, uint f_img_width,
			 float lambda, float epislon, float theta) {
  const uint V_SCALE = 0, H_SCALE = 1, V_TRANS = 2, H_TRANS = 3, NUM_TRANS = 4;

  int group_size = get_local_size(0);
  int lid = get_local_id(0);
  int kth = get_global_id(0) + startInd;
  int nth = get_global_id(1);
  int dth = get_global_id(2);
  uint f_img_height = D/f_img_width;

  // ourPixs[0] = obs. image pixel
  // ourPixs[1] = current reconstructed pixel (# of features with this pixel on)
  // make sure this indexing is right!!
  uint myNKDInd = nth*K*D+kth*D+dth;

  if ((kth == 0) && (kth < K)) {
    ourPixs[0] = objs[nth*D+dth];
    ourPixs[1] = obj_recon[nth*D+dth];
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  if (kth < K) {
    int my_w_off = cur_r[nth*(K*NUM_TRANS) + kth*NUM_TRANS + H_TRANS];
    int my_h_off = cur_r[nth*(K*NUM_TRANS) + kth*NUM_TRANS + V_TRANS];
    int my_trans_d = (my_w_off % f_img_height) * f_img_width + my_h_off % f_img_width;

    int my_obs_val = ourPixs[0];
    int cur_pred = ourPixs[1];
    int my_z_val = cur_z[nth*K+kth];
    int my_y_val = cur_y[kth*D + my_trans_d];

    //everything otherwise just happens in parallel and sent to be reduced in next kernel call
    float myOnVal = log(theta);
    float myOffVal = log(1-theta);

    //TODO: process these durign local step and store as int/ retreive as float via casting
    float pow_val = (1-epislon)* native_powr(1-lambda, cur_pred);
    float pow_val_less1 = (1-epislon) *native_powr(1-lambda, max(cur_pred-1,0));
    float pow_val_plus1 = (1-epislon) *native_powr(1-lambda, cur_pred+1);

    myOffVal += my_z_val*my_obs_val*my_y_val*log(1-pow_val_less1)
              + my_z_val*my_obs_val*(1-my_y_val)*log(1-pow_val)
              + my_z_val*(1-my_obs_val)*my_y_val*log(pow_val_less1)
              + my_z_val*(1-my_obs_val)*(1-my_y_val)*log(pow_val)
              + (1-my_z_val)*my_obs_val*log(1-pow_val)
              + (1-my_z_val)*(1-my_obs_val)*log(pow_val);

    myOnVal += my_z_val*my_obs_val*my_y_val*log(1-pow_val)
            +  my_z_val*my_obs_val*(1-my_y_val) *log(1-pow_val_plus1)
            +  my_z_val*(1-my_obs_val)*my_y_val*log(pow_val)
            +  my_z_val*(1-my_obs_val)*(1-my_y_val)*log(pow_val_plus1)
            + (1-my_z_val)*my_obs_val*log(1-pow_val)
            + (1-my_z_val)*(1-my_obs_val)*log(pow_val);
    nkd_yprob_off[myNKDInd] = myOffVal;
    nkd_yprob_on[myNKDInd] = myOnVal;
  }
}

kernel void new_y_val_probs2(local int* locmem, global float* cur_z, global float* cur_y,
			     global int* comb_vec, global int* obj_recon,
			     global int* obs, global float *new_probs,
			     uint nth, uint new_k, uint N, uint D, uint K,
			     float lambda, float epislon, float theta) {
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

    float lik_pow_val = native_powr(1-lambda,my_recon_val+newKth) * (1-epislon);
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


kernel void sample_y_kd(global float* kd_ylp_on, global float* kd_ylp_off, global int* cur_y,
                        global float* rand_y, int N, int K, int D) {
  int kth = get_global_id(0);
  int dth = get_global_id(1);

  float mylpOn = kd_ylp_on[kth*D+dth];
  float mylpOff = kd_ylp_off[kth*D+dth];
  float myThresh = 1.0f / (1.0f + native_powr(exp(1.0f), mylpOn-mylpOff));
  cur_y[kth*D+dth] = (myThresh <= rand_y[kth*D+dth]);

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

kernel void new_y_val_probs(local int* locmem, global float* cur_z, global float* cur_y,  global float* cur_r,
                              global float* comb_vec, global float* obj_recon,
                              global float* obs, global float * new_probs,
                              uint N, uint K, uint D, uint newK,
                              float lambda, float epislon, float theta) {
  const int MY_OBS = 0, MY_RECON = 1, NEW_K_START = 2;
  int dth = get_global_id(0);
  int nth = get_global_id(1);
  int newKth = get_global_id(2);
  int lid = get_local_id(0);

  int my_obs_idx = nth*D+dth;

  if (lid == 0) {
    locmem[MY_OBS] = obs[my_obs_idx];
    locmem[MY_RECON] = obj_recon[my_obs_idx];
  }
  if (lid < newK) {
    locmem[NEW_K_START+lid] = comb_vec[lid];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  int my_obs_val = locmem[MY_OBS];
  int my_recon_val = locmem[MY_RECON];
  int my_knew_choose_val = locmem[NEW_K_START + newKth];
  float lik_pow_val = native_powr(1-lambda,my_recon_val+newKth) * (1-epislon);
  float pr_pow_val = my_knew_choose_val * native_powr(theta, newKth) * native_powr(1-theta, newK-newKth);

  new_probs[newKth*D+dth] = my_obs_val * (1-lik_pow_val) * pr_pow_val +
                            (1-my_obs_val) * lik_pow_val * pr_pow_val;

}

//from pg 229-231 of opencl in action
 //TODO: rewrite for float4s
kernel void reduc_vec_sum3d(local float* part_sums, global float* data, global float* out,
                          uint D1, uint D2, uint D3, uint startInd){

  int group_size = get_local_size(0);
  int lid = get_local_id(0);
  int rth = get_global_id(0) + startInd;
  int nth = get_global_id(1);
  int kth = get_global_id(2);

  uint myNKInd = nth*D2*D3+kth*D3;
  if (rth < D3) {
    part_sums[lid] = data[myNKInd+rth];
  }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = group_size/2; i > 0; i >>= 1) {
      if  ((lid < i) && ((rth+i)< D3)) {
//      if ((nth == 1) && (kth == 1) ) {
//        printf("rth %d, lid %d, and i %d currently summing indices %d and %d with values %f and %f\n",
 //              rth, lid, i, lid+startInd, startInd+lid+i, part_sums[lid],part_sums[lid+i]);
 //     }
        part_sums[lid] +=  part_sums[lid+i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }


  if (lid == 0) {
     out[myNKInd + rth/group_size] = part_sums[0];
     if (startInd > 0) {
     // get rid of the empty spot
        out[myNKInd+ rth] = 0.;
     }

    }
    else if (rth < D3) {
    out[myNKInd+rth] = 0.;
   }
}

kernel void finish_reduc_vec_sum3d(local float* part_sums, global float* data, global float* sum,
                                  int D1, int D2, int D3, int numLeft) {
  int lid = get_local_id(0);
  int group_size = get_local_size(0);
  int rth = get_global_id(0);
  int nth = get_global_id(1);
  int kth = get_global_id(2);

  int myNKInd = nth*D2*D3+kth*D3;

  if ((lid <= numLeft) && (rth <= numLeft)) {

    part_sums[lid] = data[myNKInd + rth];
 //   if ((nth == 1) && (kth == 1))
 //  printf("in finish: lid %d, rth %d, kth %d nth %d myNKInd %d part_sums val %f\n", lid, rth, kth, nth, myNKInd, part_sums[lid]);
  }
  else {
    part_sums[lid] = 0.; // for safety
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = group_size/2; i > 0; i >>=1) {
         if ((lid < i) && ((lid+i) <= numLeft))  {
        //if ((nth == 1) && (kth == 1)) {
        //           printf("rth %d, lid %d, and i %d currently summing indices %d and %d with values %f and %f\n",
      //                  rth, lid, i, lid, lid+i, part_sums[lid],part_sums[lid+i]);
          // }


          part_sums[lid] += part_sums[lid+i];

        }
        barrier(CLK_LOCAL_MEM_FENCE);
       }
    if (lid == 0) {
  //      printf("nth %d, kth %d, rth %d, total sum = %f\n", nth, kth, rth, part_sums[0]);
        sum[myNKInd] = part_sums[0];
    }
}


 //TODO: rewrite for float4s
kernel void reduc_vec_max3d(local float* parts, global float* data, global float* out,
                          uint D1, uint D2, uint D3, uint startInd){

  int group_size = get_local_size(0);
  int lid = get_local_id(0);
  int rth = get_global_id(0) + startInd;
  int nth = get_global_id(1);
  int kth = get_global_id(2);

  uint myNKInd = nth*D2*D3+kth*D3;
  if (rth < D3) {
    parts[lid] = data[myNKInd+rth];
  }
    barrier(CLK_LOCAL_MEM_FENCE);



    for (int i = group_size/2; i > 0; i >>= 1) {
      if  ((lid < i) && ((rth+i)< D3)) {
//      if ((nth == 1) && (kth == 1) ) {
//        printf("rth %d, lid %d, and i %d currently summing indices %d and %d with values %f and %f\n",
 //              rth, lid, i, lid+startInd, startInd+lid+i, part_sums[lid],part_sums[lid+i]);
 //     }
        float other = parts[lid+i];
        float mine = parts[lid];
        // switch to min/max operation?
        parts[lid] = (mine > other) ? mine:other;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }


  if (lid == 0) {
     out[myNKInd + rth/group_size] = parts[0];
     if (startInd > 0) {
     // get rid of the empty spot
        out[myNKInd+ rth] = -9000000.;
     }

    }
    else if (rth < D3) {
    out[myNKInd+rth] = -9000000;
   }

}



kernel void finish_reduc_vec_max3d(local float* parts, global float* data, global float* out,
                                  int D1, int D2, int D3, int numLeft) {
  int lid = get_local_id(0);
  int group_size = get_local_size(0);
  int rth = get_global_id(0);
  int nth = get_global_id(1);
  int kth = get_global_id(2);

  int myNKInd = nth*D2*D3+kth*D3;

  if ((lid < numLeft) && (rth < numLeft)) {
//    printf("nth %d; kth: %d; rth: %d; myNKInd: %d, data_id: %d; data_val: %f\n",
//              nth, kth, rth, myNKInd, myNKInd+rth, data[myNKInd+rth]);
    parts[lid] = data[myNKInd + rth];
   // if ((nth == 1) && (kth == 1))
    //printf("in finish: lid %d, rth %d, kth %d nth %d myNKInd %d part_sums val %f\n", lid, rth, kth, nth, myNKInd, part_sums[lid]);
  }
  else {
    parts[lid] = -9000000.; // for safety
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int i = group_size/2; i > 0; i >>=1) {
         if ((lid < i) && ((lid+i) <= numLeft))  {
        //if ((nth == 1) && (kth == 1)) {
        //           printf("rth %d, lid %d, and i %d currently summing indices %d and %d with values %f and %f\n",
      //                  rth, lid, i, lid, lid+i, part_sums[lid],part_sums[lid+i]);
          // }
      float other = parts[lid+i];
        float mine = parts[lid];
        // switch to min/max operation?
        parts[lid] = (mine > other) ? mine:other;


        }
        barrier(CLK_LOCAL_MEM_FENCE);
       }
    if (lid == 0) {
  //      printf("nth %d, kth %d, rth %d, total sum = %f\n", nth, kth, rth, part_sums[0]);
        out[myNKInd] = parts[0];
    }
}

 // split to separate f'ns for synch issues...
kernel void sample_rz_noscOvRP4b(global int* cur_y, global int* cur_z, global int *cur_r, global int* obj_recon,
                          global float* lp_nkr_on, global float* lp_nk_off, global float * lp_nk_rmax,
                          global float* lp_rsums, global int* z_col_sum, global int * obs,
                          global float* randForZ, global float* randForR,
                          uint N, uint D, uint K, uint f_img_width,
                          float lambda, float epislon, float theta) {

  const uint V_SCALE = 0, H_SCALE = 1, V_TRANS = 2, H_TRANS = 3, NUM_TRANS = 4;
  uint nth = get_global_id(0);
  uint kth = get_global_id(1);
  uint rth = get_global_id(2);

    //if ((nth==0) && (kth == 0) && (rth==0)) {
  // printf("one started part 4\n");
   //}


    /*uint f_img_height = D/f_img_width;

    uint my_z_val = cur_z[nth*K+kth];
  uint v_tr = rth % f_img_width;
  uint h_tr = rth / f_img_width;
  uint t_index = (v_tr) %f_img_height*f_img_width+(h_tr)%f_img_width;

  uint old_v_tr = cur_r[nth*(K*NUM_TRANS) + kth*NUM_TRANS +V_TRANS];
  uint old_h_tr = cur_r[nth*(K*NUM_TRANS) + kth*NUM_TRANS +H_TRANS];
  uint t_old_index = (old_v_tr) % f_img_height*f_img_width+(old_h_tr)%f_img_width;
  uint my_z_sum = z_col_sum[kth] - my_z_val;
  uint my_y_val = cur_y[kth*D+t_index];
  uint my_old_y_val = cur_y[kth*D+t_old_index];*/

  uint myNKInd =nth*K*D+kth*D;


    // at end lp_rsums[nth*K*D+kth*D] has total sum
    //normalize all
    float myNorm = lp_rsums[myNKInd];
    //printf("sum is %f, next is %f, next is %f \n", myNorm, lp_rsums[nth*K*D+kth*D+1], lp_rsums[nth*K*D+kth*D+2]);
    //if (isnan(myNorm)) {
    //   printf("my norm is nan at nth %d, kth %d, with myNKInd %d\n", nth, kth, myNKInd);
    //}
    //if ((myNorm <= 0.0f) || (myNorm > 1.0f)) {
    //   printf("my norm is too close to zero or bigger than one (%f) at nth %d, kth %d, with myNKInd %d\n", myNorm, nth, kth, myNKInd);
    //}
    if (fEqualsP(myNorm, 0.0f)) {
      lp_nkr_on[myNKInd+rth] = 0.0f;
    }
    else {
      lp_nkr_on[myNKInd+rth] = lp_nkr_on[myNKInd+rth]/myNorm;
      }

      /*if ((lp_nkr_on[myNKInd+rth] < 0.0f) || (lp_nkr_on[myNKInd+rth] > 1.0f)) {
        printf("bad prob val %f, at %d\n", lp_nkr_on[myNKInd+rth], (myNKInd+rth));
      }*/

 //  if ((nth==0) && (kth == 0) && (rth==0)) {
   //printf("one reached end of part 4\n");
  // }
}

// split to separate f'ns for synch issues...
kernel void sample_rz_noscOvRP5(global int* cur_y, global int* cur_z, global int *cur_r, global int* obj_recon,
                          global float* lp_nkr_on, global float* lp_nk_off, global float * lp_nk_rmax,
                          global float* lp_rsums, global int* z_col_sum, global int * obs,
                          global float* randForZ, global float* randForR,
                          uint N, uint D, uint K, uint f_img_width,
                          float lambda, float epislon, float theta) {

  const uint V_SCALE = 0, H_SCALE = 1, V_TRANS = 2, H_TRANS = 3, NUM_TRANS = 4;
  uint nth = get_global_id(0);
  uint kth = get_global_id(1);
   //printf("my nth: %d kth: %d\n", nth, kth);

  uint f_img_height = D/f_img_width;

  uint my_z_val = cur_z[nth*K+kth];

  uint my_z_sum = z_col_sum[kth] - my_z_val;
  uint myNKInd =nth*K*D+kth*D;
  float myNorm = lp_rsums[myNKInd];
  float myLogMax = lp_nk_rmax[nth*K+kth];

  float myLPZOff = lp_nk_off[nth*K+kth] + log(1.0f-((float)my_z_sum)/N);
  float myLPZOn = log(myNorm) + log(1.0f-((float)my_z_sum)/N) + myLogMax;

  //if (isnan(myLPZOff) || isnan(myLPZOn)) {

  //}
  float myThresh = 1.0f / (1.0f + native_powr(exp(1.0f),myLPZOn-myLPZOff));
//  printf("myLPZOff %f, myLPZOn %f, myThresh %f, randZ %f,  for n %d k %d: \n",
 //           myLPZOff, myLPZOn,myThresh, randForZ[nth*K+kth], nth, kth);

  if (myThresh > randForZ[nth*K + kth]) {
      cur_z[nth*K+kth] = 0;
  }
  else {
     cur_z[nth*K+kth] = 1;
     float curP = 0.0f;
      uint curR = 0;
      float myRRand = randForR[nth*K + kth];
      bool firstBad = false;
      while ((curP <= myRRand)&&(curR < D)) {
        curP += lp_nkr_on[myNKInd+curR];
        //if ((firstBad == false) && (curP < 0)) {
   //     if ((nth==0) && (kth==0)) printf("curP: %f; myRRand %f; curR %d\n", curP, myRRand, curR);
       //   firstBad = true;
       //   }
        curR++;
      }
  //    if (curR == D) {
   //     printf("HIT D curP: %f; myRRand %f; curR %d\n", curP, myRRand, curR);
    //  }
      cur_r[nth*K*NUM_TRANS+kth*NUM_TRANS+V_TRANS] = curR % f_img_width;
      cur_r[nth*K*NUM_TRANS+kth*NUM_TRANS+H_TRANS] =  curR / f_img_width;
   }
}

kernel void sample_zr_precalc(global int* cur_z, global int* cur_r, global float* lp_on_max,
                              global float* lp_on_sums, global float* lp_on, global float* lp_off,
                              global int* z_col_sum, global float* rand_z, global float* rand_r,
                              uint N, uint K, uint D, float T, int f_img_width) {

    const uint V_TRANS = 0, H_TRANS = 1, NUM_TRANS = 2;

    int nth = get_global_id(0);
    int kth = get_global_id(1);
    int my_nk_ind = nth*K*D+kth*D;
    int my_z_ind = nth*K + kth;
    int my_z_val = 0;
    if ((nth < N) && (kth < K)) {
        my_z_val = cur_z[my_z_ind];
        int mk = z_col_sum[kth]-my_z_val;
        float lpPr = (1.0*mk)/N;
        float lpOff = lp_off[my_z_ind] + log(1-lpPr);
        float lpOn = log(lp_on_sums[my_nk_ind]) + lp_on_max[my_nk_ind] + log(lpPr);

        if (isinf(lpOff)) {
            my_z_val = 1;
        }
        else if (!isinf(lpOn)){
             float logpost[2] = {lpOn * T, lpOff*T};
             uint labels[2] = {1, 0};

             lognormalize(logpost, 0, 2);
             //printf("nth: %d; kth %d; pOn: %f; pOff %f\n", nth,kth,logpost[0],logpost[1]);
             my_z_val = sample(2, labels, logpost, 0, rand_z[my_z_ind]);
        }
        cur_z[my_z_ind] = my_z_val;
        if (my_z_val) {
            int r = 0;
            float rand_val = rand_r[my_z_ind];
            float curP = lp_on[my_nk_ind+r];
            while ((curP < rand_val) && (r < D)) {
              r++;
              curP+=lp_on[my_nk_ind+r];
            }
            //printf("r: %d; nk_ind: %f")
            cur_r[nth*K*NUM_TRANS+kth*NUM_TRANS+H_TRANS] = r/f_img_width;
            cur_r[nth*K*NUM_TRANS+kth*NUM_TRANS+V_TRANS] = r%f_img_width;
        }
    }

}







kernel void compute_z_by_ry(global int *cur_y, global int *cur_z, global int *cur_r,
			    global int *transformed_y, global int *temp_y, global int *z_by_ry,
			    uint N, uint D, uint K, uint f_img_width) {

  const uint V_SCALE = 0, H_SCALE = 1, V_TRANS = 2, H_TRANS = 3, NUM_TRANS = 4;
  uint nth = get_global_id(0); // nth is the index of images
  uint kth = get_global_id(1); // kth is the index of features
  uint f_img_height = D / f_img_width;
  
  if (cur_z[nth * K + kth] == 0) {
    for (int dth = 0; dth < D; dth++) {
      transformed_y[nth * K * D + kth * D + dth] = 0;
    }
  } else {

  for (int dth = 0; dth < D; dth++) {
    temp_y[nth * K * D + kth * D + dth] = cur_y[kth * D + dth];
  }
  
  // vertically scale the feature image
  uint v_scale = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + V_SCALE];
  scale_global(temp_y, transformed_y, nth, kth, f_img_height, f_img_width, 0, v_scale, K, D);
  for (int dth = 0; dth < D; dth++) {
    temp_y[nth * K * D + kth * D + dth] = transformed_y[nth * K * D + kth * D + dth];
  }
  
  // horizontal scale the feature image
  uint h_scale = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + H_SCALE];
  scale_global(temp_y, transformed_y, nth, kth, f_img_height, f_img_width, h_scale, 0, K, D);
  for (int dth = 0; dth < D; dth++) {
    temp_y[nth * K * D + kth * D + dth] = transformed_y[nth * K * D + kth * D + dth];
  }

  // vertically translate the feature image
  uint v_dist = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + V_TRANS];
  v_translate_global(temp_y, transformed_y, nth, kth, f_img_height, f_img_width, v_dist, K, D);
  for (int dth = 0; dth < D; dth++) {
    temp_y[nth * K * D + kth * D + dth] = transformed_y[nth * K * D + kth * D + dth];
  }
  
  // horizontally translate the feature image
  uint h_dist = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + H_TRANS];
  h_translate_global(temp_y, transformed_y, nth, kth, f_img_height, f_img_width, h_dist, K, D);
  }
  // wait until copying is done
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 

  
  /* at this point, for each object, a transformed y has been generated */
  
  if (kth == 0) {
    for (int dth = 0; dth < D; dth++) {
      z_by_ry[nth * D + dth] = 0;
      for (int k = 0; k < K; k++) {
	z_by_ry[nth * D + dth] += transformed_y[nth * K * D + k * D + dth] * cur_z[nth * K + k];
      }
    }
  }
}

kernel void compute_z_by_ry_nosc(global int *cur_y, global int *cur_z, global int *cur_r,
			    global int *transformed_y, global int *temp_y, global int *z_by_ry,
			    uint N, uint D, uint K, uint f_img_width) {

  const uint V_SCALE = 0, H_SCALE = 1, V_TRANS = 2, H_TRANS = 3, NUM_TRANS = 4;
  uint nth = get_global_id(0); // nth is the index of images
  uint kth = get_global_id(1); // kth is the index of features
  uint f_img_height = D / f_img_width;

  if (cur_z[nth * K + kth] == 0) {
    for (int dth = 0; dth < D; dth++) {
      transformed_y[nth * K * D + kth * D + dth] = 0;
    }
  } else {

  for (int dth = 0; dth < D; dth++) {
    temp_y[nth * K * D + kth * D + dth] = cur_y[kth * D + dth];
  }

  // vertically translate the feature image
  uint v_dist = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + V_TRANS];
  v_translate_global(temp_y, transformed_y, nth, kth, f_img_height, f_img_width, v_dist, K, D);
  for (int dth = 0; dth < D; dth++) {
    temp_y[nth * K * D + kth * D + dth] = transformed_y[nth * K * D + kth * D + dth];
  }

  // horizontally translate the feature image
  uint h_dist = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + H_TRANS];
  h_translate_global(temp_y, transformed_y, nth, kth, f_img_height, f_img_width, h_dist, K, D);
  }
  // wait until copying is done
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);


  /* at this point, for each object, a transformed y has been generated */

  if (kth == 0) {
    for (int dth = 0; dth < D; dth++) {
      z_by_ry[nth * D + dth] = 0;
      for (int k = 0; k < K; k++) {
	z_by_ry[nth * D + dth] += transformed_y[nth * K * D + k * D + dth] * cur_z[nth * K + k];
      }
    }
  }
}

kernel void compute_z_by_ry_local(global int *cur_y, global int *cur_z, global int *cur_r,
				  global int *z_by_ry, local int *orig_y, local int *new_y,
				  uint N, uint D, uint K, uint f_img_width) {

  const uint V_SCALE = 0, H_SCALE = 1, V_TRANS = 2, H_TRANS = 3, NUM_TRANS = 4;
  uint nth = get_global_id(0); // nth is the index of images
  uint kth = get_global_id(1); // kth is the index of features
  uint f_img_height = D / f_img_width;

  // copy the original feature image to local memory
  for (int dth = 0; dth < D; dth++) {
    orig_y[kth * D + dth] = cur_y[kth * D + dth];
  }
  // wait until copying is done
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 
    
  // vertically scale the feature image
  uint v_scale = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + V_SCALE];
  scale(orig_y, new_y, kth, f_img_height, f_img_width, 0, v_scale, D);

  // copy new_y back to orig_y so that new_y can be used again
  for (int dth = 0; dth < D; dth++) {
    orig_y[kth * D + dth] = new_y[kth * D + dth];
  }
  // wait until copying is done
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE); 

  // horizontal scale the feature image
  uint h_scale = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + H_SCALE];
  scale(orig_y, new_y, kth, f_img_height, f_img_width, h_scale, 0, D);
  
  // copy new_y back to orig_y so that new_y can be used again
  for (int dth = 0; dth < D; dth++) {
    orig_y[kth * D + dth] = new_y[kth * D + dth];
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

  /* at this point, for each object, a transformed y (new_y) has been generated */
  if (kth == 0) {
    for (int dth = 0; dth < D; dth++) {
      z_by_ry[nth * D + dth] = 0;
      for (int k = 0; k < K; k++) {
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
  
  const uint V_SCALE = 0, H_SCALE = 1, V_TRANS = 2, H_TRANS = 3, NUM_TRANS = 4;

  uint kth = get_global_id(0); // k is the index of features
  uint dth = get_global_id(1); // d is the index of pixels

  uint f_img_height = D / f_img_width;

  // unpack dth into h and w
  uint h = dth / f_img_width;
  uint w = dth % f_img_width;

  // calculate the prior probability of each cell is 1
  float on_loglik_temp = log(theta); 
  float off_loglik_temp = log(1 - theta);

  int v_scale, h_scale, v_dist, h_dist, new_height, new_width, new_index, n, hh, ww;
  // extremely hackish way to calculate the loglikelihood
  for (n = 0; n < N; n++) {
    // if the nth object has the kth feature
    if (cur_z[n * K + kth] == 1) {
      // retrieve the transformation applied to this feature by this object
      v_scale = cur_r[n * (K * NUM_TRANS) + kth * NUM_TRANS + V_SCALE];
      h_scale = cur_r[n * (K * NUM_TRANS) + kth * NUM_TRANS + H_SCALE];
      v_dist = cur_r[n * (K * NUM_TRANS) + kth * NUM_TRANS + V_TRANS];
      h_dist = cur_r[n * (K * NUM_TRANS) + kth * NUM_TRANS + H_TRANS];
      new_height = f_img_height + v_scale;
      new_width = f_img_width + h_scale;

      // loop over all pixels
      for (hh = 0; hh < f_img_height; hh++) {
	for (ww = 0; ww < f_img_width; ww++) {
	  if ((int)round((float)hh / new_height * f_img_height) == h &
	      (int)round((float)ww / new_width * f_img_width) == w) {
	    new_index = ((v_dist + hh) % f_img_height) * f_img_width + (h_dist + ww) % f_img_width;

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
	    } else { // else obs[n * D + new_index] == 0
	      on_loglik_temp += log(1 - lambda);
	      off_loglik_temp += log(1.0f);
	    }
	  }
	}
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

kernel void sample_y_nosc(global int *cur_y,
		     global int *cur_z,
		     global int *z_by_ry,
		     global int *cur_r,
		     global int *obs,
		     global float *rand,
		     uint N, uint D, uint K, uint f_img_width,
		     float lambda, float epislon, float theta) {

  const uint V_SCALE = 0, H_SCALE = 1, V_TRANS = 2, H_TRANS = 3, NUM_TRANS = 4;

  uint kth = get_global_id(0); // k is the index of features
  uint dth = get_global_id(1); // d is the index of pixels

  uint f_img_height = D / f_img_width;

  // unpack dth into h and w
  uint h = dth / f_img_width;
  uint w = dth % f_img_width;

  // calculate the prior probability of each cell is 1
  float on_loglik_temp = log(theta);
  float off_loglik_temp = log(1 - theta);

  int v_dist, h_dist, new_index, n;

  int cur_y_val = cur_y[kth*D+dth];

  // extremely hackish way to calculate the loglikelihood
  for (n = 0; n < N; n++) {
    // if the nth object has the kth feature
    if (cur_z[n * K + kth] == 1) {
      // retrieve the transformation applied to this feature by this object
      //v_scale = cur_r[n * (K * NUM_TRANS) + kth * NUM_TRANS + V_SCALE];
      //h_scale = cur_r[n * (K * NUM_TRANS) + kth * NUM_TRANS + H_SCALE];
      v_dist = cur_r[n * (K * NUM_TRANS) + kth * NUM_TRANS + V_TRANS];
      h_dist = cur_r[n * (K * NUM_TRANS) + kth * NUM_TRANS + H_TRANS];
      //new_height = f_img_height + v_scale;
      //new_width = f_img_width + h_scale;

      new_index = (v_dist) % f_img_height * f_img_width + (h_dist) % f_img_width;
      int cur_obs = obs[n*D + new_index];
      int cur_z_val = z_by_ry[n*D+new_index];
      float cur_pow = pow(1-lambda, cur_z_val) * (1-epislon);
      float cur_pow_less1 = cur_pow/(1-lambda);
      float cur_pow_plus1 = cur_pow * (1-lambda);

      on_loglik_temp += cur_obs * cur_y_val * log(1-cur_pow) + cur_obs * (1-cur_y_val) * log(1-cur_pow_plus1)
                        + (1-cur_obs) * (1-lambda);
      off_loglik_temp += cur_obs * cur_y_val * log(1-cur_pow_less1) + cur_obs*(1-cur_y_val) * log(1-cur_pow);

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
  
  const uint V_SCALE = 0, H_SCALE = 1, V_TRANS = 2, H_TRANS = 3, NUM_TRANS = 4;
  uint h, w, new_index; // variables used in the for loop

  uint nth = get_global_id(0); // n is the index of data
  uint kth = get_global_id(1); // k is the index of features

  uint f_img_height = D / f_img_width;

  // calculate the prior probability of each cell is 1
  float on_prob_temp = (z_col_sum[kth] - cur_z[nth * K + kth]) / (float)N; 
  float off_prob_temp = 1 - (z_col_sum[kth] - cur_z[nth * K + kth]) / (float)N;

  // retrieve the transformation applied to this feature by this object
  int v_scale = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + V_SCALE];
  int h_scale = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + H_SCALE];
  int v_dist = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + V_TRANS];
  int h_dist = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + H_TRANS];
  int new_height = f_img_height + v_scale, new_width = f_img_width + h_scale;
  
  uint d, hh, ww;
  // extremely hackish way to calculate the likelihood
  for (d = 0; d < D; d++) {
    // if the kth feature can turn on a pixel at d
    if (cur_y[kth * D + d] == 1) {
      // unpack d into h and w and get new index
      h = d / f_img_width;
      w = d % f_img_width;

      for (hh = 0; hh < f_img_height; hh++) {
	for (ww = 0; ww < f_img_width; ww++) {
	  if ((int)round((float)hh / new_height * f_img_height) == h &
	      (int)round((float)ww / new_width * f_img_width) == w) {
	    new_index = ((v_dist + hh) % f_img_height) * f_img_width + (h_dist + ww) % f_img_width;
      
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

kernel void sample_z_nosc(global int *cur_y,
		     global int *cur_z,
		     global int *cur_r,
		     global int *z_by_ry,
		     global int *z_col_sum,
		     global int *obs,
		     global float *rand,
		     uint N, uint D, uint K, uint f_img_width,
		     float lambda, float epislon, float theta) {

  const uint V_SCALE = 0, H_SCALE = 1, V_TRANS = 2, H_TRANS = 3, NUM_TRANS = 4;
  uint h, w, new_index; // variables used in the for loop

  uint nth = get_global_id(0); // n is the index of data
  uint kth = get_global_id(1); // k is the index of features

  uint f_img_height = D / f_img_width;

  // calculate the prior probability of each cell is 1
  float on_prob_temp = (z_col_sum[kth] - cur_z[nth * K + kth]) / (float)N;
  float off_prob_temp = 1 - (z_col_sum[kth] - cur_z[nth * K + kth]) / (float)N;

  // retrieve the transformation applied to this feature by this object
  //int v_scale = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + V_SCALE];
  //int h_scale = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + H_SCALE];
  int v_dist = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + V_TRANS];
  int h_dist = cur_r[nth * (K * NUM_TRANS) + kth * NUM_TRANS + H_TRANS];
  //int new_height = f_img_height + v_scale, new_width = f_img_width + h_scale;
  int cur_z_val = cur_z[nth * K + kth];
  uint d, hh, ww;
  // extremely hackish way to calculate the likelihood
  for (d = 0; d < D; d++) {
    // if the kth feature can turn on a pixel at d
    if (cur_y[kth * D + d] == 1) {
      // unpack d into h and w and get new index
      //h = d / f_img_width;
      //w = d % f_img_width;


	    new_index = ((v_dist) % f_img_height) * f_img_width + (h_dist) % f_img_width;
	    int cur_obs_val = obs[nth*D+new_index];
	    float cur_pow = pow(1-lambda, z_by_ry[nth*D+new_index]) * (1-epislon);
	    float cur_pow_less1 = cur_pow/(1-lambda);
	    float cur_pow_plus1 = cur_pow*(1-lambda);

	    on_prob_temp*= cur_obs_val*cur_z_val*cur_pow + cur_obs_val*(1-cur_z_val)*cur_pow_plus1
	                    + (1-cur_obs_val) *(1-lambda);
	    off_prob_temp*= cur_obs_val*cur_z_val*cur_pow_less1 + cur_obs_val*(1-cur_z_val)*cur_pow;

	    // then the corresponding observed pixel is at new_index
	    // so, if the observed pixel at new_index is on
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

kernel void sample_r(global int *replace_r, global int *z_by_ry_old, global int *z_by_ry_new,
		     global float *logprior_old, global float *logprior_new,
		     global int *obs, global float *rand,
		     uint N, uint D, uint K,
		     float lambda, float epislon) {

  uint nth = get_global_id(0);
  float loglik_old = 0;
  float loglik_new = 0;
  for (int kth = 0; kth < K; kth++) {
    loglik_old += logprior_old[nth * K + kth];
    loglik_new += logprior_new[nth * K + kth];
  }
  for (int dth = 0; dth < D; dth++) {
    if (obs[nth * D + dth] == 1) {
      loglik_old += log(1 - pow(1 - lambda, z_by_ry_old[nth * D + dth]) * (1 - epislon));
      loglik_new += log(1 - pow(1 - lambda, z_by_ry_new[nth * D + dth]) * (1 - epislon));
    } else {
      loglik_old += log(1 - lambda) * z_by_ry_old[nth * D + dth] + log(1 - epislon);
      loglik_new += log(1 - lambda) * z_by_ry_new[nth * D + dth] + log(1 - epislon);
    }
  }
  float move_prob = 1 / (1 + exp(loglik_old - loglik_new));
  //printf("%f %f\n", loglik_old, loglik_new);
  replace_r[nth] = move_prob > rand[nth];
}

kernel void logprior_z(global uint *cur_z, global float *logprob, local uint *novel_feat,
		       uint N, uint D, uint K, float alpha) {
  
  uint nth = get_global_id(0); // nth is the index of data
  uint kth = get_global_id(1); // kth is the index of features
  uint m = 0;
  float logprob_temp = 0;
  novel_feat[kth] = 0;
  
  /* calculate the log probability of the nth row of Z 
     i.e., the prior probability of having the features
     of the nth object.
   */
  for (int n = 0; n < nth; n++) {
    m += cur_z[n * K + kth];
  }
  if (m > 0) { // if other objects have had this feature
    if (cur_z[nth * K + kth] == 1) {
      logprob_temp += log(m / (nth + 1.0f));
    }
    else {
      logprob_temp += log(1 - m / (nth + 1.0f));
    }
  } else { // if this is a novel feature
    novel_feat[kth] = cur_z[nth * K + kth] == 1;
  }

  // wait until tallying is done
  barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

  if (kth == 0) {
    uint novel_count = 0;
    for (int i = 0; i < K; i++) {
      novel_count += novel_feat[i];
    }
    logprob_temp += (novel_count > 0) * pois_logpmf(novel_count, alpha / (nth+1.0f));
  }
  logprob[nth * K + kth] = logprob_temp;
}

kernel void loglik(global int *z_by_ry,
		   global int *obs,
		   global float *loglik,
		   uint N, uint D, uint K,
		   float lambda, float epislon) {

  uint nth = get_global_id(0); // nth is the index of data
  uint dth = get_global_id(1); // dth is the index of flattened pixels

  /* calculate the log-likelihood of the dth pixel of the nth object
     given the corresponding weight in z_by_ry
  */
  uint weight = z_by_ry[nth * D + dth];
  if (obs[nth * D + dth] == 1) {
    loglik[nth * D + dth] = log(1 - pow(1 - lambda, weight) * (1 - epislon));
  } else {
    loglik[nth * D + dth] = weight * log(1 - lambda) + log(1 - epislon);
  }
}

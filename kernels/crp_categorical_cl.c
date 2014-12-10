//#pragma OPENCL EXTENSION cl_khr_fp64: enable
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
    logp[i] = native_powr(exp(1.0f), logp[i] - m);
  }
  float p_sum = sum(logp, start, length);
  for (int i = start; i < start + length; i++) {
    logp[i] = logp[i] / p_sum;
  }
}

int sample(uint a_size, global uint *a, global float *p, int start, float rand) {
  float total = 0;
  for (int i = start; i < start + a_size; i++) {
    total += p[i];
    if (total > rand) return a[i-start];
  }
  return a[a_size - 1];
}

__kernel void cat_logpost(global uint *labels, global uint *data, global uint *uniq_label, global float *count, global uint *n,  uint cluster_num, uint outcome_num, float alpha, float beta, global float *logpost, global float *rand) {
  
  uint data_size = get_global_size(0);
  uint i = get_global_id(0);
  uint c = get_global_id(1);
  uint old_label = labels[i];
  uint new_label = uniq_label[c];
  uint new_size = n[c];
  uint original_cluster = old_label == new_label;
  float loglik;

  loglik = (beta + count[outcome_num * c + data[i]] - original_cluster) / 
    (outcome_num * beta + new_size - original_cluster);
 
  loglik += (new_size > 0) ? log(new_size/(alpha + data_size)) : log(alpha/(alpha + data_size));
  logpost[i * cluster_num + c] = loglik;
}  

__kernel void resample_labels(global uint *labels, global uint *uniq_label, uint cluster_num, global float *rand, global float *logpost) {

  uint i = get_global_id(0);
  lognormalize(logpost, i * cluster_num, cluster_num);
  //printf("Data %d Before: %d\n", i, labels[i]);
  labels[i] = sample(cluster_num, uniq_label, logpost, i * cluster_num, rand[i]);
  //printf("Data %d After: %d\n", i, labels[i]);
}

__kernel void cat_logpost_loopy(global uint *labels, global int *data, global uint *uniq_label, global uint *count, global uint *n, uint cluster_num, uint outcome_num, float alpha, float beta, global float *logpost, global float *rand) {
  
  uint i = get_global_id(0);
  uint data_size = get_global_size(0);
  uint old_label = labels[i];
  uint new_label;
  uint new_size;
  uint original_cluster;
  uint empty_n = 1;

  for (int c = 0; c < cluster_num; c++) {
    new_label = uniq_label[c];
    new_size = n[c];
    empty_n += (old_label == new_label && new_size == 1); // discounting for the slim chance that 
    original_cluster = old_label == new_label;

    logpost[i * cluster_num + c] = (beta + count[outcome_num * c + data[i]] - original_cluster) / 
      (outcome_num * beta + new_size - original_cluster);
    //printf("i: %d c: %d logpost: %f\n", i, c, logpost[i * cluster_num + c]);
    logpost[i * cluster_num + c] += (new_size > original_cluster) ? 
      log((new_size - original_cluster) / (alpha + data_size-1)) : log(alpha / empty_n / (alpha + data_size-1));
  }
  lognormalize(logpost, i * cluster_num, cluster_num);
  //printf("Data %d Before: %d\n", i, labels[i]);
  labels[i] = sample(cluster_num, uniq_label, logpost, i * cluster_num, rand[i]);
  //printf("Data %d After: %d\n", i, labels[i]);
}

__kernel void compute_suff_stats(global uint *uniq_label, global int *labels, global uint *data, global uint *count, global uint *n, uint data_size, uint outcome_num) {
  
  uint c = get_global_id(0);
  n[c] = 0;
  for (int i = 0; i < outcome_num; i++) {
    count[outcome_num * c + i] = 0;
  }

  for (int i = 0; i < data_size; i++) {
    n[c] += (uniq_label[c] == labels[i]);
    count[outcome_num * c + (uniq_label[c] == labels[i]) * data[i]] += (uniq_label[c] == labels[i]);
  }
}

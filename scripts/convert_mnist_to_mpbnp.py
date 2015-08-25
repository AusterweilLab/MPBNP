#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import division, print_function
import idx2numpy, gzip, sys
import numpy as np

DIM_SIZE = 28

print("Reading training data from original MNIST file ...", file=sys.stderr)
training_idx_fp = gzip.open('../data/MNIST/train-images-idx3-ubyte.gz')
training_arr = idx2numpy.convert_from_file(training_idx_fp)
training_idx_fp.close()

# convert to binary
print("Converting to binary images ...", file=sys.stderr)
training_arr = training_arr.astype(bool).astype(int)
# flatten each image
training_arr = training_arr.reshape(training_arr.shape[0], DIM_SIZE ** 2)

print("Save results in csv format for IBP noisyor ...", file=sys.stderr)
# write out csv file for ibp
header_str = ','.join(['p' + str(_) for _ in range(DIM_SIZE)])
training_ibp_fp = gzip.open('../data/MNIST/train-images-binary-ibp.csv.gz', 'w')
np.savetxt(training_ibp_fp, training_arr, fmt='%d', delimiter=',', header=header_str, comments='')
training_ibp_fp.close()

print("Save results in csv format for tIBP noisyor ...", file=sys.stderr)
# write out csv file for ibp
training_tibp_fp = gzip.open('../data/MNIST/train-images-binary-tibp.csv.gz', 'w')
training_arr_tibp = np.insert(training_arr, 0, DIM_SIZE, axis=1)
np.savetxt(training_tibp_fp, training_arr_tibp, fmt='%d', delimiter=',', header='width,'+header_str, comments='')
training_tibp_fp.close()

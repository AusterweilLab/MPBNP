#!/usr/bin/env python2
#-*- coding: utf-8 -*-

from __future__ import print_function
import pyopencl as cl, numpy as np
import pyopencl.array
import sys, copy, random, math, csv, gzip, mimetypes, os.path
from time import time

def lognormalize(x):
    # adapt it to numpypy
    x = x - np.max(x)
    xp = np.exp(x)
    return xp / xp.sum()

def print_matrix_in_row(npmat, file_dest):
    """Print a matrix in a row.
    """
    row, col = npmat.shape
    print(col, *npmat.reshape((1, row * col))[0], sep=',', file=file_dest)
    return True

class BasePredictor(object):

    def __init__(self, cl_mode = True, cl_device = None):
        """Initialize the class.
        """
        if cl_mode:
            if cl_device == 'gpu':
                gpu_devices = []
                for platform in cl.get_platforms():
                    try: gpu_devices += platform.get_devices(device_type=cl.device_type.GPU)
                    except: pass
                self.ctx = cl.Context(gpu_devices)
            elif cl_device == 'cpu':
                cpu_devices = []
                for platform in cl.get_platforms():
                    try: cpu_devices += platform.get_devices(device_type=cl.device_type.CPU)
                    except: pass
                self.ctx = cl.Context([cpu_devices[0]])
            else:
                self.ctx = cl.create_some_context()

            self.queue = cl.CommandQueue(self.ctx)
            self.mf = cl.mem_flags
            self.device = self.ctx.get_info(cl.context_info.DEVICES)[0]
            self.device_type = self.device.type
            self.device_compute_units = self.device.max_compute_units

        self.cl_mode = cl_mode
        self.obs = []
        self.samples = {}
        
    def read_test_csv(self, filepath, header = True):
        """Read test data from a csv file.
        """
        # determine if the type file is gzip
        filetype, _ = mimetypes.guess_type(filepath)
        if filetype == 'gzip':
            csvfile = gzip.open(filepath, 'r')
        else:
            csvfile = open(filepath, 'r')

        csvfile.seek(0)
        reader = csv.reader(csvfile)
        if header: reader.next()
        for row in reader:
            self.obs.append(row)
        return

    def read_samples_csv(self, var_name, filepath, header = True):
        """Read test data from a csv file.
        """
        # determine if the type file is gzip
        filetype, encoding = mimetypes.guess_type(filepath)
        if encoding == 'gzip':
            csvfile = gzip.open(filepath, 'r')
        else:
            csvfile = open(filepath, 'r')

        csvfile.seek(0)
        reader = csv.reader(csvfile)
        # skip the header
        if header: reader.next()
        # add the samples to variables
        self.samples[var_name] = []
        for row in reader:
            self.samples[var_name].append(row)
        return

    def predict(self, thining = 0, burnin = 0, use_iter=None, output_file = None):
        """Predict the test cases
        """
        return

if __name__ == '__main__':
    
    p = BasePredictor(cl_mode=False)
    p.read_test_csv('../data/ibp-image-test.csv')
    print(p.obs)

#!/usr/bin/env python2
#-*- coding: utf-8 -*-

from __future__ import print_function
import pyopencl as cl, numpy as np
import pyopencl.array
import sys, copy, random, math, csv, gzip, mimetypes, os.path
from time import time

def smallest_unused_label(int_labels):
    
    if len(int_labels) == 0: return [], [], 0
    label_count = np.bincount(int_labels)
    try: 
        new_label = np.where(label_count == 0)[0][0]
    except IndexError: 
        new_label = max(int_labels) + 1
    uniq_labels = np.unique(int_labels)
    return label_count, uniq_labels, new_label

def lognormalize(x):
    # adapt it to numpypy
    x = x - np.max(x)
    xp = np.exp(x)
    return xp / xp.sum()

def sample(a, p):
    """Step sample from a discrete distribution using CDF
    """
    n = len(a)
    r = random.random()
    total = 0           # range: [0,1]
    for i in xrange(n):
        total += p[i]
        if total > r:
            return a[i]
    return a[i]

def print_matrix_in_row(npmat, file_dest):
    """Print a matrix in a row.
    """
    row, col = npmat.shape
    print(col, *npmat.reshape((1, row * col))[0], sep=',', file=file_dest)
    return True

class BaseSampler:

    def __init__(self, cl_mode = True, inference_mode = True, cl_device = None):
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
        self.inference_mode = inference_mode
        if inference_mode:
            self.obs = []
            self.input_csv_name = None
            self.niter = 1000
            self.thining = 0
            self.burnin = 0

    def read_csv(self, filepath, header = True):
        """Read data from a csv file.
        """
        if type(filepath) is str:
            self.input_csv_name, _ = os.path.splitext(filepath)
        
        # determine if the type file is gzip
        filetype, _ = mimetypes.guess_type(filepath)
        if filetype == 'gzip':
            csvfile = gzip.open(filepath, 'r')
        else:
            csvfile = open(filepath, 'r')

        #dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        reader = csv.reader(csvfile)#, dialect)
        if header: 
            reader.next()
        for row in reader:
            self.obs.append([_ for _ in row])
        return

    def direct_read_obs(self, obs):
        self.obs = obs

    def set_sampling_params(self, niter = 1000, thining = 0, burnin = 0):
        self.niter, self.thining, self.burnin = niter, thining, burnin

    def do_inference(self, output_file = None):
        if not self.inference_mode: 
            print("Cannot perform inference in generative mode")
            sys.exit(0)
        return

    def generate(self, n = 1000, output_file = None):
        """Generate n random samples from the assumed generative model.
        """
        if self.inference_mode: 
            print("Cannot generate random samples in inference mode")
            sys.exit(0)
        return

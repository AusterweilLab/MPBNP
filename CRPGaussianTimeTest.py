#!/usr/bin/env python2
#-*-coding: utf-8 -*-

from __future__ import print_function
import argparse, sys, os.path
pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(pkg_dir)

#from CRPGaussianSamplers import *
import numpy as np
import pyopencl as cl
from time import time
from datetime import datetime
from MPBNP import crp

def print_args_summary(args):
    summary = "Running the CRP Gibbs sampler with the following arguments:\n"
    summary += "%d-dimensional Multivariate Normal\n" % args.dim
    summary += "OpenCL mode: %s\n" % args.opencl
    summary += "Number of iterations: %d\n" % args.iter
    summary += "Number of clusters: %d\n" % args.cluster_num
    summary += "Write output to a log file: %s\n" % args.output_to_file
    #summary += "Cluster size: %d\n" % args.cluster_size
    print(summary, file=sys.stderr)

parser = argparse.ArgumentParser(description="""
A test unit for automatically assessing the computation time of the CRP Gaussian Gibbs Sampler with and without OpenCL support.
Contact Joseph Austerweil <Joseph.austerweil@gmail.com for questions and feedback.
""")
parser.add_argument('--iter', '-t', type=int, default=10000, help='The number of iterations the sampler should run')
parser.add_argument('--burnin', '-b', type=int, default=2000, help='The number of iterations discarded as burn-in.')
parser.add_argument('--opencl_device', choices=['ask', 'gpu', 'cpu'], default='ask', help='The device to use OpenCL acceleration on. Default behavior is asking the user (i.e., you).')
parser.add_argument('--opencl', action='store_true', help='Use OpenCL acceleration')
parser.add_argument('--cuda', action='store_true', help='Use CUDA acceleration')
#parser.add_argument('--cluster_size', type=int, required=True,  help='The size of each cluster (for generating data)')
parser.add_argument('--dim', type=int, default=2, help="dimensionality of the gaussian")
parser.add_argument('--cluster_num', type=int, required=True, help='The number of clusters (for generating data)')
parser.add_argument('--kernel', choices=['kd_gaussian'], default='kd_gaussian')
parser.add_argument('--output_to_file', action='store_true', help="Write to a log file in the current directory if turned on")
parser.add_argument('--repeat', type=int, default=1, help='The number of times this test should be run.')

args = parser.parse_args()
print_args_summary(args)

#c = CRPGaussianCollapsedGibbs(cl_mode = args.opencl)

c = crp.gaussian.CollapsedGibbs(cl_mode = args.opencl,
                                cl_device=args.opencl_device,
                                cuda_mode=args.cuda)

"""if args.opencl:
    device = c.ctx.get_info(cl.context_info.DEVICES)[0]
    device_type = device.type
    device_platform = device.platform.name.replace('\x00', '').strip()
    device_name = device.name.replace('\x00', '').strip()
    device_vendor = device.vendor.replace('\x00', '').strip()
    device_max_cu = device.max_compute_units
"""
if args.opencl:
    device_type = c.device_type
    device_platform = c.device_platform
    device_name = c.device_name
    device_vendor = c.device_vendor
    device_max_cu = c.device_max_cu

if args.output_to_file is False: 
    file_dest = sys.stdout
else: 
    if args.opencl:
        file_dest = open('%s-%s-%d-dim-t%d-c%d-r%d-cu%d.csv' % (device_name, device_platform, args.dim, args.iter, args.cluster_num, args.repeat, device_max_cu), 'w')
    else:
        file_dest = open('%d-dim-t%d-c%d-r%d-nocl.csv' % (args.dim, args.iter, args.cluster_num, args.repeat), 'w')

print('timestamp,no.clusters,cluster.size,dimension,n.iter,opencl,device.vendor,device.name,device.platform,device.type,device_max_cu,gpu_time,total_time', file=file_dest)

for r in xrange(args.repeat):
    timestamp = str(datetime.now()).split('.')[0] 
#    for j in (5, 10, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 20000, 50000, 100000, 1000000):
    for j in (5, 10, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 20000, 50000, 100000):
        data_size = j
        print('Run timestamp: %s Testing data size %d' % (timestamp, data_size), file=sys.stderr)
        
        for i in xrange(args.cluster_num):
            if i == 0:
                data = np.random.multivariate_normal([1] * args.dim, np.identity(args.dim), data_size / args.cluster_num)
            elif i < args.cluster_num - 1:
                data = np.vstack((data, np.random.multivariate_normal([i*10 + 1] * args.dim, np.identity(args.dim), data_size / args.cluster_num)))
            else:
                data = np.vstack((data, np.random.multivariate_normal([i*10 + 1] * args.dim, np.identity(args.dim), max(data_size / args.cluster_num, data_size - data.shape[0]))))

        c.direct_read_obs(data)
        c.set_sampling_params(niter = args.iter, burnin=args.burnin)
        gpu_time, total_time, _ = c.do_inference()
        
        if args.opencl:
            print('%s,%d,%d,%d,%d,%s,"%s","%s","%s",%d,%d,%f,%f' % 
                  (timestamp, args.cluster_num, data_size, args.dim, args.iter, args.opencl, device_vendor, device_name, device_platform, device_type, device_max_cu,gpu_time,total_time),
                  file = file_dest)
        else:
            print('%s,%d,%d,%d,%d,%s,,,,,,%f,%f' % 
                  (timestamp, args.cluster_num, data_size, args.dim, args.iter, args.opencl, gpu_time, total_time),
                  file = file_dest)
    
        if file_dest is not sys.stdout: file_dest.flush()

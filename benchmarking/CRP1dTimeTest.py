#!/usr/bin/env python2
#-*-coding: utf-8 -*-

from __future__ import print_function
import argparse, sys
import numpy as np
import pyopencl as cl
from CRPMM import CRPMM
from time import time
from datetime import datetime

def print_args_summary(args):
    summary = "Running the sampler with the following arguments:\n"
    summary += "OpenCL mode: %s\n" % args.opencl
    summary += "Number of iterations: %d\n" % args.iter
    summary += "Number of clusters: %d\n" % args.cluster_num
    #summary += "Cluster size: %d\n" % args.cluster_size
    print(summary, file=sys.stderr)

parser = argparse.ArgumentParser(description="""
A test unit for automatically assesing the computation time of the CRPMM program with and without OpenCL support.
Contact Ting Qian <ting_qian@brown.edu> for questions and feedback.
""")
parser.add_argument('--iter', '-t', type=int, default=100, help='The number of iterations the sampler should run')
parser.add_argument('--opencl', action='store_true', help='Use OpenCL acceleration')
#parser.add_argument('--cluster_size', type=int, required=True,  help='The size of each cluster (for generating data)')
parser.add_argument('--cluster_num', type=int, required=True, help='The number of clusters (for generating data)')
parser.add_argument('--kernel', choices=['1d_gaussian'], default='1d_gaussian')
parser.add_argument('--output', type=argparse.FileType('w'))
parser.add_argument('--repeat', type=int, default=1, help='The number of times this test should be run.')

args = parser.parse_args()
print_args_summary(args)

if args.opencl:
    c = CRPMM(cl_mode = True)
else:
    c = CRPMM(cl_mode = False)

if args.output is None: file_dest = sys.stdout
else: file_dest = args.output

print('timestamp,no.clusters,cluster.size,n.iter,opencl,device.vendor,device.name,device.type,time.elapsed', file=file_dest)

for r in xrange(args.repeat):
    timestamp = str(datetime.now()).split('.')[0] 
    #for j in xrange(1, 10, 1):
    for j in (5, 10, 50, 100, 250, 500, 1000, 2000, 3000, 4000, 5000, 7500, 10000, 20000):
        per_cluster_size = j
        print('Run timestamp: %s Testing cluster size %d' % (timestamp, per_cluster_size), file=sys.stderr)
        
        for i in xrange(args.cluster_num):
            if i == 0:
                data = np.random.normal(1, 1, per_cluster_size)
            else:
                data = np.hstack((data, np.random.normal(i*10 + 1, 1, per_cluster_size)))

        sample_size = len(data)
        init_labels = np.random.randint(low = 0, high = min(sample_size, 10), size = sample_size)

        if args.opencl:
            a_time = time()
            clusters = c.cl_infer_1dgaussian(obs = data, niter=args.iter, init_labels = init_labels)
            device = c.ctx.get_info(cl.context_info.DEVICES)[0]
            device_type = device.get_info(cl.device_info.TYPE)
            device_name = device.get_info(cl.device_info.NAME).strip()
            device_vendor = device.get_info(cl.device_info.VENDOR).strip()
            #print(clusters)
        else:
            a_time = time()
            clusters = c.infer_1dgaussian(obs = data, niter=args.iter, init_labels = init_labels)
            #print(clusters)
        
        b_time = time()
        
        if args.opencl:
            print('%s,%d,%d,%d,%s,"%s","%s",%d,%f' % 
                  (timestamp, args.cluster_num, per_cluster_size,args.iter, args.opencl, device_vendor, device_name, device_type,b_time - a_time),
                  file = file_dest)
        else:
            print('%s,%d,%d,%d,%s,,,,%f' % 
                  (timestamp, args.cluster_num, per_cluster_size,args.iter, args.opencl, b_time - a_time),
                  file = file_dest)
    
    if file_dest is not sys.stdout: file_dest.flush()

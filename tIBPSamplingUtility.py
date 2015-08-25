#!/usr/bin/env python2
#-*-coding: utf-8 -*-

from __future__ import print_function, division

import sys, os.path
pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(pkg_dir)

import argparse, sys, csv, gzip, os.path
import numpy as np
import pyopencl as cl
from MPBNP import tibp
from time import time

def print_args_summary(args):
    summary = "Running the sampler with the following arguments:\n"
    summary += "Input data file: %s\n" % args.data_file
    summary += "OpenCL mode: %s\n" % args.opencl
    if args.opencl: summary += "Which OpenCL device to use: %s\n" % args.opencl_device
    summary += "Distribution of each component: %s\n" % args.kernel
    summary += "Number of iterations: %d\n" % args.iter
    summary += "Number of burn-in iterations: %d\n" % args.burnin
    summary += "Write output to a log file: %s\n" % args.output_to_file
    summary += "Number of chains: %s\n" % args.chain
    if args.chain > 1 and args.opencl:
        summary += "Distribute chains across multiple OpenCL devices: %s\n" % args.distributed_chains
    print(summary, file=sys.stderr)

parser = argparse.ArgumentParser(description="""
A sampler for the Transformed Indian Buffet Process model with and without OpenCL support.
Please contact Ting Qian <ting_qian@brown.edu> for questions and feedback.
""")
parser.add_argument('--opencl', action='store_true', help='Use OpenCL acceleration')
parser.add_argument('--opencl_device', choices=['ask', 'gpu', 'cpu'], default='ask', help='The device to use OpenCL acceleration on. Default behavior is asking the user.')
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--kernel', choices=['noisyor'], 
                    default='noisyor', help='The likelihood function of each feature. Default is noisyor for binary images.')
parser.add_argument('--iter', '-t', type=int, default=10000, help='The number of iterations the sampler should run')
parser.add_argument('--burnin', '-b', type=int, default=2000, help='The number of iterations discarded as burn-in.')
parser.add_argument('--output_to_file', action='store_true', help="Write posterior samples to a log file in the current directory. Default behavior is not keeping records of posterior samples")
parser.add_argument('--output_to_stdout', action='store_true', help="Write posterior samples to standard output (i.e., your screen). Default behavior is not keeping records of posterior samples")
parser.add_argument('--output_mode', choices=['best', 'all'], default='best', help='Output mode. Default is keeping only the sample that yields the highest logliklihood of data. The other option is to keep all samples.')
parser.add_argument('--chain', '-c', type=int, default=1, help='The number of chains to run. Default is 1.')
parser.add_argument('--distributed_chains', action='store_true', default=False, help="If there are multiple OpenCL devices, distribute chains across them. Default is no. Will not distribute to CPUs if GPU is specified in opencl_device, and vice versa")
parser.add_argument('--epsilon',default=.02, type=float, help="Sets the epsilon parameter (prob. pixels on by chance) via command line")
parser.add_argument('--lam',default=.98, type=float, help="Sets the lambda parameter (prob. a feature turns on a pixel it is supposed to turn on")
parser.add_argument('--theta',default=.2, type=float, help="Sets the theta parameter (prior prob. a pixel in a feature image is on). Small values promote feature images with few pixels turned on (large the opposite)")

# parse and print out the arguments
args = parser.parse_args()

# check for imcompatibilities
if args.output_mode == 'all' and args.output_to_stdout:
    print('Recording all samples is chosen, but printing to screen is also selected. This is not recommended.', file=sys.stderr)
    sys.exit(0)

print_args_summary(args)

# parse the name of the input file and set up output file path
if type(args.data_file) is str:
    input_filename, _ = os.path.splitext(os.path.basename(args.data_file))
output_path = os.path.dirname(os.path.realpath(args.data_file)) + '/'

# set up the sampler
if args.kernel == 'noisyor':
    c = tibp.noisyor.Gibbs(cl_mode = args.opencl, cl_device = args.opencl_device,
                          record_best = args.output_mode == 'best', epsilon=args.epsilon,
                           lam=args.lam, theta=args.theta)
else:
    sys.exit()
c.read_csv(args.data_file)
c.set_sampling_params(niter = args.iter, burnin = args.burnin)

# run the sample through multiple chains
for chain in xrange(args.chain):
    # set up the output file
    if args.output_to_file: 
        if args.opencl:
            sample_dest = output_path + input_filename + '-%d-%s-%s-chain-%d-cl/' % (args.iter - args.burnin, args.kernel, args.output_mode, chain + 1)
        else:
            sample_dest = output_path + input_filename + '-%d-%s-%s-chain-%d-nocl/' % (args.iter - args.burnin, args.kernel, args.output_mode, chain + 1)
    elif args.output_to_stdout:
        sample_dest = sys.stdout
    else:
        sample_dest = None

    print("Chain %d running, please wait ..." % (chain + 1), file=sys.stderr)
    gpu_time, total_time, common_clusters = c.do_inference(output_file = sample_dest)
    print("Chain %d finished. OpenCL device time: %f; Total_time: %f seconds\n" % (chain + 1, gpu_time, total_time), file=sys.stderr)

#!/usr/bin/env python
#-*- coding: utf-8 -*-

from __future__ import print_function, division

import sys, os.path
pkg_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'
sys.path.append(pkg_dir)

import argparse, sys, csv, gzip, os.path, png

def print_args_summary(args):
    summary = "Converting the following gzipped CSV file outputted from sampler to a viewable image:\n"
    summary += "Input data file: %s\n" % args.data_file
    summary += "Output file: %s\n" % args.output_file
    print(summary, file=sys.stderr)

parser = argparse.ArgumentParser(description="""
Converts an image in CSV format from the sampler to a viewable image
""")
parser.add_argument('--data_file', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)

args = parser.parse_args()

print_args_summary(args)
imageCSVStrFP = None
splitPath = args.data_file.split(".")
if splitPath[-1] == "gz":
    imageCSVStrFP = gzip.open(args.data_file,"rb")
else:
    imageCSVStrFP = open(args.data_file)

imageCSVStr = imageCSVStrFP.readlines()
imageCSVStr = map(lambda x: x.replace(",",""), imageCSVStr)
imageCSVStr = map(lambda x: x.replace("\n",""), imageCSVStr)
imageList = map(lambda x: map(int, x), imageCSVStr)

outFileStr = args.output_file
if outFileStr[-4:] != ".png":
    outFileStr = outFileStr + ".png"
fOut = open(outFileStr, "wb")
wOut = png.Writer(len(imageList[0]), len(imageList), greyscale=True, bitdepth=1)
wOut.write(fOut,imageList)
fOut.close()
#!/usr/bin/env python
#-*- coding: utf-8 -*-

from os.path import isfile, join
from os import listdir
import png, argparse, numpy as np

parser = argparse.ArgumentParser(description="""
A simple script to convert a directory of pngs to the format expected by the MPBNP package (CSV with image per row and leading column giving the width of th eimage).
Written by Joe Austerweil (joseph_austerweil@brown.edu)
""")
parser.add_argument('--in_dir', type=str, required=True, help="The directory with the pngs to convert to MPBNP CSV format (this is required).")
parser.add_argument('--out_file', type=str, help="Name for output (default will be the directory name.csv)")
args = parser.parse_args()
mypath = args.in_dir

out_file = getattr(args, 'out_file', None)
if out_file is None:
    out_file = mypath+".csv"
if not out_file.endswith(".csv"):
    out_file+= '.csv'

allPNGs = [f for f in listdir(mypath) if (isfile(join(mypath,f)) and f.endswith(".png"))]

out_data = None
r0 = png.Reader(join(mypath,allPNGs[0]))
f_width = r0.read()[0]

for f in allPNGs:
    r = png.Reader(join(mypath,f))
    assert(r.read()[0] == f_width)
    image_1d = np.hstack((np.array(f_width), np.array(list(r.asDirect()[2])).ravel()))
    #image_1d = np.hstack(itertools.imap(np.uint16, r.asDirect()))
    if out_data is None:
        out_data = image_1d
    else:
        out_data = np.vstack((out_data, image_1d))
#join with ../?
headstr = "w"
for i in np.arange(out_data.shape[1]-1):
    headstr += ",c" + np.array_str(i+1)
np.savetxt(join(mypath,out_file),out_data,delimiter=",", fmt="%2g",header=headstr,comments="")
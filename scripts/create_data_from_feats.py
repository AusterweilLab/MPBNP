#!/usr/bin/env python
#-*- coding: utf-8 -*-

from os.path import isfile, join, isdir
from os import listdir, makedirs
import png,argparse, csv, numpy as np
from skimage.io import imread, imsave, imshow

parser = argparse.ArgumentParser(description="""
Loads a feature set of pngs and file containing a feature ownership matrix for creating the data set.
Assume alphabetic ordering of file names for column order of feature ownership matrix.
Written by Joe Austerweil (joseph_austerweil@brown.edu)
""")
parser.add_argument('--in_dir', type=str, required=True, help="The directory with the pngs -- 1 for each feature img.")
parser.add_argument('--feat_own_file', type=str, required=True, help="file containing a feature ownership matrix for creating the data set")
parser.add_argument('--out_dir', type=str, help="Name of directory for output (default will be the in_dir_out)")
parser.add_argument('--feat_own_header', default=False, help="Set to true if the feature ownership matrix file has a header")
parser.add_argument('--feat_own_delim', type=str, default=',', help="Delimiter used in feature ownership file")
parser.add_argument('--out_dataname', type=str, help="filename for the MPBNP datafile created for the given data set (defaults to data.csv in the out_dir)")
parser.add_argument('--compress', action='store_true', help="save MPBNP data in compressed format")
parser.add_argument('--thresh', default=50, type=int, help="threshold for binarizing input pngs (aliased binary pngs will create fuzzy borders)")
args = parser.parse_args()
mypath = args.in_dir

out_dir = getattr(args, 'out_dir', None)
if out_dir is None:
    out_dir = mypath+"_out"
# if not out_file.endswith(".csv"):
#     out_file+= '.csv'


allPNGs = [f for f in listdir(mypath) if (isfile(join(mypath,f)) and f.endswith(".png"))]

feat_imgs = None
#r0 = png.Reader(join(mypath,allPNGs[0]))

blah = imread(join(mypath,allPNGs[0]))
f_width,f_height, _ = blah.shape

for f in allPNGs:
    #r = png.Reader(join(mypath,f))
    #assert(r.read()[0] == f_width)
    #image_1d = np.hstack((np.array(f_width), np.array(list(r.asDirect()[2])).ravel()))
    #blah = np.array(list(r.asDirect()[2]))
    #image_1d = np.array(list(r.asDirect()[2])).ravel()
    #image_1d = np.array(list(itertools.imap(np.uint16, r.asDirect())))
    #image_1d = np.array(list(r.asRGB8()[2]), dtype=np.uint8)

    image_1d = 255-imread(join(mypath,f))
    if feat_imgs is None:
        feat_imgs = image_1d[:,:,0].ravel()
    else:
        feat_imgs = np.vstack((feat_imgs, image_1d[:,:,0].ravel()))

feat_imgs = (feat_imgs > args.thresh).astype(np.uint8)
z_fid = open(args.feat_own_file)
z_reader = csv.reader(z_fid,delimiter=args.feat_own_delim)
z = None
if not args.feat_own_header:
    z = np.array([line for line in z_reader]).astype(np.uint8)
else:
    i = 0
    for line in z_reader:
        if i == 1:
            z = np.array(line).astype(np.uint8)
        elif i > 1:
            z = np.vstack((z, np.array(line).astype(np.uint8)))

z_fid.close()

#recreate objs from feat imgs and z
N = z.shape[0]
K = z.shape[1]
assert(K == feat_imgs.shape[0])
D = feat_imgs.shape[1]
x = np.zeros(shape=(N,D),dtype=np.uint8)

for n in np.arange(N):
    for k in np.arange(K):
        x[n,:] += z[n,k] * feat_imgs[k,:]
    x[n,:] = (x[n,:] == 0)*255

#x=np.require(x,dtype=np.uint8)

try:
    makedirs(out_dir)
except OSError:
    if not isdir(out_dir):
        raise

#f_height = D // f_width
for n in np.arange(N):
#    fOut = open(join(out_dir,"img"+str(n)) + ".png","wb")
#    wOut = png.Writer(width=f_width, height=f_height, greyscale=True, bitdepth=1)
#    wOut.write(fOut,x[n,:].reshape((f_width,f_height)))
#    fOut.close()
    imsave(join(out_dir,"img"+str(n)) + ".png",x[n,:].reshape((f_width,f_height)))

x = np.hstack((f_width*np.ones(shape=(N,1)), x//255))

#create data set file in MPBNP format
headstr = "w"
for i in np.arange(D):
    headstr += ",c" + np.array_str(i+1)

out_file = getattr(args, 'out_filename', None)
if out_file is None:
    out_file = join(out_dir,"data.csv")
if args.compress is True and not out_file.endswith(".gz"):
    out_file = out_file+".gz"
np.savetxt(out_file, x ,delimiter=",", fmt="%2g",header=headstr,comments="")
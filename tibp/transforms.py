#!/usr/bin/env python2
#-*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import sys
from scipy import ndimage

import warnings
#warnings.filterwarnings('error')

def v_translate(f_img, f_img_width, distance):
    """Translate a feature image vertically by distance (+ 
    indicates down, - indicates up).
    """
    if distance == 0: return f_img
    f_img_height = int(f_img.shape[0] / f_img_width)
    f_img_mat = f_img.reshape((f_img_height, f_img_width))
    if distance > 0:
        shift = int(distance % f_img_height)
        t = np.vstack((f_img_mat[f_img_height-shift:],
                       f_img_mat[:f_img_height-shift])).reshape(f_img.shape)
    if distance < 0:
        shift = int(abs(distance) % f_img_height)
        t = np.vstack((f_img_mat[shift:], 
                       f_img_mat[:shift])).reshape(f_img.shape)
    return t

def v_trans(f_img, f_img_width, distance):
    """Non-numpy version. Slower but as a template for OpenCL C code.
    """
    if distance == 0: return f_img
    f_img_height = int(f_img.shape[0] / f_img_width)
    t = np.empty(shape=f_img.shape, dtype=f_img.dtype)
    for h in xrange(f_img_height):
        for w in xrange(f_img_width):
            t[((h + distance) % f_img_height) * f_img_width + w] = f_img[h * f_img_width + w]
    return t

def h_translate(f_img, f_img_width, distance):
    """Translate a feature image horizontally by distance (+ 
    indicates right, - indicates left).
    """
    if distance == 0: return f_img
    f_img_height = int(f_img.shape[0] / f_img_width)
    f_img_mat = f_img.reshape((f_img_height, f_img_width))
    if distance > 0:
        shift = int(distance % f_img_width)
        t = np.hstack((f_img_mat[:,f_img_width-shift:],
                       f_img_mat[:,:f_img_width-shift])).reshape(f_img.shape)
    if distance < 0:
        shift = int(abs(distance) % f_img_height)
        t = np.hstack((f_img_mat[:,shift:], 
                       f_img_mat[:,:shift])).reshape(f_img.shape)
    return t

def h_trans(f_img, f_img_width, distance):
    """Non-numpy version. Slower but as a template for OpenCL C code.
    """
    if distance == 0: return f_img
    f_img_height = int(f_img.shape[0] / f_img_width)
    t = np.empty(shape=f_img.shape, dtype=f_img.dtype)
    for h in xrange(f_img_height):
        for w in xrange(f_img_width):
            t[h * f_img_width + (w + distance) % f_img_width] = f_img[h * f_img_width + w]
    return t

def scale(f_img, f_img_width, pixel):
    """Scale a feature image by pixel from right and bottom
    while holding the top-left corner constant
    """
    if pixel == 0: return f_img
    f_img_height = int(f_img.shape[0] / f_img_width)
    if pixel > min(f_img_height, f_img_width) - 2:
        warnings.warn("Scale magnitude is greater than allowed maximum. No scaling performed.")
        return f_img
    
    percent = min(f_img_height - pixel, f_img_width - pixel) / min(f_img_width, f_img_height)
    f_img_mat = f_img.reshape((f_img_height, f_img_width))

    f_img_mat_new = ndimage.interpolation.zoom(f_img_mat, zoom = percent)

    if f_img_mat_new.shape[0] < f_img_mat.shape[0] or f_img_mat_new.shape[1] < f_img_mat.shape[1]:
        f_img_mat_new = np.pad(f_img_mat_new, 
                               ((0, f_img_mat.shape[0] - f_img_mat_new.shape[0]), 
                                (0, f_img_mat.shape[1] - f_img_mat_new.shape[1])), 
                               mode="constant")
    elif f_img_mat_new.shape[0] > f_img_mat.shape[0] or f_img_mat_new.shape[1] > f_img_mat.shape[1]:
        f_img_mat_new = f_img_mat_new[:f_img_mat.shape[0], :f_img_mat.shape[1]]
        
    return f_img_mat_new.reshape(f_img.shape)
    
if __name__ == "__main__":
    
    a = np.random.randint(0,2,10)
    from time import time
    a_time = time()
    print(scale(a, 5, 1.2))
    print(time() - a_time)

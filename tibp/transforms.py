#!/usr/bin/env python2
#-*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np

def v_translate(f_img, f_img_width, distance):
    """Translate a feature image vertically by distance (+ 
    indicates down, - indicates up).
    """
    if distance == 0: return f_img
    f_img_height = int(f_img.shape[0] / f_img_width)
    f_img_mat = f_img.reshape((f_img_height, f_img_width))
    if distance > 0:
        shift = distance % f_img_height
        t = np.vstack((f_img_mat[f_img_height-shift:],
                       f_img_mat[:f_img_height-shift])).reshape(f_img.shape)
    if distance < 0:
        shift = abs(distance) % f_img_height
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
        shift = distance % f_img_width
        t = np.hstack((f_img_mat[:,f_img_width-shift:],
                       f_img_mat[:,:f_img_width-shift])).reshape(f_img.shape)
    if distance < 0:
        shift = abs(distance) % f_img_height
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

def scale(f_img, f_img_width, percent):
    """Scale a feature image by holding the top left corner
    of the image constant.
    """
    pass
    
if __name__ == "__main__":
    
    a = np.random.randint(0,20,60000)
    print(a)
    from time import time
    a_time = time()
    h_translate(a, 30, 15)
    print(time() - a_time)
    a_time = time()
    h_trans(a, 30, 15)
    print(time() - a_time)
    print(np.array_equal(h_translate(a, 30, 15), h_trans(a, 30, 15)))

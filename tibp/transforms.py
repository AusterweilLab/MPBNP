#!/usr/bin/env python2
#-*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np

def v_translation(f_img, f_img_width, distance):
    """Translate a feature image vertically by distance (+ 
    indicates down, - indicates up).
    """
    if distance == 0: return f_img
    f_img_height = f_img.shape[0] / f_img_width
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

def h_translation(f_img, f_img_width, distance):
    """Translate a feature image horizontally by distance (+ 
    indicates right, - indicates left).
    """
    if distance == 0: return f_img
    f_img_height = f_img.shape[0] / f_img_width
    f_img_mat = f_img.reshape((f_img_height, f_img_width))
    print(f_img_mat)
    if distance > 0:
        shift = distance % f_img_width
        t = np.hstack((f_img_mat[:,f_img_width-shift:],
                       f_img_mat[:,:f_img_width-shift])).reshape(f_img.shape)
    if distance < 0:
        shift = abs(distance) % f_img_height
        t = np.hstack((f_img_mat[:,shift:], 
                       f_img_mat[:,:shift])).reshape(f_img.shape)
    return t

def scale(f_img, f_img_width, percent):
    """Scale a feature image by holding the top left corner
    of the image constant.
    """
    pass
    
if __name__ == "__main__":
    
    a = np.random.randint(0,2,6)
    print(a)
    print(h_translation(a, 3, 10))
    

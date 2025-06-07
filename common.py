#!/usr/bin/env python3
#############################################################################
# Copyright (C) 2025  Xu Ruijun
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#############################################################################
import functools
import numpy as np
import shapely
import cv2


def proj(H, x):
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((x, ones), axis=1)
    x = np.matmul(x, H.transpose())  #(A . x^T)^T = x . A^T
    x /= x[:,2].reshape((-1,1))
    return x[:,:2]

def shapely_perspective(x, H):
    return shapely.transform(x, functools.partial(proj, H))

def findPerspective(x1, y1, x2, y2, coord_lt, coord_rt, coord_rb, coord_lb):
    try:
        H, _ = cv2.findHomography(
            np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)]),
            np.array([coord_lt, coord_rt, coord_rb, coord_lb]))
    except:
        return None
    else:
        return H

def try_func(f, *args, **kwargs):
    try:
        retval = f(*args, **kwargs)
    except Exception as e:
        print(e)
        return None
    else:
        return retval
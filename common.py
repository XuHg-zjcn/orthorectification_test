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
import numpy as np


def shaply_proj(H, x):
    ones = np.ones((x.shape[0], 1))
    x = np.concatenate((x, ones), axis=1)
    x = np.matmul(x, H.transpose())  #(A . x^T)^T = x . A^T
    x /= x[:,2].reshape((-1,1))
    return x[:,:2]

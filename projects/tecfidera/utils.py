# encoding: utf-8

import numpy as np


def readcfl(cfl):
    h = open(cfl + ".hdr", "r")
    h.readline()  # skip
    line = h.readline()
    h.close()
    dims = [int(i) for i in line.split()]

    # remove singleton dimensions from the end
    n = int(np.prod(dims))
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n) + 1]

    # load data and reshape into dims
    d = open(cfl + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    a = a.reshape(dims, order='F')  # column-major

    return a

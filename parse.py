from graph_tool import all as gt
import matplotlib.pyplot as plt
from typing import *
import numpy as np
import scipy as sp
import gzip
import time

def csr_matrix(path: str) -> sp.sparse.csr_matrix:
    start = time.time() 
    cache = {}
    n = 0
    m = 0

    with gzip.open(path) as f:
        while True:
            line = f.readline()
            if line.startswith(b'#'):
                continue
            if not line:
                break

            i, j = map(int, line.strip().split(b'\t'))
            cache[i] = cache.get(i, 0) + 1
            n = max(i + 1, j + 1, n)
            m += 1

    data = np.zeros(m, dtype=np.float32)
    indices = np.zeros(m, dtype=np.int32)
    indptr = np.zeros(n + 1, dtype=np.int32)

    indptr[0] = cache[0]
    for i in range(1, n): indptr[i] = cache.get(i, 0) + indptr[i - 1]
    indptr[n] = indptr[n - 1]

    with gzip.open(path) as f:
        for _ in range(4): f.readline()
        while True:
            line = f.readline()
            if line.startswith(b'#'):
                continue
            if not line:
                break

            i, j = map(int, line.strip().split(b'\t'))
            indptr[i] = indptr[i] - 1
            indices[indptr[i]] = j
            data[indptr[i]] = 1
            
    return sp.sparse.csr_matrix((data, indices, indptr))

adjmat = csr_matrix("graph.txt.gz")
G = gt.Graph(adjmat, directed=False)

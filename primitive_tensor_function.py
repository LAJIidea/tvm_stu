from __future__ import absolute_import, print_function
import tvm
from tvm import te
import numpy as np

n = te.var('n')
m = te.var('m')
A = te.placeholder((m, n), name='A')
B = te.placeholder((m, n), name='B')
C = te.compute((m, n), lambda i, j: A[i, j] * B[i, j], name='C')

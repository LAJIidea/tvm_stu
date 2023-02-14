import tvm
from tvm import te
from numpy.testing import assert_allclose

import numpy as np

n = 1024
A = te.placeholder((n,), name='A')
B = te.placeholder((n,), name='B')
C = te.compute(A.shape, lambda i: A[i] + B[i], name="C")

# create a default schedule
s = te.create_schedule(C.op)

# create a module
target = "llvm"
fadd = tvm.build(s, [A, B, C], target, name="myadd")

# run the module
ctx = tvm.device(target, 0)
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), ctx)
b = tvm.nd.array(np.random.uniform(size=n).astype(B.dtype), ctx)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), ctx)
fadd(a, b, c)

print(a)
print(b)
print(c)

# tvm.testing.assert_allclose(c.asnumpy(), a.asnumpy() + b.asnumpy())
assert_allclose(c.numpy(), a.numpy() + b.numpy())

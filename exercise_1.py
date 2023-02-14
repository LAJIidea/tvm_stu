import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

# init data
a = np.arange(16).reshape(4, 4).astype(np.int64)
b = np.arange(16, 0, -1).reshape(4, 4).astype(np.int64)
# 下面的代码会因为没有知道type而报错
# a = np.arange(16).reshape(4, 4)
# b = np.arange(16, 0, -1).reshape(4, 4)

print(a)

print(b)

c_np = a + b
print(c_np)


def lnumpy_add(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    for i in range(4):
        for j in range(4):
            c[i, j] = a[i, j] + b[i, j]


c_lnumpy = np.empty((4, 4), dtype=np.int64)
lnumpy_add(a, b, c_lnumpy)
print(c_lnumpy)


# TensorIR version
@tvm.script.ir_module
class MyAdd:
    @T.prim_func
    def add(A: T.Buffer[(4, 4), "int64"],
            B: T.Buffer[(4, 4), "int64"],
            C: T.Buffer[(4, 4), "int64"]):
        T.func_attr({"global_symbol": "add"})
        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)
                C[vi, vj] = A[vi, vj] + B[vi, vj]


rt_lib = tvm.build(MyAdd, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_ = tvm.nd.array(np.empty((4, 4), dtype=np.int64))
# print(c_)
# c_tvm = tvm.nd.empty((4, 4), dtype="int64")
# print(c_tvm)
rt_lib["add"](a_tvm, b_tvm, c_)
np.testing.assert_allclose(c_.numpy(), c_np, rtol=1e-5)

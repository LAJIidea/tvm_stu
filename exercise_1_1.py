import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

a = np.arange(16).reshape(4, 4).astype(np.int64)
b = np.arange(4, 0, -1).reshape(4).astype(np.int64)
print(a)
print(b)
c_np = a + b
print(c_np)

@tvm.script.ir_module
class MyAdd:
    @T.prim_func
    def add(A: T.Buffer[(4, 4), "int64"],
            B: T.Buffer[(4), "int64"],
            C: T.Buffer[(4, 4), "int64"]):
        T.func_attr({"global_symbol": "add", "tir.noalias": True})
        for i, j in T.grid(4, 4):
            with T.block("C"):
                vi = T.axis.spatial(4, i)
                vj = T.axis.spatial(4, j)
                C[vi, vj] = A[vi, vj] + B[vj]


rt_lib = tvm.build(MyAdd, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.empty((4, 4), dtype=np.int64))
rt_lib["add"](a_tvm, b_tvm, c_tvm)
print(c_tvm)
np.testing.assert_allclose(c_tvm.numpy(), c_np, rtol=1e-5)

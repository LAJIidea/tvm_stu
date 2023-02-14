import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import torch

N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N * CI * H * W).reshape(N, CI, H, W).astype(np.int64)
weight = np.arange(CO * CI * K * K).reshape(CO, CI, K, K).astype(np.int64)

data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
print(data_torch)
print(weight_torch)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
# print(conv_torch)


def conv(in_put, out_put, core):
    for batch in range(N):
        for out_item in range(CO):
            for input_item in range(CI):
                for i in range(OUT_H):
                    for j in range(OUT_W):
                        filter_sum = 0
                        # convolution operation: [i, i+1, i+2] * [j, j+1, j+2]
                        for m in range(K):
                            for n in range(K):
                                out_put[batch, out_item, i, j] += in_put[batch, input_item, i + m, j + n] * \
                                                                  core[out_item, input_item, m, n]


# conv_np = np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64)
# conv(data, conv_np, weight)

# print(conv_np)
# print(conv_torch)

# np.testing.assert_allclose(conv_torch, conv_np, rtol=1e-5)


@tvm.script.ir_module
class MyConv:
    @T.prim_func
    def conv(d: T.Buffer[(1, 1, 8, 8), "int64"], w: T.Buffer[(2, 1, 3, 3), "int64"],
             out: T.Buffer[(1, 2, 6, 6), "int64"]):
        T.func_attr({"global_symbol": "conv", "tir.noalias": True})
        for batch, co, ci, i, j, m, n in T.grid(1, 2, 1, 6, 6, 3, 3):
            with T.block("out"):
                v_batch = T.axis.spatial(1, batch)
                v_co = T.axis.spatial(2, co)
                v_ci = T.axis.spatial(1, ci)
                vi = T.axis.spatial(6, i)
                vj = T.axis.spatial(6, j)
                vm = T.axis.spatial(3, m)
                vn = T.axis.spatial(3, n)
                out[v_batch, co, vi, vj] = out[v_batch, v_co, vi, vj] + \
                                           d[batch, v_ci, vi + vm, vj + vn] * w[v_co, v_ci, vm, vn]


rt_lib = tvm.build(MyConv, target="llvm")
data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
conv_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
# print(conv_tvm.numpy())
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)

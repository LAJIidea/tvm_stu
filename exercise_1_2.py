import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T
import torch

N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W).astype(np.int64)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K).astype(np.int64)

data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
print(data_torch)
print(weight_torch)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
print(conv_torch)


def conv(in_put, out_put, core, basis):
    for batch in range(batch_size):
        for out_item in range(output_channel):
            for input_item in range(input_channel):
                for i in range(output_height):
                    for j in range(output_width):
                        filter_sum = 0
                        # convolution operation: [i, i+1, i+2] * [j, j+1, j+2]
                        for m in range(filter_height):
                            for n in range(filter_width):
                                out_put[batch, i, j, out_item] += in_put[batch, i + m][j + n][input_item] * \
                                                                  core[m, n, input_item, out_item]
                        if input_item == input_channel - 1:
                            np_final_output[batch, i, j, out_item] += basis[batch][0][0][out_item]

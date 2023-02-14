# import numpy as np
#
# batch_size = 1
# input_height = 5
# input_width = 5
# filter_height = 3
# filter_width = 3
# output_height =input_height - filter_height + 1
# output_width = input_width - filter_width + 1
# input_channel = 3
# output_channel = 2
#
# np_input_arg = np.ones([batch_size, input_height, input_width, input_channel])
# np_filter_arg = np.ones([filter_height, filter_width, input_channel, output_channel])
# np_biases = np.ones([batch_size, 1, 1, output_channel])
#
# np_final_output = np.zeros([batch_size, output_height, output_width, output_channel])
#
#
# def conv(in_put, out_put, weight, basis):
#     for batch in range(batch_size):
#         for out_item in range(output_channel):
#             for input_item in range(input_channel):
#                 for i in range(output_height):
#                     for j in range(output_width):
#                         filter_sum = 0
#                         # convolution operation: [i, i+1, i+2] * [j, j+1, j+2]
#                         for m in range(filter_height):
#                             for n in range(filter_width):
#                                 out_put[batch, i, j, out_item] += in_put[batch, i + m][j + n][input_item] * \
#                                                                   weight[m, n, input_item, out_item]
#                         if input_item == input_channel - 1:
#                             np_final_output[batch, i, j, out_item] += basis[batch][0][0][out_item]
#

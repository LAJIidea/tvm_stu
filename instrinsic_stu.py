from tvm import te
import tvm


def intrin_gemv(m, l):
    a = te.placeholder((1, ), name='a')
    b = te.placeholder()

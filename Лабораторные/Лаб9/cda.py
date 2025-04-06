from __future__ import division
import numpy as np
from cuda_driver import *

N=2048

cuInit(0)

cnt = c_int(0)
cuDeviceGetCount(byref(cnt))
    
if cnt.value == 0:
    raise Exception('No GPU device found!')

cuDevice = c_int(0)
cuDeviceGet(byref(cuDevice), 0)
    
cuContext = c_void_p()
cuCtxCreate(byref(cuContext), 0, cuDevice)  


a_a = np.linspace(0, 2*(N-1), N, dtype=np.float32)
a = a_a.ctypes.data_as(POINTER(c_float))
b_a = np.linspace(1, 2*N-1, N, dtype=np.float32)
b = b_a.ctypes.data_as(POINTER(c_float))

a_d = c_void_p(0)
cuMemAlloc(byref(a_d), c_size_t(N*sizeof(c_float)))
b_d = c_void_p(0)
cuMemAlloc(byref(b_d), c_size_t(N*sizeof(c_float)))

cuMemcpyHtoD(a_d, a, c_size_t(N*sizeof(c_float)))
cuMemcpyHtoD(b_d, b, c_size_t(N*sizeof(c_float)))

cuModule = c_void_p()
cuModuleLoad(byref(cuModule), c_char_p(b'./cda.ptx'))

gStub_kern = c_void_p(0)
cuModuleGetFunction(byref(gStub_kern), cuModule, c_char_p(b'gStub'))


gStub_args=[c_void_p(addressof(a_d)), c_void_p(addressof(b_d))]
gStub_params = (c_void_p * len(gStub_args))(*gStub_args)
cuLaunchKernel(gStub_kern, int(N/128), 1, 1, 128, 1, 1, 0, 0, gStub_params, 0)
cuCtxSynchronize()

cuMemcpyDtoH( a, a_d,  c_size_t(N*sizeof(c_float)))

for i in range(0, N, N/16):
    print(a_a[i])

cuMemFree(a_d)
cuMemFree(b_d)
cuCtxDestroy(cuContext)

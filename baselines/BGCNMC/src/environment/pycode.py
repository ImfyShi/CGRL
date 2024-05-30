# from ctypes import *
import ctypes
import numpy as np

# load the shared object file
cdll = ctypes.CDLL('./minknap.so')
cdll.minknap.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cdll.minknap.restype = ctypes.c_double

array_len = 22
capacity = 1000

sol = (ctypes.c_int * array_len)(*np.zeros(array_len, dtype=np.int32))

pyarray_p = np.asarray([0.0,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0625,0.0,0.0625,0.,0.0,0.0625], dtype=np.double)
if not pyarray_p.flags['C_CONTIGUOUS']:
    pyarray_p = np.ascontiguous(pyarray_p, dtype=pyarray_p.dtype)
carray_p = pyarray_p.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

pyarray_w = np.asarray([71, 70, 70, 70, 58, 57, 57, 55, 55, 53, 53, 53, 51, 51, 51, 49, 48, 27, 20, 18, 18, 16], dtype=np.int32)
if not pyarray_w.flags['C_CONTIGUOUS']:
    pyarray_w = np.ascontiguous(pyarray_w, dtype=pyarray_w.dtype)
carray_w = pyarray_w.ctypes.data_as(ctypes.POINTER(ctypes.c_int))

obj = cdll.minknap(array_len, carray_p, carray_w, sol, capacity)
# obj = cdll.main()
print('obj={}'.format(obj))
'''
print('n4')
cdll.n4.restype = ctypes.c_double
n4obj = cdll.n4()
print('n4obj={}'.format(n4obj))
'''
'''
print('main')
cdll.main.restype = ctypes.c_double
mainobj = cdll.main()
print('mainobj={}'.format(mainobj))
'''
'''
sol_np = np.frombuffer(sol, dtype=np.int32)
print('sol={}'.format(sol_np))
'''
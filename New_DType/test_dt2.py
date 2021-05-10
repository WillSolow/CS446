import numpy as np
import DMC_rs_dtype_lib as l

walker = l.make_dtype(3)

arr = l.read_xyz('tmp.xyz',dt=walker)
print(arr)
print(l.print_xyz(arr))

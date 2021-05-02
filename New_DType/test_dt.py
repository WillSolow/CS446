import numpy as np
import DMC_rs_dtype_lib as l

walker = l.make_dtype(3)

walkers = np.array([(1,'water',[(8,[0,0,0]),(1,[1,1,1]),(1,[-1,-1,-1])])],dtype=walker)

tmp = open('tmp.xyz','w')
xyz = l.print_xyz(walkers)
tmp.write(xyz)
print(xyz)

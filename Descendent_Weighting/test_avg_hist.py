import numpy as np
import matplotlib.pyplot as plt
import DMC_rs_lib as lib

wlk_list = [np.random.normal(60,10,(1000,3,3,3)) for i in range(100)]
plt.bar(align='edge', width=1.5, **lib.avg_hist(wlk_list))
plt.show()

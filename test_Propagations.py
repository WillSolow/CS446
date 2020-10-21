import numpy as np

walkers = .5 + (np.random.rand(5, 1, 3, 3)-.5)
print('walkers: ', walkers)

offset = np.array([.0001, 10, 100000])
propogations = np.random.normal(0, np.sqrt(np.transpose(np.tile(offset, (5, 1, 3, 1)), (0,1,3,2))))
print('\n\n\npropagations', propogations)

output = walkers + propogations
print('\n\nOutput', output)
from scipy.ndimage import rotate
import numpy as np
from time import time
import pyclesperanto_prototype as cle


data = np.random.rand(int(300/1),int(512/1),int(512/1))

t0 = time()
for i in range(5):
    foo=cle.rotate(data, angle_around_x_in_degrees=10, rotate_around_center=True,auto_size=False, linear_interpolation=False)
print(time()-t0)
print(np.array(foo))

t0 = time()
for i in range(5):
    rotate(data, angle=10, axes=(1,2), order=0, reshape=False, prefilter=False, mode='constant')
print(time()-t0)

t0 = time()
for i in range(5):
    rotate(data, angle=10, axes=(1,2), order=0, reshape=False, prefilter=False, mode='nearest')
print(time()-t0)

t0 = time()
for i in range(5):
    rotate(data, angle=10, axes=(1,2), order=0, reshape=False, prefilter=False, mode='reflect')
print(time()-t0)

t0 = time()
for i in range(5):
    rotate(data, angle=10, axes=(1,2), order=0, reshape=False, prefilter=False, mode='grid-mirror')
print(time()-t0)

t0 = time()
for i in range(5):
    rotate(data, angle=10, axes=(1,2), order=0, reshape=False, prefilter=False, mode='grid-constant')
print(time()-t0)

t0 = time()
for i in range(5):
    rotate(data, angle=10, axes=(1,2), order=0, reshape=False, prefilter=False, mode='mirror')
print(time()-t0)

t0 = time()
for i in range(5):
    rotate(data, angle=10, axes=(1,2), order=0, reshape=False, prefilter=False, mode='grid-wrap')
print(time()-t0)

t0 = time()
for i in range(5):
    rotate(data, angle=10, axes=(1,2), order=0, reshape=False, prefilter=False, mode='wrap')
print(time()-t0)

t0 = time()
for i in range(5):
    rotate(data, angle=10, axes=(1,2), order=1, reshape=False, prefilter=False)
print(time()-t0)



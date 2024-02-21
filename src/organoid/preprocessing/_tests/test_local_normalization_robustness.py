from organoid.preprocessing.preprocessing import local_image_normalization
from organoid.preprocessing.preprocessing import make_array_isotropic
from organoid.preprocessing.preprocessing import compute_mask
from organoid.preprocessing.preprocessing import align_array_major_axis
from organoid.preprocessing.preprocessing import crop_array_using_mask
import tifffile
import numpy as np
import napari

"""
Conclusion: Box size must be chosen around 1 typical cell diameter
"""

data = tifffile.imread(
    '/home/jvanaret/data/project_egg/raw/fusion4/fusion4.tif'
)[[1,-1]]
print(data.shape)


data_iso = make_array_isotropic(data, zoom_factors=(1.5/0.91,1,1))
print(data_iso.shape)



mask = compute_mask(data_iso, box_size=10)#, sigma_blur=16)
print(mask.shape)

viewer = napari.Viewer()
viewer.add_image(data)
viewer.add_image(data_iso)
viewer.add_image(mask)
for box_size in [5, 10, 20, 40, 80]:

    data_iso_norm = local_image_normalization(data_iso, box_size, 1, 99)
    data_iso_norm = np.where(mask, data_iso_norm, 0.0)


    viewer.add_image(data_iso_norm, name=f'box_size={box_size}')

napari.run()
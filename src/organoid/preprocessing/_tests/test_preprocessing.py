from organoid.preprocessing.preprocessing import local_image_normalization
from organoid.preprocessing.preprocessing import make_array_isotropic
from organoid.preprocessing.preprocessing import compute_mask
from organoid.preprocessing.preprocessing import align_array_major_axis
from organoid.preprocessing.preprocessing import crop_array_using_mask
import tifffile
import numpy as np
import napari


data = tifffile.imread(
    '/home/jvanaret/data/project_egg/raw/fusion4/fusion4.tif'
)[:5]
print(data.shape)


data_iso = make_array_isotropic(data, zoom_factors=(1.5/0.91,1,1))
print(data_iso.shape)



mask = compute_mask(data_iso, box_size=10)#, sigma_blur=16)
print(mask.shape)

data_iso_norm = local_image_normalization(data_iso, 12, 1, 99)
print(data_iso_norm.shape)

data_iso_norm = np.where(mask, data_iso_norm, 0.0)




mask_aligned,data_iso_norm_aligned = align_array_major_axis(
    target_axis='X',
    rotation_plane='XY',
    mask=mask,
    image=data_iso_norm,
    order=1
)
print(mask_aligned.shape)
print(data_iso_norm_aligned.shape)


data_iso_norm_aligned_cropped, mask_aligned_cropped = crop_array_using_mask(
    data_iso_norm_aligned, mask_aligned
)
print(data_iso_norm_aligned_cropped.shape)
print(mask_aligned_cropped.shape)


viewer = napari.Viewer()

viewer.add_image(data)
viewer.add_image(data_iso)
viewer.add_image(data_iso_norm)
viewer.add_image(mask)
viewer.add_image(mask_aligned)
viewer.add_image(data_iso_norm_aligned)
viewer.add_image(data_iso_norm_aligned_cropped)
viewer.add_image(mask_aligned_cropped)

napari.run()


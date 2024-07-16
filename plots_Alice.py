import tifffile
from skimage.measure import regionprops
import napari
import numpy as np

path = rf'C:\Users\eqplenne\Desktop\Data_Alice\delme'
probamap = tifffile.imread(rf'{path}\registered_movie_ch1_t100_probamap.tif')
all_cells = tifffile.imread(rf'{path}\registered_movie_ch1_t100_labels.tif')

def smooth_gaussian(array, sigma, scale: tuple, mask=None, mask_for_volume=None, return_effective_volume: bool=False):
    """
    Performs convolution of 'array' with a gaussian kernel of
    width(s) 'sigma'. 
    If 'mask' is specified, the convolution will not take the
    masked value into account.
    """

    scale=np.array(scale)
    sigmas = sigma/scale
    
    if mask is None:
        #return skimage_gaussian(array, sigmas, preserve_range=True, mode='constant', cval=0.0)
        return scipy_gaussian(array, sigmas, mode='constant', cval=0.0)
    else:

        if mask_for_volume is None:
            mask_for_volume = mask.copy()

        mask = mask.astype(bool)
        mask_for_volume = mask_for_volume.astype(bool)
        array_copy = array.copy() * 1.0

        array_copy[~mask_for_volume] = 0.0

        smooth_array = scipy_gaussian(
            array_copy, sigmas,
            mode='constant', cval=0.0
        )

        smooth_array_copy = smooth_array.copy()
        
        # calculate renormalization factor for masked gaussian (the 'effective'
        # volume of the gaussian kernel taking the mask into account)
        effective_volume = scipy_gaussian(
            mask_for_volume*1.0, sigmas,
            mode='constant', cval=0.0
        )

        smooth_array[mask] = smooth_array[mask] / effective_volume[mask]
        smooth_array[~mask] = 0.0

        if return_effective_volume:
            return smooth_array, effective_volume, smooth_array_copy
        else:
            return smooth_array


labels = all_cells[:,:36,:,:]

labels_ppties = regionprops(labels)
centroids_labels = np.array([prop.centroid for prop in labels_ppties]).T
coords_in_array = [coord.astype(int).tolist() for coord in centroids_labels]
im_centroids = np.zeros_like(labels)
im_centroids[coords_in_array[0],coords_in_array[1],coords_in_array[2]] = 1
density_labels = smooth_gaussian


viewer=napari.Viewer()
viewer.add_image(probamap)
viewer.add_labels(labels)
napari.run()
print(probamap.shape,labels.shape)
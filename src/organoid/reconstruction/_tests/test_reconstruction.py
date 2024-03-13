from organoid.reconstruction import (
    extract_positions,
    plot_positions,
    manual_registration_fct,
    register,
    create_folders,
    check_napari,
    fuse_sides,
    sigmoid,
    write_hyperstacks,
    add_centermass,
    remove_previous_files,
)
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import regionprops
def add_random_landmark(array,limits,radius,label):
    x1,y1,z1 = np.random.randint(limits[0],limits[1]),np.random.randint(limits[0],limits[1]),np.random.randint(limits[0],limits[1])
    array[
        x1-radius:x1+radius,
        y1-radius:y1+radius,
        z1-radius:z1+radius,
    ] = label
    return array


def test_manual_registration():
    array_ref = np.zeros((100, 100, 100), dtype=int)
    radius=3
    limits=[20,80] #landmarks are placed far from the border to avoid going out of the image after rotation
    array_ref=add_random_landmark(array_ref,limits,radius,1)
    array_ref=add_random_landmark(array_ref,limits,radius,2)
    array_ref=add_random_landmark(array_ref,limits,radius,3)

    #create a floating image from the ref, with rotations and translations
    rotation_applied=[40,0,10]
    translation_applied=[15,8,10]
    array_float = np.copy(array_ref)
    array_float = ndi.rotate(array_float, rotation_applied[0], axes=(1,2), order = 0,reshape=False)
    array_float = ndi.rotate(array_float, -rotation_applied[1], axes=(2,0), order = 0,reshape=False) 
    array_float = ndi.rotate(array_float, rotation_applied[2], axes=(0,1), order = 0,reshape=False)
    array_float = ndi.shift(array_float, rotation_applied, order=0, mode='constant', cval=0,prefilter=False)

    #recover the rotation and translation applied
    rot, trans1, trans2 = manual_registration_fct(array_ref, array_float)

    #apply the rotations and translations found, in the same way as the registration code : translation 1 to bring the center of mass to the center of the image
    #then rotation, then translation 2 to bring the center of mass to the reference center of mass
    array_float_aftertrans1 = ndi.shift(array_float, trans1, order=0, mode='constant', cval=0,prefilter=False)
    array_float_rotated = ndi.rotate(array_float_aftertrans1, -rot[0], axes=(1,2), order = 0,reshape=False)
    array_float_rotated = ndi.rotate(array_float_rotated, -rot[1], axes=(2,0), order = 0,reshape=False) 
    array_float_rotated = ndi.rotate(array_float_rotated, -rot[2], axes=(0,1), order = 0,reshape=False)
    array_float_registered = ndi.shift(array_float_rotated, trans2, order=0, mode='constant', cval=0,prefilter=False)

    #check if the float image is registered at the same position as the reference image
    rg_ref = regionprops(array_ref)
    centroids_ref = np.array([prop.centroid for prop in rg_ref]).T
    rg_float = regionprops(array_float_registered)
    centroids_float = np.array([prop.centroid for prop in rg_float]).T
    error = centroids_ref-centroids_float
    max_error = np.full((3, 3), 5) # 5 pixels of error allowed
    assert((error<max_error).all())
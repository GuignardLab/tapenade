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


def test_manual_registration():
    array_ref = np.zeros((100, 100, 100), dtype=int)
    array_float = np.zeros((100, 100, 100), dtype=int)
    array_ref[40,55,30] =1
    array_ref[63,37,49] =2
    array_ref[51,25,28] =3
    array_float[40,74,45] =1
    array_float[63,49,53] =2
    array_float[51,49,29] =3

    rot_applied= [30,0,0]
    trans1_applied= [-1.3,-7.3,7.6]
    trans2_applied= [1.3,-11,-14.3]
    
    rot, trans1, trans2 = manual_registration_fct(array_ref, array_float)
    error_rot = abs(np.subtract(rot_applied,rot))
    error_trans1 = abs(np.subtract(trans1_applied,trans1))
    error_trans2 = abs(np.subtract(trans2_applied,trans2))
    assert(all(error_rot<[3,3,3]))
    assert(all(error_trans1<[3,3,3]))
    assert(all(error_trans2<[3,3,3]))
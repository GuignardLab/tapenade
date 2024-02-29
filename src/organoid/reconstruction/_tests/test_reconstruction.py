from organoid.reconstruction import (
    extract_positions,
    plot_positions,
    associate_floatbottom,
    manual_registration_fct,
    register,
    create_folders,
    check_napari,
    fuse_sides,
    sigmoid,
    write_hyperstacks,
    add_centermass,
)
import numpy as np
from scipy import ndimage as ndi

# def test_position_array():
#     test_obj = reconstruct_foo()
#     assert test_obj == -1

# def test_manual_registration():
#     array_ref = np.zeros((50,50,50))
#     array_ref[20,12,31] = 1
#     array_ref[24,42,18] = 2
#     array_ref[8,37,29] = 3

#     array_float =np.copy(array_ref)
#     array_float = ndi.rotate(array_float, 10, axes=(0,1), order = 0)
#     # array_float = ndi.rotate(array_float, 7, axes=(1,2), order = 0, reshape=False)
#     array_float[:,10:,10:]=array_float[:,:-10,:-10]

#     rot,trans1,trans2 = manual_registration_fct(array_ref, array_float)

#     assert rot == 10

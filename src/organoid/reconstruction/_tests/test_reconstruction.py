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
    array_ref = np.zeros((50, 50, 50), dtype=int)
    array_ref[
        np.random.randint(0, 20),
        np.random.randint(0, 20),
        np.random.randint(0, 20),
    ] = 1
    array_ref[
        np.random.randint(0, 20),
        np.random.randint(0, 20),
        np.random.randint(0, 20),
    ] = 2
    array_ref[
        np.random.randint(0, 20),
        np.random.randint(0, 20),
        np.random.randint(0, 20),
    ] = 3

    array_float = np.copy(array_ref)
    array_float = ndi.rotate(array_float, 10, axes=(0, 1), order=0)
    array_float[:, 10:, 10:] = array_float[:, :-10, :-10]

    rot, trans1, trans2 = manual_registration_fct(array_ref, array_float)

    assert len(rot) == 3
    assert len(trans1) == 3
    assert len(trans2) == 3
    # assert rot != [0, 0, 0]
    # assert trans1 != [0, 0, 0]
    # assert trans2 != [0, 0, 0]

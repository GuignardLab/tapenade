import scipy.ndimage as ndi

def change_voxelsize(array, input_vs:tuple = (1,1,1), output_vs:tuple = (1,1,1), order:int=1) :
    """
    Rescale an array to a new voxel size.
    :param array: The array to rescale.
    :param input_voxelsize: The voxel size of the input array.
    :param output_voxelsize: The voxel size of the output array.
    :param order: The order of the spline interpolation, default is 1.
    :return: The rescaled array.
    """
    assert len(input_vs) == len(output_vs)
    zoom_factor = tuple([input_vs[i]/output_vs[i] for i in range(len(input_vs))])
    return ndi.zoom(array,zoom = zoom_factor,order=order)

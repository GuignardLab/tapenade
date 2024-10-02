from nd2reader import ND2Reader
import os
import numpy as np

def _nd2_reader(path:str) -> list:
    #from https://www.napari-hub.org/plugins/napari-nikon-nd2
    '''Read a Nikon ND2 file
    
    Parameters
    ----------
    path : str
        Path to the image to open
        
    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of layer.
    '''

    ndx = ND2Reader(path)
    name = os.path.basename(path)[:-4]
    print(name)
    sizes = ndx.sizes
    
    if 't' not in sizes:
        sizes['t'] = 1
    if 'z' not in sizes:
        sizes['z'] = 1
    if 'c' not in sizes:
        sizes['c'] = 1

    ndx.bundle_axes = 'zcyx'
    ndx.iter_axes = 't'
    n = len(ndx)

    shape = (sizes['t'], sizes['z'], sizes['c'], sizes['y'], sizes['x'])
    image  = np.zeros(shape, dtype=np.float32)

    for i in range(n):
        image[i] = ndx.get_frame(i)

    image = np.squeeze(image)
    
    if sizes['c'] > 1:
        channel_axis = len(image.shape) - 3
    else:
        channel_axis = None 
    params = {
        "channel_axis":channel_axis,
        "name":name,
    }
    layer_type = "image"  # optional, default is "image"

    return [(image, params, layer_type)]

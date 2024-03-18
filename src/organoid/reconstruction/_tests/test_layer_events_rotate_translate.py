import numpy as np
import napari


data = np.random.rand(10,40,40)

viewer = napari.Viewer()
layer = viewer.add_image(data)

def on_rotate(event):
    print('hey')
    print('rotate', event.source.rotate)

layer.events.rotate.connect(on_rotate)

layer.rotate = (10,0,0)

napari.run()


# This code plots a correlation maps between two intensity images in 3D.
# It is recommended to compute this maps after removing any optical artifacts and normalizing the intensities (see the preprocessing notebook for more details).
# You can plot and/or save the correlation maps for all the samples in a folder.

import tifffile
from skimage.measure import block_reduce
from pathlib import Path
from glob import glob
import numpy as np
from tapenade import analysis
import os

#Reading all the data in one folder
folder_data = ...
samples = []

for name in os.listdir(folder_data):
    if name.endswith('.tif'):
        samples.append(name.split('.')[0])

print('list of the samples :',samples)

scale = (1,1,1)
sigma=10

#here we take the first sample but you can add a for loop to iterate over all the samples
index_sample=samples[0]

labels = tifffile.imread(Path(folder_data) / f"{index_sample}_label.tif")
mask = tifffile.imread(Path(folder_data) / f"{index_sample}_mask.tif")
image = tifffile.imread(Path(folder_data) / f"{index_sample}_image.tif")
gene_1 = image[:,1,:,:] 
gene_2 = image[:,2,:,:]
labels = labels*mask.astype(bool)

gene_1[mask==0]='nan' #masking the values to not take them into account when applying mean pooling
gene_2[mask==0]='nan'
gene_1_coarse_grained = block_reduce(gene_1,block_size=sigma,func=np.mean)
gene_2_coarse_grained = block_reduce(gene_2,block_size=sigma,func=np.mean)

### uncomment to visualize the data in 3D
# viewer=napari.Viewer()
# viewer.add_image(gene_1_coarse_grained,name='gene_1_coarse_grained',colormap='inferno')
# viewer.add_image(gene_2_coarse_grained,name='gene_2_coarse_grained',colormap='inferno')
# napari.run()

x_name='gene_1'
y_name='gene_2'

analysis.plot_heatmap(
            X = gene_1_coarse_grained,
            Y = gene_2_coarse_grained ,
            bins=[100,100],
            X_extent=[0,10],
            Y_extent=[0,40],
            X_label=f'{x_name} expression',
            Y_label=f'{y_name} expression',
            plot_title=f'Correlation plot of {x_name} vs {y_name}, #{index_sample}',
            path_to_save = Path(folder_data) / f"{x_name}_{y_name}_#{index_sample}.png",
            plot_map=True
)
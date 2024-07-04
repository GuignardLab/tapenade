# This code plots a correlation maps between two intensity images in 3D.
# It is recommended to compute this maps after removing any optical artifacts and normalizing the intensities (see the preprocessing notebook for more details).
# You can plot and/or save the correlation maps for all the samples in a folder.

import tifffile
import scipy.ndimage as ndi
from skimage.measure import block_reduce
from pathlib import Path
from glob import glob
import numpy as np
# from organoid import analysis
import napari
from skimage.measure import regionprops
import os
#Reading all the data in one folder
folder_data = rf'C:\Users\gros\Desktop\DATA\Valentin\registered'
samples = []
paths = Path(folder_data) / 'data'

for name in os.listdir(paths):
    if name.endswith('.tif'):
        samples.append(name.split('.')[0])

print('list of the samples :',samples)

scale = (1,1,1)
sigma=20

#here we take the first sample but you can add a for loop to iterate over all the samples
index_sample=4

labels = tifffile.imread(Path(folder_data) / 'seg' / f"{index_sample}_seg.tif")
mask = tifffile.imread(Path(folder_data) /'masks' / f"{index_sample}_mask.tif")
image = tifffile.imread(Path(folder_data) / 'data' / f"{index_sample}.tif")
gene_1 = image[:,2,:,:] 
gene_2 = image[:,3,:,:]
labels = labels*mask.astype(bool)

gene_1[mask==0]='nan' #masking the values to not take them into account when applying mean pooling
gene_2[mask==0]='nan'
gene_1_coarse_grained = block_reduce(gene_1,block_size=sigma,func=np.mean)
gene_2_coarse_grained = block_reduce(gene_2,block_size=sigma,func=np.mean)
### uncomment to visualize the data in 3D

# viewer=napari.Viewer()
# viewer.add_image(gene_1_coarse_grained,name='gene_1_coarse_grained',colormap='inferno')
# viewer.add_image(gene_2_coarse_grained,name='gene_2_coarse_grained',colormap='inferno')
# x_name='gene_1'
# y_name='gene_2'

# labels_ero=ndi.binary_erosion(labels,iterations=1)
# labels_relabeled=labels_ero.astype(bool)*labels
# new_im=np.zeros_like(labels_relabeled)
# bra_ppties = regionprops(labels_relabeled, intensity_image=gene_2)
# for prop in bra_ppties :
#     new_im[prop.coords[:,0],prop.coords[:,1],prop.coords[:,2]]=prop.intensity_mean
# cellular_bra = [prop.intensity_mean for prop in bra_ppties]
# viewer.add_image(new_im,name='bra',colormap='inferno')
# viewer.add_image(gene_2)
# napari.run()
# # 
# analysis.plot_heatmap(
#             X = gene_1_coarse_grained,
#             Y = gene_2_coarse_grained ,
#             bins=[100,100],
#             X_extent=[0,10],
#             Y_extent=[0,40],
#             X_label=f'{x_name} expression',
#             Y_label=f'{y_name} expression',
#             plot_title=f'Correlation plot of {x_name} vs {y_name}, #{index_sample}',
#             path_to_save = Path(folder_data) / f"{x_name}_{y_name}_#{index_sample}.png",
#             plot_map=True
# )

from organoid.preprocessing.preprocessing import make_array_isotropic
from organoid.preprocessing.preprocessing import compute_mask
from scipy.ndimage import zoom, rotate, uniform_filter
from scipy.signal import argrelextrema
import tifffile
import numpy as np
import napari
import matplotlib.pyplot as plt
from tqdm import tqdm

"""
Conclusion: Box size must be chosen between and 1 and 3 typical cell diameters (this seems quite robust) 
"""

data = tifffile.imread(
    '/home/jvanaret/data/project_lecuit/image.tif'
)
# data = tifffile.imread(
#     '/home/jvanaret/data/project_egg/raw/fusion4/fusion4.tif'
# )#[0][:,69:324, 26:325][:, 109:221, 12:140]
print(data.shape)

pmin, pmax = np.percentile(data, (1, 99))

data_iso = np.clip((data.copy() - pmin) / (pmax - pmin), 0,1) 
# data_iso = make_array_isotropic(data, zoom_factors=(1.5/0.91,1,1))
# print(data_iso.shape)

def variance_threshold_binarization_debug(image, box_size):

    variance1 = uniform_filter(image, box_size)**2
    variance2 = uniform_filter(image**2, box_size)
    sigma = np.sqrt(np.clip(variance2 - variance1, 0, None))

    freqs,bins = np.histogram(sigma.ravel(), bins=256)
    threshold_candidates_args = argrelextrema(freqs, np.less_equal, order=1)[0][1:-1]
    threshold = bins[threshold_candidates_args[np.argmin(freqs[threshold_candidates_args])]]

    plt.hist(sigma.ravel(), bins=256)
    plt.axvline(threshold, color='r')
    plt.yscale('log')
    # plt.show()
    return sigma > threshold

mask = variance_threshold_binarization_debug(data_iso, box_size=7)#, sigma_blur=16)

def variance_threshold_binarization_debug_thresholds(image):

    thresholds=[]

    for box_size in tqdm(range(2,10)):
    # for box_size in tqdm(range(2,50)):

        variance1 = uniform_filter(image, box_size)**2
        variance2 = uniform_filter(image**2, box_size)
        sigma = np.sqrt(np.clip(variance2 - variance1, 0, None))

        freqs,bins = np.histogram(sigma.ravel(), bins=256)
        threshold_candidates_args = argrelextrema(freqs, np.less_equal, order=1)[0][1:-1]

        try:
            threshold = bins[threshold_candidates_args[np.argmin(freqs[threshold_candidates_args])]]
        except ValueError:
            plt.figure()
            plt.hist(sigma.ravel(), bins=256)
            plt.yscale('log')
            plt.show()

        thresholds.append(threshold)

    plt.figure()
    plt.plot(thresholds)

variance_threshold_binarization_debug_thresholds(data_iso)

print(mask.shape)
plt.show()

viewer = napari.Viewer()

viewer.add_image(data)
viewer.add_image(data_iso)
viewer.add_image(mask)

napari.run()


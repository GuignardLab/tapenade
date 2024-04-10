import matplotlib.pyplot as plt
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path
import matplotlib
import napari
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from organoid.utils import filter_percentiles
from organoid.utils import linear_fit as utils_linear_fit

def plot_heatmap(X, Y, bins: tuple, mask = None,
                 X_label: str='', Y_label: str='', fig=None, ax=None, 
                 linear_fit: bool = False,
                 average_XY: bool = False,
                 X_extent: tuple = None,
                 Y_extent: tuple = None,
                 preprocess_percentiles: bool = False,
                 plot_log: bool = False,
                 plot_title:str='',
                 lasso_select: bool = True,
                 path_to_save:str = '',
                 plot_map:str=True
):

    X_copy = X.copy()
    Y_copy = Y.copy()
    if not (mask is None):
        X = X[mask]
        Y = Y[mask] 
    
    X = X.ravel()
    Y = Y.ravel()
    matplotlib.rc('ytick', labelsize=10) 
    matplotlib.rc('xtick', labelsize=10) 
    
    mask_nan_X = ~np.isnan(X)
    mask_nan_Y = ~np.isnan(Y)
    mask_nan_both = np.logical_and(mask_nan_X, mask_nan_Y)
    X = X[mask_nan_both]
    Y = Y[mask_nan_both]

    if preprocess_percentiles:
        X, Y = filter_percentiles(
            X=np.array(X),percentilesX=(5,95),
            Y=np.array(Y),percentilesY=(5,95)
        )


    heatmap, xedges, yedges = np.histogram2d(
    X,
    Y,
    bins=bins,
    range= [X_extent, Y_extent]
)
    

    if plot_log:
        logbinsx = np.logspace(np.log10(xedges[0]),np.log10(xedges[-1]),len(xedges))
        logbinsy = np.logspace(np.log10(yedges[0]),np.log10(yedges[-1]),len(yedges))

        heatmap, xedges, yedges = np.histogram2d(
            X,
            Y,
            bins=(logbinsx, logbinsy)
        )

    if X_extent is None:
        X_extent = [xedges[0], xedges[-1]]
    if Y_extent is None:
        Y_extent = [yedges[0], yedges[-1]]

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111,facecolor='white')

        if not (plot_title is None) :
            fig.suptitle(plot_title, fontsize='xx-large')

    extent = X_extent+Y_extent
    # changing colormap to get white for 0 values 
    inferno_cmap = plt.cm.get_cmap('plasma')
    inferno_colors = inferno_cmap(np.linspace(0, 1, 256))
    inferno_colors[0] = (1, 1, 1, 1)  # (R, G, B, Alpha)
    custom_cmap = LinearSegmentedColormap.from_list('custom_inferno', inferno_colors)
    # Create heatmap
    ims = ax.imshow(
        heatmap.T, origin='lower', cmap=custom_cmap, extent=extent,
        aspect='auto', interpolation='none'
    )
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

    # ax.ticklabel_format(style='plain', axis='x')
    if X_label !=""  :
        ax.set_xlabel(X_label, fontsize=15)
    if Y_label !=""  :
        ax.set_ylabel(Y_label, fontsize=15)

    if plot_log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    cbar = fig.colorbar(ims, ax=ax)
    cbar.set_label('Number of cells', fontsize=10)

    # ### LEAST SQUARE LINEAR FIT
    if linear_fit:
        intercept, slope, r2 = utils_linear_fit(
            x=X,
            y=Y,
            return_r2=True
        )

        print(f"R-squared: {r2:.6f}")

        x = np.array(X[::10])

        lin_fit = intercept + slope*np.linspace(x.min(), x.max(), len(x))
        
        ax.plot(
            np.linspace(x.min(), x.max(), len(x)), lin_fit, 'red',
            
        )
    if average_XY:
            
        # PLOT AVERAGE VALUE ALONG THE Y-AXIS
        X_bins = (xedges[1::] + xedges[:-1:])/2
        Y_bins = (yedges[1::] + yedges[:-1:])/2
        # compute average of the signal weighted by the frequency of
        # each value along the y-axis
        average_Y = np.sum((heatmap * Y_bins).T, axis=0) \
                        / np.sum(heatmap.T, axis=0)

        average_X = np.sum((heatmap.T * X_bins), axis=1) \
                        / np.sum(heatmap.T, axis=1)


        ax.plot(X_bins, average_X)
        ax.plot(average_Y, Y_bins)
    if lasso_select:
        def onSelect(verts):

            path = Path(verts)
            points = np.array([X_copy.ravel(), Y_copy.ravel()]).T
            mask_both = path.contains_points(points).reshape(X_copy.shape)
            
            viewer = napari.Viewer(ndisplay=2)
            
            viewer.add_image(X_copy,colormap='inferno',interpolation='nearest')
            viewer.add_image(Y_copy,colormap='inferno',interpolation='nearest')
            # viewer.layers[-1].visible = False
            for l in viewer.layers:
                l.refresh()
            viewer.add_image(mask_both * 1, colormap='blue',interpolation='nearest')

            viewer.window.resize(300, 300)
            
            napari.run()


        lsso = LassoSelector(ax=ax, onselect=onSelect)
    if plot_map :
        plt.show() 
    if not (path_to_save is None) :
        fig.savefig(path_to_save)
    
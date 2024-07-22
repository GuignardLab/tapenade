import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from skimage.measure import regionprops
from skimage.filters import threshold_otsu

import organoid.utils as utils


class SpatialCorrelationPlotter:
    def __init__(self, quantity_X, quantity_Y, 
                 mask=None, labels=None):

        if labels is not None:
            if mask is None:
                props = regionprops(labels, intensity_image=quantity_X)
                quantity_X = np.array([prop.mean_intensity for prop in props])
                props = regionprops(labels, intensity_image=quantity_Y)
                quantity_Y = np.array([prop.mean_intensity for prop in props])
            else:
                props = regionprops(labels, intensity_image=quantity_X)
                quantity_X = np.array([
                    prop.mean_intensity for prop in props if np.any(mask[prop.slice][prop.image])
                ])
                props = regionprops(labels, intensity_image=quantity_Y)
                quantity_Y = np.array([
                    prop.mean_intensity for prop in props if np.any(mask[prop.slice][prop.image])
                ])
        elif mask is not None:
            quantity_X = quantity_X[mask]
            quantity_Y = quantity_Y[mask]
        else:
            quantity_X = quantity_X.ravel()
            quantity_Y = quantity_Y.ravel()


        self.quantity_X = quantity_X
        self.quantity_Y = quantity_Y
        self.mask = mask
        self.labels = labels

        self.xys = np.array([self.quantity_X, self.quantity_Y]).T

    def normalize_quantity(self, quantity):
        return (quantity - np.median(quantity)) / np.std(quantity)

    def get_heatmap_figure(
            self, 
            bins: tuple = (20, 20),
            show_individual_cells: bool = False,
            show_linear_fit: bool = True,
            normalize_quantities: bool = False,
            extent_X: tuple = None,
            extent_Y: tuple = None,
            percentiles_X: tuple = (0,100),
            percentiles_Y: tuple = (0,100),
            log_scale_X: bool = False,
            log_scale_Y: bool = False,
            figsize: tuple = (7, 4),
            label_X: str = 'X',
            label_Y: str = 'Y',
            colormap: str = 'plasma',
            sample_fraction: float = 0.005,
            display_quadrants: bool = False
    ):
        
        quantity_X = self.quantity_X.copy()
        quantity_Y = self.quantity_Y.copy()
        
        if normalize_quantities:
            quantity_Y = self.normalize_quantity(quantity_Y)
            quantity_X = self.normalize_quantity(quantity_X)
            log_scale_X = False
            log_scale_Y = False
        
        quantity_X, quantity_Y = utils.filter_percentiles(
            X=quantity_X, percentilesX=percentiles_X,
            Y=quantity_Y, percentilesY=percentiles_Y
        )
        
        if extent_X is None:
            extent_X = (np.min(quantity_X), np.max(quantity_X))
        if extent_Y is None:
            extent_Y = (np.min(quantity_Y), np.max(quantity_Y))

        bins = list(bins)
        fig, ax = plt.subplots(figsize=figsize)

        if log_scale_X:
            bins[0] = np.logspace(np.log10(extent_X[0]), np.log10(extent_X[1]), bins[0])
        if log_scale_Y:
            bins[1] = np.logspace(np.log10(extent_Y[0]), np.log10(extent_Y[1]), bins[1])

        heatmap, xedges, yedges = np.histogram2d(
            quantity_X.ravel(),
            quantity_Y.ravel(),
            bins=tuple(bins),
        )

        extent = extent_X + extent_Y
        
        custom_cmap = self._get_custom_cmap(colormap)

        # Create heatmap
        ims = ax.imshow(
            heatmap.T, origin='lower', cmap=custom_cmap, extent=extent,
            aspect='auto', interpolation='none'
        )
        cbar = fig.colorbar(ims, ax=ax)
        cbar.set_label('Number of cells', fontsize=10)

        if self.labels is not None:
            sample_indices = None
            self._add_individual_scatter(
                ax, normalize_quantities,
                show_individual_cells, ims, heatmap, xedges, yedges, custom_cmap
            )
        else:
            n = int(sample_fraction * len(self.quantity_X))
            sample_indices = np.random.choice(len(self.quantity_X), n, replace=False)

            self._add_individual_scatter(
                ax, normalize_quantities,
                show_individual_cells, ims, heatmap, xedges, yedges, custom_cmap,
                sample_indices=sample_indices
            )

        if display_quadrants:
            self._display_quadrants(ax, quantity_X, quantity_Y)


        if show_linear_fit:
            self._add_linear_fit(ax, quantity_X, quantity_Y, xedges)

        if not log_scale_X and not log_scale_Y:
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))


        self._format_figure(
            fig, ax,
            log_scale_X, log_scale_Y,
            label_X, label_Y,
            extent_X, extent_Y
        )

        return fig, ax
    
    def _get_custom_cmap(self, colormap: str):
        # changing colormap to get white for 0 values 
        inferno_cmap = plt.cm.get_cmap(colormap)
        inferno_colors = inferno_cmap(np.linspace(0, 1, 256))
        inferno_colors[0] = (1, 1, 1, 1)  # (R, G, B, Alpha)
        custom_cmap = LinearSegmentedColormap.from_list('custom_inferno', inferno_colors)

        return custom_cmap
    
    def _add_individual_scatter(self, ax, normalize_quantities,
                                show_individual_cells,
                                ims, heatmap, xedges, yedges, colormap,
                                sample_indices: list = None):
        
        if sample_indices is None:
            quantity_X = self.quantity_X
            quantity_Y = self.quantity_Y
        else:
            quantity_X = self.quantity_X[sample_indices]
            quantity_Y = self.quantity_Y[sample_indices]


        if normalize_quantities:
            quantity_X = self.normalize_quantity(quantity_X)
            quantity_Y = self.normalize_quantity(quantity_Y)

        indX = np.clip(np.digitize(quantity_X, xedges) - 1, 0, heatmap.shape[0] - 1)
        indY = np.clip(np.digitize(quantity_Y, yedges) - 1, 0, heatmap.shape[1] - 1)
        
        vals = heatmap[indX, indY]

        scatter = ax.scatter(
            quantity_X, 
            quantity_Y, 
            c=vals, cmap=colormap, s=3,
            vmin=heatmap.min(), vmax=heatmap.max()
        )

        if show_individual_cells:
            # change alpha of imshow
            ims.set_alpha(0.25)
        else:
            # change alpha of scatter
            scatter.set_alpha(0)
    
    def _add_linear_fit(self, ax, quantity_X, quantity_Y, xedges):
        intercept, slope, r2 = utils.linear_fit(
            x=quantity_X,
            y=quantity_Y,
            return_r2=True
        )
        
        x = np.array([xedges[0], xedges[-1]])
        lin_fit = intercept + slope*x
        ax.plot(x, lin_fit, c='red', lw=3, 
                label=f"Y = {slope:.1E} X + {intercept:.1E}, R²={r2:.2f}")
        ax.legend(fontsize=10)

    def _display_quadrants(self, ax, quantity_X, quantity_Y):

        threshold_X = threshold_otsu(quantity_X)
        threshold_Y = threshold_otsu(quantity_Y)

        ax.axvline(threshold_X, color='black', linestyle='--')
        ax.axhline(threshold_Y, color='black', linestyle='--')

    def _format_figure(
            self, fig, ax,
            log_scale_X, log_scale_Y,
            label_X, label_Y,
            extent_X, extent_Y
    ):

        if log_scale_X:
            ax.set_xscale('log')
        if log_scale_Y:
            ax.set_yscale('log')

        ax.set_xlim(extent_X)
        ax.set_ylim(extent_Y)

        xlim = list(ax.get_xlim())
        ylim = list(ax.get_ylim())

        x_diff = xlim[1] - xlim[0]
        xlim[0] = xlim[0] - 0.025 * x_diff 
        xlim[1] = xlim[1] + 0.025 * x_diff
        ax.set_xlim(xlim)

        y_diff = ylim[1] - ylim[0]
        ylim[0] = ylim[0] - 0.025 * y_diff
        ylim[1] = ylim[1] + 0.025 * y_diff
        ax.set_ylim(ylim)

        if not log_scale_X and not log_scale_Y:
            ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))


        matplotlib.rc('ytick', labelsize=10) 
        matplotlib.rc('xtick', labelsize=10)

        ax.set_xlabel(label_X, fontsize=12)
        ax.set_ylabel(label_Y, fontsize=12)

        fig.tight_layout()

    def plot_heatmap(self, bins: tuple = (20, 20),
                    show_individual_cells: bool = False,
                    show_linear_fit: bool = True,
                    extent_X: tuple = None,
                    extent_Y: tuple = None,
                    extent_percentiles: tuple = None,
                    log_scale_X: bool = False,
                    log_scale_Y: bool = False,
                    display_quadrants: bool = False):
        
        fig, ax = self.get_heatmap_figure(
            bins=bins,
            show_individual_cells=show_individual_cells,
            show_linear_fit=show_linear_fit,
            extent_X=extent_X,
            extent_Y=extent_Y,
            extent_percentiles=extent_percentiles,
            log_scale_X=log_scale_X,
            log_scale_Y=log_scale_Y,
            display_quadrants=display_quadrants
        )
        
        fig.show()
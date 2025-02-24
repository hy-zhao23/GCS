from utils.logging import log_info
import os
import matplotlib.pyplot as plt
from utils.settings import LAYERS, CONCEPTS, OBSERVED_NUM
from utils.ProcessPlotData import *
import numpy as np
from matplotlib import font_manager


# Add font file
font_path = os.path.join(os.path.expanduser("~"), ".local/share/fonts/lato/Lato-Regular.ttf")
font_manager.fontManager.addfont(font_path)

# Set font family globally
plt.rcParams['font.family'] = 'Lato'
# Make sure to set the figure up with a font that supports these Unicode characters
plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] 


def one_hist(
        layer: int,
        file: str,
        fig_size = (10, 10),
        colors: dict[str, str] = {'Observed': '#43d4ea', 'Sampled': '#ea437d', 'O-S': '#eab043'},
        x_range: tuple = (0.6, 1.0), y_range: tuple = (0.0, 0.8),
        n_bins: int = 101
    ) -> None:
    
    try:
        sim_hist = load_sim_hist(file, layer)
        
        n_plots, n_rows, n_cols = get_rows_cols_size()
        # fig_size = get_fig_size(fig_size)
        bins = np.linspace(*x_range, n_bins)

        if layer in range(33, 40):
            log_info(f'test2: Plot histogram for layer: {layer}')
        # Create the main figure with all subfigures
        fig, axs = plt.subplots(n_rows, n_cols, figsize=fig_size, squeeze=False)
        ax = axs.flatten()

        if layer in range(33, 40):
            log_info(f'test3: Plot histogram for layer: {layer}')
        for idx, (xo, xs, xos) in enumerate(sim_hist):
            # plot the histogram for all concepts within each layer
            plot_one_hist_subfig(ax[idx], idx, xo, xs, xos, bins, colors, x_range, y_range)
            # save the histogram for per concepts per layer separately
            plot_one_hist_subfig_save(layer, idx, xo, xs, xos, bins, colors, x_range, y_range)

        # Remove any unused subplots
        remove_unused_fig(n_plots, n_rows, n_cols, axs, fig)
        # Apply tight layout
        fig.tight_layout(pad=0.5, h_pad=0.5, w_pad=0.5)

        desc = 'hist'
        name = f'One-hist-{len(CONCEPTS)}-{layer}.pdf'
        save_fig(fig, desc, name)
    except Exception as e:
        log_error(f"Error in one_hist: {str(e)}")

def cos_sim_heatmap(sim_mats: dict, desc: str, diag=True, fig_size=(15,30)):
    o_sim_mat, s_sim_mat = load_sim_mat(sim_mats)

    plot_heatmap(o_sim_mat, f'{desc}-o', diag)
    plot_heatmap(s_sim_mat, f'{desc}-s', diag)

def distance_heatmap(sim_mats: dict, desc: str, diag=False, fig_size=(15,30)):
    """
    Plot heatmaps for the distance matrices of each layer.
    """
    plot_heatmap(sim_mats, f'{desc}', diag)

# Plot concept accuracies for per concept per layer and save separately/together
def plot_concept_accuracies(
        data: dict, 
        label: str, val_names: list, x_label: 'Layers', y_label: 'Accuracy',
        figsize=(12, 8),
        colors: list =None, markers: list =None
    ):
    # Generate color schemes for 8 categories, with 4 shades in each category
    colors, markers = get_accuracy_colors_markers(colors, markers)
    figsize = get_fig_size(figsize)
    cnt, n_rows, n_cols = get_rows_cols_size()

    try:
        # set up main figure configuration
        fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False) 

        for i, c in enumerate(CONCEPTS):
            row, col = get_index_row_col(i, n_cols)
            ax = axs[row, col]
            
            if not (num := check_markers_availability(data[c][LAYERS[0]], markers)):
                return
            
            accuracies = get_accuracy_vals(data[c], num, c)         
            plot_line_subfig(accuracies, c, val_names, x_label, y_label, num, markers, colors, ax)
            plot_line_subfig_save(accuracies, c, label, val_names, x_label, y_label, num, markers, colors)
            log_info(f"Created subplot for concept {c} at position ({row}, {col})")
                
    except Exception as e:
        log_info(f"An error occurred while accessing the data: {str(e)}")
    
    remove_unused_fig(cnt, n_rows, n_cols, axs, fig)
    
    plt.tight_layout()
    desc = "linear"
    name = f"{label}-o-s-{OBSERVED_NUM}.pdf" 
    save_fig(fig, desc, name)


def plot_layer_pca(layer: int, reduced_data: list, num: int, label: str):
    
    try:
        fig, ax = plt.subplots(figsize=(4, 3))
        num = len(reduced_data) // len(CONCEPTS)
        
        pca_plot_scatter_points(ax, reduced_data, CONCEPTS, num, layer)
        
        desc = 'pca'
        name = f'pca_{label}_layer_{layer}.pdf'
        save_fig(fig, desc, name)
    except Exception as e:
        print(f"Error creating PCA plot: {str(e)}")
        # Optionally, you can add more specific error handling or logging here
import os
from utils.settings import FIG_DIR, CONCEPTS, LAYERS, OBSERVED_NUM
from utils.files import read_pkl
from utils.logging import log_info, log_error, log_warning
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
from matplotlib.lines import Line2D


title_size = 10
legend_size = 6
normal_size = 8
font_weight = 'bold'
dpi = 100
SHOW = False

# create dir and get path for the saved file
def new_fig_dir(label: str, name: str):
    dir = os.path.join(FIG_DIR, f"{label}")
    os.makedirs(dir, exist_ok=True)
    return os.path.join(dir, name) 

def save_fig(fig, label: str, name: str):    
    try:
        path = new_fig_dir(label, name)
        fig.savefig(path, dpi=dpi, format='pdf', bbox_inches='tight', pad_inches=0.1)
        log_info(f"Successfully saved {path}")

        if SHOW:
            plt.show()
        plt.close(fig)

    except Exception as e:
        log_error(f"Error saving {name}: {str(e)}")

def set_hist_legend(ax: plt.Axes, labels: list, colors: list):
    legend_elements = [Rectangle((0, 0), 1, 1, facecolor=colors[label], alpha=0.7) for label in labels]
    legend_labels = [label for label in labels]
    legend = ax.legend(legend_elements, legend_labels, loc='upper left', fontsize=legend_size, handlelength=1, handleheight=1)
    legend.get_frame().set_facecolor('none')  # Transparent background


def plot_one_hist_subfig(ax: plt.Axes, idx: int, xo: list, xs: list, xos: list, bins: np.ndarray, colors: list, x_range: tuple, y_range: tuple):
    try:
        data_sets = {'Observed': xo, 'Sampled': xs, 'O-S': xos}
        labels = []
        for label, data in data_sets.items():
            if data is not None:
                labels.append(label)
                weights = np.ones_like(data) / len(data)
            ax.hist(data, bins=bins, weights=weights, color=colors[label], alpha=0.7, label=label)
    
        ax.set_title(f"{CONCEPTS[idx]}", fontsize=12, fontweight='bold')
        ax.set_xlim(*x_range)
        ax.set_ylim(*y_range)
        ax.tick_params(axis='both', which='major', labelsize=legend_size)
        ax.set_xlabel("Cosine Similarity", fontsize=normal_size, fontweight='bold')
        ax.set_ylabel("Proportion", fontsize=normal_size, fontweight='bold')

        set_hist_legend(ax, labels, colors)        
        # Format y-axis as proportion
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
    except Exception as e:
        log_error(f"Error in plot_one_hist_subfig: {str(e)}")


def plot_one_hist_subfig_save(layer: int, idx: int, xo: list, xs: list, xos: list, bins: np.ndarray, colors: list, 
                              x_range: tuple, y_range: tuple):
    try:
        # Create a new figure with a single subplot
        fig, ax = plt.subplots(figsize=(3, 2.5))  
        plot_one_hist_subfig(ax, idx, xo, xs, xos, bins, colors, x_range, y_range)

        # Adjust layout
        fig.tight_layout()
        
        # Save the single subfigure
        desc = 'hist'
        name = f'hist-{CONCEPTS[idx]}-{layer}.pdf'
        save_fig(fig, desc, name)
    except Exception as e:
        log_error(f"Error in plot_one_hist_subfig_save for {CONCEPTS[idx]} in layer {layer}: {str(e)}")
    finally:
        plt.close(fig)


def load_sim_hist(file, layer):
    sim_hist = None
    # Constantly read file until it is not None
    while sim_hist is None:
        log_info(f"Waiting for data in {file} to be available...")
        time.sleep(1)  # Wait for 1 second before trying again
        sim_hist = read_pkl(file)
        sim_hist = sim_hist[layer]

    return sim_hist 

def get_accuracy_colors_markers(colors: list, markers: list):
    if not colors:
        colors = plt.cm.viridis(np.linspace(0, 1, 6))

    if not markers:
        # Default markers list
        markers = ['^', 'o', 's', 'D', 'v', 'x']

    return colors, markers

def get_fig_size(figsize:set):
    fig_size = (figsize[0], figsize[0])
    return fig_size

def get_rows_cols_size(n_plots = None):
    if n_plots is None:
        n_plots = len(CONCEPTS)
    
    n_cols = min(4, n_plots)  # Max 3 columns
    n_rows = (n_plots - 1) // n_cols + 1

    return n_plots, n_rows, n_cols

# return idex of row and col
def get_index_row_col(idx: int, n_cols: int):
    return idx // n_cols, idx % n_cols

def check_markers_availability(vals: list, markers: list):
    if not vals:
        log_error(f'Accuracy value is empty!')
        raise ValueError(f"No accuracy values found for this concept and layer")
    if len(vals) > len(markers):
        log_error(f'Markers are not enough, please redefine!')
        raise ValueError(f"Markers are not enough, please redefine!")
    return len(vals)

def get_accuracy_vals(vals: list, num: int, concept: str):
    accuracies = [[] for _ in range(num)]
    for l in LAYERS:
        for k, acc in enumerate(vals[l]):
            if not acc:
                log_error(f"Accuracy value is empty for concept {concept}, layer {l}, sigma {k}")
                raise ValueError(f"Empty accuracy value encountered for concept {concept}, layer {l}, sigma {k}")
            accuracies[k].append(acc) 
    return accuracies

def set_line_legend(ax: plt.Axes, labels: list, colors: list, markers: list):
    legend_labels = [label for label in labels]
    legend_elements = []

    for k in range(len(labels)):
        legend_elements.append(
            Line2D([0], [0], color=colors[k % len(colors)], lw=0.7, marker=markers[k % len(markers)], markersize=1.2, alpha=0.5)
        )

    legend = ax.legend(legend_elements, legend_labels, fontsize=legend_size)
    legend.get_frame().set_facecolor('none')  # Transparent background

# plot concept-level subfigure
def plot_line_subfig(accuracies: list, concept: str, val_names: list, x_label: str, y_label: str, num: int, markers: list, colors: list, ax: plt.Axes):
    try:
        for k in range(num):
            ax.plot(LAYERS, accuracies[k], marker=markers[k % len(markers)], markersize=1.2, color=colors[k % len(colors)], linestyle='-', linewidth=0.7, alpha=0.5)
            
            ax.set_title(f"{concept}", fontsize=title_size, fontweight=font_weight)
            ax.set_xlabel(x_label, fontsize=normal_size, fontweight='bold')
            ax.set_ylabel(y_label, fontsize=normal_size, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=legend_size)
            ax.grid(True, color='lightgray')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            
            set_line_legend(ax, val_names, colors, markers)
    except Exception as e:
        log_error(f"Error in plotting concept subfigure: {str(e)}")
        raise

def plot_line_subfig_save(accuracies: list, concept: str, label:str, val_names: list, x_label: str, y_label: str, num: int, markers: list, colors: list):
    try:
        # Remove the unused subplots for the single figure
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        plot_line_subfig(accuracies, concept, val_names, x_label, y_label, num, markers, colors, ax)
        
        # Adjust layout
        fig.tight_layout()
        
        # Save the single subfigure
        desc = 'linear'
        name = f'{label}-{concept}-o-s-{OBSERVED_NUM}.pdf'
        save_fig(fig, desc, name)
    except Exception as e:
        log_error(f"Error in plot_line_subfig_save for {concept}: {str(e)}")
    finally:
        plt.close(fig)

# Remove any unused subplots
def remove_unused_fig(cnt: int, n_rows: int, n_cols: int, axs: np.ndarray, fig):
    try:
        for i in range(cnt, n_rows * n_cols):
            row = i // 4
            col = i % 4
            fig.delaxes(axs[row, col])
    except Exception as e:
        log_error(f"Error in removing unused subplots: {str(e)}")
        raise

def get_sim_vmin_vmax(sim_mat: np.ndarray, diag: bool):
    vmin = np.min(sim_mat)
    vmax = min(np.max(sim_mat), 0.5)
    if not diag:
        no_diag = sim_mat[~np.eye(sim_mat.shape[0], dtype=bool)]
        vmin = np.min(no_diag)
        vmax = np.max(no_diag)

    return vmin, vmax

def load_sim_mat(sim_mat: dict):
    o_sim_mat, s_sim_mat = {}, {}
    for l in LAYERS:
        o_sim_mat[l], s_sim_mat[l] = sim_mat[l]

    return o_sim_mat, s_sim_mat

def plot_heatmap_subfig(ax: plt.Axes, sim_mat: np.ndarray, layer: int, diag=True):
    try:
        # Format the annotations to display 2 decimal places
        annot = np.round(sim_mat, decimals=2)
        vmin, vmax = get_sim_vmin_vmax(sim_mat, diag)
        if not diag:
            vmax = vmax/3 * 2
        # Rotate and align the tick labels
    
        # ax.tick_params(axis='x', labelsize=normal_size, rotation=45)
        # for label in ax.get_xticklabels():
        #     label.set_ha('right')

        # ax.tick_params(axis='y', labelsize=normal_size, rotation=45)
        # for label in ax.get_yticklabels():
        #     label.set_va('top')
  
        label = 'Cosine similarity' if diag else 'Distribution distance'
        ax.set_title(f'{label} among {len(CONCEPTS)} linear concept vectors at layer {layer}', fontsize=title_size, pad=20, fontweight='bold')

        cmap = 'coolwarm' if diag else 'coolwarm_r'
        heatmap = sns.heatmap(sim_mat, annot=annot, fmt='.2f', cmap=cmap, cbar=True, xticklabels=CONCEPTS, yticklabels=CONCEPTS, vmin=vmin, vmax=vmax, annot_kws={'fontsize': legend_size}, ax=ax)
        
        # Rotate and align the tick labels after creating the heatmap
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=45, va='top', rotation_mode='anchor')
        
        # Adjust tick label size
        ax.tick_params(axis='both', which='major', labelsize=normal_size)

        # Adjust the width of the colorbar
        cbar = heatmap.collections[0].colorbar
        cbar.ax.set_box_aspect(40)  # This makes the colorbar thinner

        # Set evenly distributed colorbar ticks
        num_ticks = 5  # You can adjust this number as needed
        tick_locator = plt.LinearLocator(numticks=num_ticks)
        cbar.locator = tick_locator
        cbar.update_ticks()

        cbar.ax.tick_params(labelsize=legend_size)

        if not diag:
            cbar.ax.set_yticklabels([f'{x:.2f}' for x in cbar.get_ticks()])
    except Exception as e:
        log_error(f"Error in plotting heatmap subfigure: {str(e)}")

# Plot heatmap for each layer and save separately
def plot_heatmap_subsifg_save(sim_mat: dict, desc: str, layer: int, diag=True) -> None:
    fig, ax = plt.subplots(figsize=(5.5,5))
    plot_heatmap_subfig(ax, sim_mat, layer, diag)
    
    # Adjust layout to prevent clipping of tick-labels
    fig.tight_layout()
    label = 'heatmap'
    name = f'{desc}-{layer}.pdf'
    save_fig(fig, label, name)

def plot_heatmap(sim_mat: dict, desc: str, diag=True, fig_size=(25,50)):
    """
    Plot heatmap for each layer and save separately and together
    """
    try:
        n_plots, n_rows, n_cols = get_rows_cols_size(len(sim_mat))
        fig, axs = plt.subplots(n_rows, n_cols, figsize=fig_size, squeeze=False)
        ax = axs.flatten()

        for i, (layer, mat) in enumerate(sim_mat.items()):
            plot_heatmap_subsifg_save(mat, desc, layer, diag=diag)
            plot_heatmap_subfig(ax[i], mat, layer, diag=diag)

        remove_unused_fig(n_plots, n_rows, n_cols, axs, fig)

        plt.tight_layout()

        label = 'heatmap'
        name = f'{desc}-all.pdf'
        save_fig(fig, label, name)
    except Exception as e:
        log_error(f"Error in plotting heatmap: {str(e)}")
        raise

def pca_get_colors(num_concepts: int, group_size=4):
    custom_colors = []
    num_groups = (num_concepts + group_size - 1) // group_size  # Round up division
    
    # Generate colors for each group
    group_colors = plt.cm.viridis(np.linspace(0, 1, num_groups))
    
    # Assign colors to concepts
    for i in range(num_concepts):
        group_index = i // group_size
        custom_colors.append(group_colors[group_index])
    
    return custom_colors

def pca_plot_scatter_points(ax: plt.Axes, reduced_data, CONCEPTS, num, layer):
    try:
        custom_colors = pca_get_colors(len(CONCEPTS))
        # Define a list of marker sizes to cycle through
        marker_sizes = [15, 30, 45, 60]
        
        for i, concept in enumerate(CONCEPTS):
            start = i * num
            end = (i + 1) * num
            points = reduced_data[start:end]
            
            # Determine number and marker size (cycling through 1-4)
            size = marker_sizes[i % 4]
            
            # Plot scatter points
            scatter = ax.scatter(points[:, 0], points[:, 1], 
                                 c=[custom_colors[i]], 
                                 s=size,  # Cycle through sizes
                                 edgecolors='black', 
                                 linewidths=1,
                                 alpha=0.5,
                                 label=concept)  # Remove number from label

        pca_set_plot_labels(ax, reduced_data, layer, num)
        
    except Exception as e:
        log_error(f"Error in pca_plot_scatter_points: {e}")

def pca_set_plot_labels(ax, reduced_data, layer, num_concepts):
    try:
        # Set axis limits with some padding
        x_min, x_max = reduced_data[:, 0].min(), reduced_data[:, 0].max()
        y_min, y_max = reduced_data[:, 1].min(), reduced_data[:, 1].max()
        x_padding = (x_max - x_min) * 0.1
        y_padding = (y_max - y_min) * 0.1
        ax.set_xlim(x_min - x_padding, x_max + x_padding)
        ax.set_ylim(y_min - y_padding, y_max + y_padding)
        ax.tick_params(axis='both', which='major', labelsize=normal_size)

        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(vmin=0, vmax=num_concepts))
        sm.set_array([])

        # Add grid for better readability
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=legend_size)

        ax.set_title(f'PCA Visualization of Concept Vectors at Layer {layer}', fontsize=title_size, fontweight='bold')
        ax.set_xlabel('First Principal Component', fontsize=normal_size, fontweight='bold')
        ax.set_ylabel('Second Principal Component', fontsize=normal_size, fontweight='bold')

        # Adjust layout
        plt.tight_layout()  # Adjusted rect to accommodate colorbar
    except Exception as e:
        log_error(f"Error in pca_set_plot_labels: {e}")
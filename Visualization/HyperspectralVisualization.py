from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib


class HyperspectralVisualizer:
    """
    This is a class that handles Hyperspectral Data Visualization

    Input: Hyperspectral Image Cube hcube: (nx, ny, nz)
    Alternative Input: Hyperspectral Image himg: (nx, ny)
    Output: matplotlib Figure fig
    """

    def __init__(self, hcube=None, himg=None):
        # If an himg is inputted, prioritize that and reshape it
        if himg is not None:
            self.nx, self.ny = himg.shape
            self.nz = 1
            # Reshapes image into 3d hypercube with 1 layer
            self.hcube = himg.reshape(self.nx, self.ny, self.nz)
        else:
            self.nx, self.ny, self.nz = hcube.shape
            self.hcube = hcube

    def layer_plot(self, title='Default Title', subtitle='Default Subtitle', display_figure_labels=False,
                       figure_labels=None, font_size=8, subplot_dims=None, color_map='viridis',
                       color_map_bounds=None, colormap_discrete = False, colormap_num_discrete = None) -> matplotlib.figure:
        """
        Creates a 2D plot of a Hyperspectral Cube displaying the layers along the n_z axis
        :return fig: A matplotlib figure
        :param title: Main Title (Default: "Default Title")
        :param subtitle: Subtitle (Default: "Default Subtitle")
        :param display_figure_labels: Whether to display figure labels or not
        :param figure_labels: Figure Labels as a list of strings for the number of plots needed, otherwise, display
                              figure labels as "Layer x" instead
        :param font_size: Font Size (Default: 8pt)
        :param subplot_dims: Desired subplot dimensions in a tuple (dim_x,dim_y). The tuple must accommodate the size
                             of x.shape[2] (Default: None)
        :param color_map: Known matplotlib colormap (Default: "viridis")
        :param color_map_bounds: Bounds of the Colormap in a tuple (lb,ub), if no value is given, maximum/minimum values
                                 will be chosen from the cube
        :param colormap_discrete: Display a discrete colormap (Default: False)
        :param colormap_num_discrete: Number of discrete colormap values                   
        """
        # Setting matplotlib style config to fast
        mplstyle.use("fast")
        # Setting overall font-size and colormap for the plot
        plt.rcParams.update({'font.size': font_size})

        cmap = matplotlib.cm.get_cmap(color_map)
        if colormap_discrete and colormap_num_discrete is not None:
            cmap = matplotlib.cm.get_cmap(color_map, colormap_num_discrete)

        # If no subplot dimensions are specified/ are not adequate, create new dimensions
        if (subplot_dims is None) or (subplot_dims[0] * subplot_dims[1] < self.nz):
            # Manually calculating the dimensions of the subplot
            dim_x, dim_y = 1, 1
            while dim_x * dim_y < self.nz:
                if dim_y <= dim_x:
                    dim_y += 1
                else:
                    dim_x += 1
        else:
            dim_x, dim_y = subplot_dims

        # If no colormap_bounds are specified, create new ones
        if color_map_bounds is None:
            color_map_bounds = (float(np.amin(self.hcube)), float(np.amax(self.hcube)))

        # Defining a new figure with subplots in a (subplot_dims[0], subplot_dims[1]) grid
        fig, axes = plt.subplots(nrows=dim_x, ncols=dim_y, squeeze=False, constrained_layout=True)
        # Looping through each subplot, stopping adding photos when plots outnumber n_z
        n_plotted = 0
        for i in range(axes.shape[0]):
            for j in range(axes.shape[1]):
                if n_plotted < self.nz:
                    # Adding the images to the subplot and setting a title
                    im = axes[i, j].imshow(self.hcube[:, :, n_plotted], vmin=color_map_bounds[0],
                                           vmax=color_map_bounds[1], cmap = cmap)
                    # Setting a subplot figure label if desired
                    if display_figure_labels:
                        if figure_labels is not None:
                            axes[i, j].set_title(figure_labels[n_plotted])
                        else:
                            axes[i, j].set_title("Layer" + " " + str(n_plotted + 1))
                    n_plotted += 1
                else:
                    axes[i, j].set_axis_off()
        # Display title and subtitle and colormaps
        plt.suptitle(title + "\n" + subtitle)
        fig.colorbar(im, ax=axes.ravel().tolist())
        return fig

    def cube_plot(self, title="Default Title", subtitle="Default Subtitle", color_map="viridis"):
        """
        Plots a 3D cube using surface plots along the n_z axis
        :return fig: A matplotlib figure
        :param title: Main Title (Default: "Default Title")
        :param subtitle: Subtitle (Default: "Default Subtitle")
        :param color_map: Known matplotlib colormap (Default: "viridis")
        """
        # Sets the plotting speed to fast and grabs the builtin matplotlib colormap
        mplstyle.use('fast')
        cmap = plt.cm.get_cmap(color_map)

        # Creating a mesh grid to map every single pixel in one 2D layer
        mesh_x, mesh_y = np.meshgrid(np.arange(self.nx + 1)[::-1], np.arange(self.ny + 1))
        # Creating and populating a figure
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_box_aspect([1, 1, 1])
        # for each layer in the cube,
        for i in range(self.nz):
            # Creating the flat height matrix for the layer
            mesh_z = ((self.nz - i) / self.nz) * np.ones((self.ny + 1, self.nx + 1))
            im = ax.plot_surface(mesh_y, mesh_x, mesh_z, shade=False, cstride=2, rstride=2,
                                 facecolors=cmap(np.reshape(self.hcube[:, :, i], (self.nx, self.ny)).T))
        # Displaying the titles and color maps
        plt.suptitle(title + "\n" + subtitle)
        fig.colorbar(im, ax=ax)
        return fig

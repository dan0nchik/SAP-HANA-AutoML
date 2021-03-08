"""
This module represents a visualizer.  Matplotlib is used for all visualizations
"""
#pylint: disable=too-many-lines

import logging
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__) #pylint: disable=invalid-name

class Visualizer(object):
    """
    Base class for all visualizations.
    It stores the axes, size, title and other drawing parameters.

    Parameters
    ----------
    ax : matplotlib.Axes, optional
        The axes to use to plot the figure.
        Default value : Current axes
    size : tuple of integers, optional
        (width, height) of the plot in dpi
        Default value: Current size of the plot
    cmap : plt.cm, optional
        The color map used in the plot.
        Default value: plt.cm.Blues
    title : str, optional
        This plot's title.
        Default value : Empty str

    """

    def __init__(self, ax=None, size=None, cmap=None, title=None): #pylint: disable=invalid-name
        self.set_ax(ax)
        self.set_title(title)
        self.set_cmap(cmap)
        self.set_size(size)

    ##////////////////////////////////////////////////////////////////////
    ## Primary Visualizer Properties
    ##////////////////////////////////////////////////////////////////////
    @property
    def ax(self): #pylint: disable=invalid-name
        """
        Returns the matplotlib Axes where the Visualizer will draw.
        """
        return self._ax

    def set_ax(self, ax):
        """
        Sets the Axes
        """
        if ax is None:
            self._ax = plt.gca()
        else:
            self._ax = ax

    @property
    def size(self):
        """
        Returns the size of the plot in pixels.
        """
        return self._size

    def set_size(self, size):
        """
        Sets the size
        """
        if size is None:
            fig = plt.gcf()
        else:
            fig = plt.gcf()
            width, height = size
            fig.set_size_inches(width / fig.get_dpi(), height / fig.get_dpi())
        self._size = fig.get_size_inches()*fig.dpi

    @property
    def title(self):
        """
        Returns the title of the plot.
        """
        return self._title

    def set_title(self, title):
        """
        Sets the title of the plot
        """
        if title is not None:
            self._title = title
        else:
            self._title = ""

    @property
    def cmap(self):
        """
        Returns the color map being used for the plot.
        """
        return self._cmap

    def set_cmap(self, cmap):
        """
        Sets the colormap
        """
        if cmap is None:
            self._cmap = plt.cm.get_cmap('Blues')

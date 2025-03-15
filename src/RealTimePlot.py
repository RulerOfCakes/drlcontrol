import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import numpy as np
import time

class MultiRealTimePlot:
    def __init__(self):
        self.plots = {}
        self.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    
    def add_plot(self, key, xlim=(0, 5), ylim=(0, 0), xlabel="X value", ylabel="Y value", title=""):
        win = pg.GraphicsLayoutWidget(title=title)
        win.show()
        
        p = win.addPlot(title=title)
        p.setLabel('bottom', xlabel)
        p.setLabel('left', ylabel)
        p.setXRange(xlim[0], xlim[1])
        p.setYRange(ylim[0], ylim[1])
        
        curve = p.plot([], [], pen='b')
        
        self.plots[key] = {
            "win": win,
            "plot": p,
            "curve": curve,
            "xdata": [],
            "ydata": [],
            "xlim": list(xlim),
            "ylim": list(ylim)
        }
    
    def add_value(self, key, x, y):
        assert key in self.plots
        
        plot_info = self.plots[key]
        plot_info["xdata"].append(x)
        plot_info["ydata"].append(y)
        
        plot_info["curve"].setData(plot_info["xdata"], plot_info["ydata"])
        
        current_xmin, current_xmax = plot_info["plot"].getViewBox().viewRange()[0]
        if x > current_xmax:
            plot_info["xlim"][1] = x + 5
            plot_info["plot"].setXRange(plot_info["xlim"][0], plot_info["xlim"][1])
        
        new_min = min(plot_info["ydata"])
        new_max = max(plot_info["ydata"])
        margin = (new_max - new_min) * 0.1 if new_max != new_min else 0.1
        plot_info["ylim"][0] = new_min - margin
        plot_info["ylim"][1] = new_max + margin
        plot_info["plot"].setYRange(plot_info["ylim"][0], plot_info["ylim"][1])
        
        self.app.processEvents()
    
    def run(self):
        self.app.exec_()

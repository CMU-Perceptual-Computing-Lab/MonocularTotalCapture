import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import utils.general


class vis_heatmap3d(object):
    def __init__(self, fig, ax, heatmap, keypoints=None, type_str=None):
        assert len(heatmap.shape) == 4
        self.fig = fig
        self.idx = 0
        self.threshold = 0.5
        self.heatmap = heatmap
        self.ax = ax
        self.keypoints = keypoints
        self.type_str = type_str

        axcolor = 'lightgoldenrodyellow'
        axx = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)

        self.slider_threshold = Slider(axx, 'threshold', 0.0, 1.0, valinit=0.5)
        self.slider_threshold.on_changed(self.update)

    def draw(self):
        self.ax.clear()
        if self.keypoints is not None:
            utils.general.plot3d(self.ax, self.keypoints, self.type_str)
        active_map = self.heatmap[:, :, :, self.idx]
        Z, Y, X = np.where(active_map >= self.threshold)
        colors = [(1 - s) * np.array([0., 0., 1.], dtype=float) for s in active_map[Z, Y, X]]
        self.ax.scatter(X, Y, Z, color=colors)

    def update(self, val):
        self.threshold = self.slider_threshold.val
        self.draw()
        self.fig.canvas.draw_idle()

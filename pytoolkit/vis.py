import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time

matplotlib.rcParams['font.sans-serif'] = "Helvetica"
matplotlib.rcParams['font.family'] = "sans-serif"

class fig_handle:
    def __init__(self, title='title', xlabel='xlabel', ylabel='ylabel', len=None):
        self.fig = plt.figure()
        self.fig.show()
        self.len = len

        plt.grid(linestyle='dotted')
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        ax = self.fig.gca()
        ax.plot([], [], 'o-', color=(0.4,0.5,1.0), lw=0.8, mew=0.8, ms=5, mfc='none')

    def update(self, x, y):
        fig = self.fig
        ax = fig.gca()
        line = ax.lines[0]
        x_data = np.append(line.get_xdata(), x)
        y_data = np.append(line.get_ydata(), y)
        if self.len is not None:
            x_data = x_data[-self.len:]
            y_data = y_data[-self.len:]

        xr = x_data.max() - x_data.min() + 0.001
        yr = y_data.max() - y_data.min() + 0.001
        s = 0.05
        ax.set_xlim(x_data.min() - s * xr, x_data.max() + s * xr)
        ax.set_ylim(y_data.min() - s * yr, y_data.max() + s * yr)

        line.set_data(x_data, y_data)
        fig.canvas.draw()
        fig.canvas.flush_events()
        time.sleep(0.05)

    def savefig(self):
        plt.savefig('final_fig.pdf')


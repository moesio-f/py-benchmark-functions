from typing import Union

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from py_benchmark_functions import core, get_fn
from py_benchmark_functions.imp import numpy as npf


class Drawer:
    def __init__(self, function: Union[core.Function, str], resolution=80):
        if isinstance(function, str):
            function = get_fn(function, 2)
        self._set_mesh(function, resolution)

    def clear(self):
        self._ax.clear()

    def draw_mesh(self, show: bool = False, save: bool = True, **kwargs):
        self.clear()
        self._ax.set_xlabel(r"$x_1$", fontsize=8)
        self._ax.set_ylabel(r"$x_2$", fontsize=8)
        self._ax.set_zlabel(r"$f(x_1, x_2)$", fontsize=8)
        self._ax.plot_surface(
            self._mesh[0],
            self._mesh[1],
            self._mesh[2],
            rstride=1,
            cstride=1,
            cmap=self._cmap,
            linewidth=0.0,
            shade=True,
            **kwargs,
        )

        plt.contour(
            self._mesh[0],
            self._mesh[1],
            self._mesh[2],
            zdir="z",
            offset=self._ax.get_zlim()[0],
            alpha=0.3,
        )

        if save:
            self._fig.savefig(f"plot-2d-{self._fn.name}")

        if show:
            plt.show()

    def close_fig(self):
        plt.close(self._fig)

    def _set_mesh(self, function: core.Function, resolution=80):
        if function.dims > 2:
            function = get_fn(function.name, 2)

        self._fn = function
        self._fig: plt.Figure = plt.figure()
        self._ax = self._fig.add_subplot(projection="3d")
        self._resolution = resolution

        # creating mesh
        linspace = np.linspace(
            self._fn.domain.min, self._fn.domain.max, self._resolution
        )
        X, Y = np.meshgrid(linspace, linspace)

        if isinstance(self._fn, npf.NumpyFunction):
            zs = [np.array([x, y]) for x, y in zip(np.ravel(X), np.ravel(Y))]
            Z = np.array([self._fn(v) for v in zs]).reshape(X.shape)
        else:
            # fn is TF Function
            import tensorflow as tf

            zs = [
                tf.constant([x, y])
                for x, y in zip(
                    np.ravel(X).astype(np.float32), np.ravel(Y).astype(np.float32)
                )
            ]
            Z = np.array([self._fn(v).numpy() for v in zs]).reshape(X.shape)

        self._cmap = cm.get_cmap("jet")
        self._mesh = (X, Y, Z)

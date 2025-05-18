from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np

from py_benchmark_functions import core, factory, get_np_function
from py_benchmark_functions.imp import numpy as npf


class Drawer:
    def __init__(self, function: Union[core.Function, str], resolution: int = 80):
        """Constructor.

        Args:
            function (Function, str): either an instance of a function
                or name of function. Internally, all functions are converted
                to 2D and with NumPy. The domain is set as the first two values
                for minimum and maximum (i.e., function.domain.min[:2] and
                function.domain.max[:2]).
            resolution (int): controls the fine-grained surface details, used
                by linspace. For saving, recommended value is >=60, for interactive
                plotting should be >=20 and <=40. Defaults to 80 for fine-grained
                surfaces.
        """
        # Guarantee it is a function
        if isinstance(function, str):
            function = get_np_function(function, 2)

        # If it isn't 2D or not NumPy
        if function.dims > 2 or not isinstance(function, npf.NumpyFunction):
            function = factory.Registry[function.name, "numpy"](
                dims=2,
                domain_min=function.domain.min[:2],
                domain_max=function.domain.max[:2],
            )

        # Initialize variables
        self._fn = function
        self._resolution = resolution
        self._cmap = plt.get_cmap("jet")
        self._fig = None
        self._ax = None
        self._mesh = None

        # Set _mesh variable
        self._set_mesh()

    def clear(self):
        if self._fig is not None:
            # Clear figure axes
            self._ax.clear()

    def draw_mesh(self, **surface_kwargs):
        # Maybe create figure
        if self._fig is None:
            self._fig = plt.figure()
            self._ax = self._fig.add_subplot(projection="3d")
        else:
            self.clear()

        # Draw surface
        self._ax.set_xlabel("$x_1$", fontsize=8)
        self._ax.set_ylabel("$x_2$", fontsize=8)
        self._ax.set_zlabel(f"{self._fn.name}$(x_1, x_2)$", fontsize=8)
        self._ax.plot_surface(
            self._mesh[0],
            self._mesh[1],
            self._mesh[2],
            rstride=1,
            cstride=1,
            cmap=self._cmap,
            linewidth=0.0,
            shade=True,
            **surface_kwargs,
        )

        # Draw contour
        plt.contour(
            self._mesh[0],
            self._mesh[1],
            self._mesh[2],
            zdir="z",
            offset=self._ax.get_zlim()[0],
            alpha=0.3,
        )

    def save(self, directory: Union[str, Path] = ""):
        # Maybe draw mesh
        if self._fig is None:
            self.draw_mesh()

        if isinstance(directory, str):
            directory = Path(directory)

        directory.mkdir(parents=True, exist_ok=True)
        self._fig.savefig(directory.joinpath(f"plot-2d-{self._fn.name}"))
        self.close()

    def show(self):
        # Maybe draw mesh
        if self._fig is None:
            self.draw_mesh()

        plt.show()
        self.close()

    def close(self):
        plt.close(self._fig)
        self._fig = None
        self._ax = None

    def _set_mesh(self):
        linspace = np.linspace(
            self._fn.domain.min, self._fn.domain.max, self._resolution
        )
        X, Y = np.meshgrid(linspace, linspace)
        zs = np.array([[x, y] for x, y in zip(np.ravel(X), np.ravel(Y))])
        Z = self._fn(zs).reshape(X.shape)
        self._mesh = (X, Y, Z)

    def __del__(self):
        self.close()

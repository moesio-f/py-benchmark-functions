"""Core classes for supported
transformations on functions.
"""

from typing import List, Union

from py_benchmark_functions.core.function import Domain, Function, Metadata


class Transformation(Function):
    """Transformation base class.

    Wraps a function and applies transformations
    with the following form:
        f(x) -> outer * f(inner*x + hshift) + vshift

    Different orders can be created by successively
        applying this class.
    """

    def __init__(
        self,
        fn: Function,
        vshift: float = 0.0,
        hshift: Union[float, List[float]] = 0.0,
        outer_scale: float = 1.0,
        inner_scale: Union[float, List[float]] = 1.0,
        has_same_domain: bool = False,
    ):
        # Store parameters
        self._vshift = vshift
        self._hshift = [hshift] * fn.dims if isinstance(hshift, float) else hshift
        self._outer_scale = outer_scale
        self._inner_scale = (
            [inner_scale] * fn.dims if isinstance(inner_scale, float) else inner_scale
        )

        # Store parent
        self._parent = fn

        # Initialize parent class
        domain = self._maybe_update_domain(fn.domain, has_same_domain)
        super().__init__(dims=fn.dims, domain=domain)

    @property
    def vshift(self) -> float:
        return self._vshift

    @property
    def hshift(self) -> List[float]:
        return self._hshift

    @property
    def outer_scale(self) -> float:
        return self._outer_scale

    @property
    def inner_scale(self) -> List[float]:
        return self._inner_scale

    @property
    def name(self) -> str:
        return f"Transformed({self.base.name})"

    @property
    def metadata(self) -> Metadata:
        return self.base.metadata

    @property
    def parent(self) -> Function:
        return self._parent

    @property
    def base(self) -> Function:
        base = self.parent
        while isinstance(base, Transformation):
            base = base.parent
        return base

    def _maybe_update_domain(self, domain: Domain, has_same_domain: bool):
        if has_same_domain:
            return domain

        return Domain(
            *(
                [(v - shift) / scale for v in d]
                for d, shift, scale in zip(domain, self.hshift, self.inner_scale)
            )
        )

    def __str__(self):
        return self.name
